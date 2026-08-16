"""Playback capability contract (issue #55) + the single media-servability seam.

Per-turn playback (issue #49) lets the operator seek the ``<audio>`` element to a
transcript segment or a diarization turn. Doing that on media that is missing,
reclaimed, or whose timeline is malformed would land the playhead on the wrong
voice — the exact harm #55 exists to prevent. So the console FAILS CLOSED: seek
is offered only when every precondition holds, and when it does not, an honest
banner lists *why*.

``resolve_servable_media`` is the ONE place that decides whether ``GET /media``
would actually serve bytes. Both the media route and this capability predicate
call it, so the capability can never advertise ``seekEnabled`` while ``/media``
would answer 404/410.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, BinaryIO

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.db.models import PipelineRun, TranscriptSegment
from voxint.media.reclaim import run_intermediate_reclaimed_at
from voxint.media.serving import MediaGate, MediaNotServableError
from voxint.pipeline.stages.context import StageDataError, normalized_audio_path

if TYPE_CHECKING:
    from voxint.config import Settings

# Fixed slack (seconds, NOT a fraction of duration) allowed between the last
# transcript ``end_seconds`` and the media duration. Whisper segment ends and the
# normalized-WAV duration are measured independently and routinely disagree by a
# few tens of milliseconds without any real out-of-bounds seek; 0.05s absorbs
# that float noise while still catching a timeline that genuinely overruns the
# recording. A percentage would grow the tolerance on long files, which is
# exactly where a stray end past the tail is most likely to be a real bug.
TAIL_TOLERANCE = 0.05


# --------------------------------------------------------------------------- #
# Media servability — the single source of truth shared with GET /media.
# --------------------------------------------------------------------------- #
class MediaResolutionError(Exception):
    """A run's processed audio cannot be served.

    Carries the capability ``code`` (surfaced verbatim to the operator via
    :func:`playback_capability`) and the HTTP status the ``/media`` route should
    answer with, so both consumers stay in lockstep.
    """

    code: str
    http_status: int

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class MediaReclaimed(MediaResolutionError):
    """The normalized-audio intermediate was GC'd (issue #15); row survives."""

    code = "media_reclaimed"
    http_status = 410


class MediaMissing(MediaResolutionError):
    """No (or not exactly one) preprocessed-audio artifact for the run."""

    code = "media_missing"
    http_status = 404


class MediaUnservable(MediaResolutionError):
    """The file exists on paper but the gate refuses to serve it."""

    code = "media_unservable"
    http_status = 404


def resolve_servable_media(
    session: Session,
    run_id: uuid.UUID,
    settings: Settings,
    gate: MediaGate,
) -> tuple[BinaryIO, int]:
    """Return an open ``(file object, size)`` for the run's servable audio.

    The single decision point for "would ``GET /media`` serve this?". Mirrors the
    media route's checks in the same order (reclaimed -> artifact -> gate) and
    raises a typed :class:`MediaResolutionError` the caller maps to an HTTP status
    or a capability reason. The caller owns the returned handle and MUST close it.
    """
    if run_intermediate_reclaimed_at(session, run_id) is not None:
        raise MediaReclaimed("media reclaimed")
    try:
        path = normalized_audio_path(session, run_id, settings.media_root)
    except StageDataError as exc:
        raise MediaMissing(str(exc)) from exc
    try:
        return gate.open_for_serving(path)
    except MediaNotServableError as exc:
        raise MediaUnservable(str(exc)) from exc


# --------------------------------------------------------------------------- #
# Capability predicate.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CapabilityReason:
    """One plain-language reason seeking is disabled (honest-copy doctrine)."""

    code: str
    message: str


# Honest, non-technical wording — shown directly in the capability banner.
_REASON_MESSAGES: dict[str, str] = {
    "media_missing": (
        "The processed audio for this run is not available, so playback is disabled."
    ),
    "media_reclaimed": (
        "The processed audio was reclaimed to free disk space. Re-run the pipeline "
        "from the source to restore playback."
    ),
    "media_unservable": (
        "The processed audio file could not be opened, so playback is disabled."
    ),
    "duration_unknown": (
        "This recording's length is unknown, so per-segment seeking cannot be verified safe."
    ),
    "duration_invalid": (
        "This recording's stored length is invalid, so per-segment seeking cannot be "
        "verified safe."
    ),
    "timeline_malformed": (
        "Some transcript timestamps are out of order or invalid, so seeking is disabled "
        "to avoid landing on the wrong moment."
    ),
    "timeline_out_of_bounds": (
        "Some transcript timestamps fall past the end of the recording, so seeking is "
        "disabled to avoid landing on the wrong moment."
    ),
    "no_segments": "This run has no transcript segments to play.",
}


def _reason(code: str) -> CapabilityReason:
    return CapabilityReason(code=code, message=_REASON_MESSAGES[code])


@dataclass(frozen=True)
class PlaybackCapability:
    """Whether per-turn seeking is safe, and — when not — every reason it is not."""

    seek_enabled: bool
    media_duration: float | None
    reasons: list[CapabilityReason]

    def to_props(self) -> dict[str, object]:
        """JSON-safe props for an island (camelCase, matches the TS interface)."""
        return {
            "seekEnabled": self.seek_enabled,
            "mediaDuration": self.media_duration,
            "reasons": [{"code": r.code, "message": r.message} for r in self.reasons],
        }


def _transcript_intervals(session: Session, run_id: uuid.UUID) -> list[tuple[float, float]]:
    rows = session.execute(
        select(TranscriptSegment.start_seconds, TranscriptSegment.end_seconds).where(
            TranscriptSegment.pipeline_run_id == run_id
        )
    ).all()
    return [(row[0], row[1]) for row in rows]


def playback_capability(
    session: Session,
    run: PipelineRun,
    settings: Settings,
    gate: MediaGate,
) -> PlaybackCapability:
    """Fail-closed capability: ``seek_enabled`` iff EVERY precondition holds.

    Accumulates *all* applicable reasons (never early-returns) so the banner can
    explain everything wrong at once. ``media_duration`` is populated only when
    the stored duration is a finite positive number.
    """
    reasons: list[CapabilityReason] = []

    # 1. Media must actually be servable — reuse the /media seam verbatim.
    try:
        fh, _size = resolve_servable_media(session, run.id, settings, gate)
    except MediaResolutionError as exc:
        reasons.append(_reason(exc.code))
    else:
        fh.close()

    # 2. Duration must be a finite positive number.
    duration = run.media_item.duration_seconds
    media_duration: float | None = None
    if duration is None:
        reasons.append(_reason("duration_unknown"))
    elif not (math.isfinite(duration) and duration > 0):
        reasons.append(_reason("duration_invalid"))
    else:
        media_duration = duration

    # 3. The transcript timeline must be well-formed and inside the recording.
    intervals = _transcript_intervals(session, run.id)
    if not intervals:
        reasons.append(_reason("no_segments"))
    else:
        malformed = any(
            not (math.isfinite(start) and math.isfinite(end) and end > start >= 0)
            for start, end in intervals
        )
        if malformed:
            reasons.append(_reason("timeline_malformed"))
        # Out-of-bounds is only meaningful against a valid duration; compute the
        # last end over finite values so a NaN interval can't poison max().
        finite_ends = [end for _start, end in intervals if math.isfinite(end)]
        if (
            media_duration is not None
            and finite_ends
            and max(finite_ends) > media_duration + TAIL_TOLERANCE
        ):
            reasons.append(_reason("timeline_out_of_bounds"))

    return PlaybackCapability(
        seek_enabled=not reasons,
        media_duration=media_duration,
        reasons=reasons,
    )
