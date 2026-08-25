"""Attributed audio-clip generation + serving orchestration (issue #88).

The DB + filesystem + MediaGate layer that sits between the pure extraction core
(:mod:`voxint.media.clips`) and the review-console routes:

- :func:`generate_or_adopt_clip` turns a live, non-stale ``word_range``
  annotation into a cached clip row (content-addressed, idempotent under an
  advisory lock), returning the clip's UUID.
- :func:`resolve_servable_clip` loads exactly one clip artifact by its UUID and
  hands back an open handle for the serve route, with the same typed-status
  discipline as :func:`voxint.api.playback.resolve_servable_media`.

Clips are a reclaimable cache: an existing clip serves independently of the
normalized audio, and a cache miss regenerates only while that source is live.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from voxint.db.models import ArtifactKind, AudioArtifact
from voxint.media.clips import (
    ClipBoundsError,
    ClipError,
    clip_idempotency_key,
    clip_relative_path,
    extract_clip,
    read_total_frames,
    resolve_sample_bounds,
)
from voxint.media.serving import MediaGate, MediaNotServableError

if TYPE_CHECKING:
    from voxint.config import Settings

# Bumped only when the stored meta snapshot's shape changes.
CLIP_META_VERSION = 1


class ClipServiceError(Exception):
    """A clip cannot be generated or served. Carries the operator-facing HTTP
    status so the route and any capability surface stay in lockstep."""

    http_status: int = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ClipAnchorUnclippable(ClipServiceError):
    """The annotation has no precise word timing (coarse/segment anchor) or the
    requested span is invalid — nothing to cut. A 422 (client/anchor problem)."""

    http_status = 422


class ClipSourceUnavailable(ClipServiceError):
    """The normalized audio needed to CUT the clip is missing or reclaimed, so a
    new clip cannot be generated. An existing clip is unaffected (it serves from
    its own file). 409: the source must be restored (re-run) first."""

    http_status = 409


class ClipNotFound(ClipServiceError):
    """No clip artifact with this UUID for this run (foreign/forged/absent). 404
    — UUID opacity is not authorization, so a cross-run id is indistinguishable
    from missing."""

    http_status = 404


class ClipReclaimed(ClipServiceError):
    """The clip's file was reclaimed by GC (issue #15); the row survives. 410."""

    http_status = 410


class ClipUnservable(ClipServiceError):
    """The clip row exists but its file cannot be opened/served. 404."""

    http_status = 404


@dataclass(frozen=True)
class ServableClip:
    """An open clip handle plus the sanitized download filename. The caller owns
    and MUST close ``handle``."""

    handle: BinaryIO
    size: int
    filename: str


def clip_download_filename(run_id: uuid.UUID, clip_id: uuid.UUID) -> str:
    """Deterministic, ASCII, server-derived attachment name — never any client
    or quote text (no CRLF/quoting/bidi surface)."""
    return f"voxint-{run_id.hex[:8]}-clip-{clip_id.hex[:8]}.wav"


def _live_clip_row(
    session: Session, run_id: uuid.UUID, idempotency_key: str
) -> AudioArtifact | None:
    return session.execute(
        select(AudioArtifact).where(
            AudioArtifact.pipeline_run_id == run_id,
            AudioArtifact.kind == ArtifactKind.AUDIO_CLIP.value,
            AudioArtifact.idempotency_key == idempotency_key,
            AudioArtifact.reclaimed_at.is_(None),
        )
    ).scalar_one_or_none()


def _source_artifact(session: Session, run_id: uuid.UUID) -> AudioArtifact:
    """The run's live normalized-audio artifact, or raise ClipSourceUnavailable.

    A missing/duplicate/reclaimed source means a clip cannot be CUT (an already
    generated clip is unaffected — it serves from its own file)."""
    rows = list(
        session.execute(
            select(AudioArtifact).where(
                AudioArtifact.pipeline_run_id == run_id,
                AudioArtifact.kind == ArtifactKind.PREPROCESSED_AUDIO.value,
            )
        ).scalars()
    )
    if len(rows) != 1:
        raise ClipSourceUnavailable(
            "the processed audio for this run is not available, so a clip cannot "
            "be extracted"
        )
    art = rows[0]
    if art.reclaimed_at is not None:
        raise ClipSourceUnavailable(
            "the processed audio was reclaimed to free disk space; re-run the "
            "pipeline from the source before extracting a clip"
        )
    return art


def generate_or_adopt_clip(
    session: Session,
    run_id: uuid.UUID,
    *,
    annotation_id: uuid.UUID,
    annotation_source_text_hash: str,
    start_seconds: float,
    end_seconds: float,
    settings: Settings,
    gate: MediaGate,
) -> uuid.UUID:
    """Return the UUID of the cached clip for this annotation span, extracting it
    if absent.

    Idempotent and content-addressed: identical requests adopt the one live row;
    a re-anchor (new sample bounds) makes a new clip. Serialized per clip by a
    transaction-scoped advisory lock keyed on the content hash, so two
    concurrent requests for the same span publish one canonical row. The caller
    is expected to have already confirmed the annotation is live, non-stale, and
    word-timed; the precise-timing guard here is a defensive backstop.
    """
    art = _source_artifact(session, run_id)
    source_path = settings.media_root / art.path
    try:
        fh, _size = gate.open_for_serving(source_path)
    except MediaNotServableError as exc:
        raise ClipSourceUnavailable(f"processed audio is not servable: {exc}") from exc

    try:
        total_frames = read_total_frames(fh)
        max_frames = int(settings.clip_max_duration_seconds * 16000)
        try:
            bounds = resolve_sample_bounds(
                start_seconds,
                end_seconds,
                total_frames,
                max_clip_frames=max_frames,
            )
        except ClipBoundsError as exc:
            raise ClipAnchorUnclippable(str(exc)) from exc

        key = clip_idempotency_key(
            normalized_artifact_id=art.id,
            annotation_id=annotation_id,
            start_sample=bounds.start_sample,
            end_sample=bounds.end_sample,
        )
        rel_path = clip_relative_path(run_id, key)

        # Fast path: an already-cached live clip with its file present.
        existing = _live_clip_row(session, run_id, key)
        if existing is not None and (settings.media_root / existing.path).is_file():
            return existing.id

        # Slow path: serialize this specific clip's generation. 63-bit digest of
        # the content key -> same span, same lock, any worker.
        lock_key = int(key[:16], 16) & 0x7FFFFFFFFFFFFFFF
        session.execute(select(func.pg_advisory_xact_lock(lock_key)))

        # Re-check under the lock: a racing request may have just published it.
        existing = _live_clip_row(session, run_id, key)
        if existing is not None and (settings.media_root / existing.path).is_file():
            return existing.id

        try:
            clip_file = extract_clip(fh, bounds, settings.media_root / rel_path)
        except ClipBoundsError as exc:
            raise ClipAnchorUnclippable(str(exc)) from exc
        except ClipError as exc:
            raise ClipSourceUnavailable(str(exc)) from exc
    finally:
        fh.close()

    meta: dict[str, Any] = {
        "meta_version": CLIP_META_VERSION,
        "annotation_id": str(annotation_id),
        "normalized_artifact_id": str(art.id),
        "annotation_source_text_hash": annotation_source_text_hash,
        "requested_start_seconds": start_seconds,
        "requested_end_seconds": end_seconds,
        "start_sample": bounds.start_sample,
        "end_sample": bounds.end_sample,
        "sample_rate": 16000,
        "frame_count": clip_file.frame_count,
    }
    if existing is not None:
        # The file was reclaimed/absent but the row survives: it now points at the
        # freshly written deterministic path again.
        existing.meta = meta
        return existing.id
    clip = AudioArtifact(
        pipeline_run_id=run_id,
        kind=ArtifactKind.AUDIO_CLIP.value,
        path=rel_path,
        meta=meta,
        idempotency_key=key,
    )
    session.add(clip)
    try:
        session.flush()
    except IntegrityError:
        # A racing writer won the unique index between our re-check and flush.
        session.rollback()
        adopted = _live_clip_row(session, run_id, key)
        if adopted is None:  # pragma: no cover - unexpected
            raise
        return adopted.id
    return clip.id


def resolve_servable_clip(
    session: Session,
    run_id: uuid.UUID,
    clip_id: uuid.UUID,
    settings: Settings,
    gate: MediaGate,
) -> ServableClip:
    """Open exactly the one clip artifact addressed by ``clip_id`` under ``run_id``.

    UUID opacity is not authorization — the route still runs behind OperatorDep.
    Reclaimed -> 410; missing row/file or a gate refusal -> 404.
    """
    row = session.execute(
        select(AudioArtifact).where(
            AudioArtifact.id == clip_id,
            AudioArtifact.pipeline_run_id == run_id,
            AudioArtifact.kind == ArtifactKind.AUDIO_CLIP.value,
        )
    ).scalar_one_or_none()
    if row is None:
        raise ClipNotFound("no such clip for this run")
    if row.reclaimed_at is not None:
        raise ClipReclaimed("this clip was reclaimed to free disk space")
    path = _confined_clip_path(settings.media_root, row.path)
    try:
        handle, size = gate.open_for_serving(path)
    except MediaNotServableError as exc:
        raise ClipUnservable(str(exc)) from exc
    return ServableClip(
        handle=handle,
        size=size,
        filename=clip_download_filename(run_id, clip_id),
    )


def _confined_clip_path(media_root: Path, stored_path: str) -> Path:
    """Join a stored relative clip path to the media root, rejecting a stored
    value that escapes it (defence in depth; MediaGate re-confines too)."""
    candidate = (media_root / stored_path).resolve()
    if not candidate.is_relative_to(media_root.resolve()):
        raise ClipUnservable(f"stored clip path {stored_path!r} escapes the media root")
    return candidate
