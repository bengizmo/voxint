"""Hardened run-audio access for plugins (issue #137, epic #136).

A plugin that consumes a run's processed audio (the near-future synthetic-speech
detector is the first consumer) needs a reference to the prepared 16 kHz mono WAV
that is safe to hand across the plugin boundary. :func:`run_audio_descriptor`
returns a :class:`RunAudioDescriptor` — run id, artifact id, media-relative path,
size, and reclamation state — never a bare ``Path``, and only for a COMPLETED
run whose file resolves to a regular file confined under ``media_root``.

This deliberately does NOT reuse ``pipeline.stages.context.normalized_audio_path``:
that helper returns ``media_root / row.path`` unresolved, trusting the caller to
confine it, which is exactly the raw-path handoff a plugin surface must not make.
The confinement here mirrors ``media/serving.py`` and ``media/peaks.py`` (resolve,
media-root containment, regular-file check). Reclamation is reported, not raised,
so a caller can present an honest "audio reclaimed" state; the GC contract (rule
in epic #136) keeps a run with an active audio-consuming plugin job out of the
reclaim sweep, so a live job never races the unlink.
"""

from __future__ import annotations

import stat as stat_module
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import select

from voxint.db.models import ArtifactKind, AudioArtifact, PipelineRun, RunStatus

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


@dataclass(frozen=True)
class RunAudioDescriptor:
    """A safe reference to a run's processed-audio artifact.

    ``media_relative_path`` is the artifact row's path (relative to
    ``media_root``); a caller resolves it through the same media gate the console
    uses to actually read bytes. ``size_bytes`` is the on-disk size (or the
    recorded reclaimed byte count once the file is gone). ``reclaimed`` is ``True``
    when the intermediate has been GC'd — the row survives, the file does not.
    """

    run_id: uuid.UUID
    artifact_id: uuid.UUID
    media_relative_path: str
    size_bytes: int
    reclaimed: bool


class RunAudioUnavailable(Exception):
    """A run has no descriptor-eligible processed audio.

    ``code`` is a stable machine token (mirrors ``playback``'s capability codes)
    so a caller can branch or map to operator copy without string-matching.
    """

    code: str = "audio_unavailable"


class RunNotCompleted(RunAudioUnavailable):
    """The run is missing or not COMPLETED (audio access is completed-only)."""

    code = "run_not_completed"


class AudioMissing(RunAudioUnavailable):
    """No single preprocessed-audio artifact exists for the run."""

    code = "audio_missing"


class AudioUnconfined(RunAudioUnavailable):
    """The artifact path escapes ``media_root`` or is not a regular file."""

    code = "audio_unservable"


def preprocessed_audio_row(
    session: Session, run_id: uuid.UUID
) -> AudioArtifact | None:
    """The run's single preprocessed-audio artifact row, or ``None``.

    ``None`` when there is no row OR more than one (an invariant break the caller
    treats as missing, never guessing which of several is canonical).
    """
    rows = (
        session.execute(
            select(AudioArtifact).where(
                AudioArtifact.pipeline_run_id == run_id,
                AudioArtifact.kind == ArtifactKind.PREPROCESSED_AUDIO.value,
            )
        )
        .scalars()
        .all()
    )
    return rows[0] if len(rows) == 1 else None


def run_audio_descriptor(
    session: Session, run_id: uuid.UUID, *, media_root: Path
) -> RunAudioDescriptor:
    """Return a confined descriptor for a COMPLETED run's processed audio.

    Raises :class:`RunNotCompleted`, :class:`AudioMissing`, or
    :class:`AudioUnconfined` — the failure modes a plugin must fail closed on. A
    reclaimed intermediate is NOT an error: the descriptor is returned with
    ``reclaimed=True`` and the recorded byte count, so the caller decides whether
    a re-run is needed.
    """
    run = session.get(PipelineRun, run_id)
    if run is None or run.status != RunStatus.COMPLETED.value:
        raise RunNotCompleted(f"run {run_id} is not completed")

    row = preprocessed_audio_row(session, run_id)
    if row is None:
        raise AudioMissing(f"run {run_id} has no single preprocessed-audio artifact")

    if row.reclaimed_at is not None:
        # The file is gone; the row is the only record. Report it rather than
        # touching disk (there is nothing to confine).
        return RunAudioDescriptor(
            run_id=run_id,
            artifact_id=row.id,
            media_relative_path=row.path,
            size_bytes=row.reclaimed_bytes or 0,
            reclaimed=True,
        )

    resolved = (media_root / row.path).resolve()
    if not resolved.is_relative_to(media_root.resolve()):
        raise AudioUnconfined(f"{row.path} escapes the media root")
    try:
        st = resolved.stat()
    except OSError as exc:
        raise AudioUnconfined(f"cannot stat {row.path}: {exc}") from exc
    if not stat_module.S_ISREG(st.st_mode):
        raise AudioUnconfined(f"{row.path} is not a regular file")

    return RunAudioDescriptor(
        run_id=run_id,
        artifact_id=row.id,
        media_relative_path=row.path,
        size_bytes=st.st_size,
        reclaimed=False,
    )
