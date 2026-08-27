"""Prepare stage: manufacture the 16 kHz mono WAV invariant.

Everything downstream — the GPU contracts included — assumes normalized audio;
this stage is where that guarantee is created and recorded.
"""

import uuid

from sqlalchemy import delete
from sqlalchemy.orm import Session

from voxint.db.models import ArtifactKind, AudioArtifact, MediaItem, PipelineRun
from voxint.media.integrity import openable_current
from voxint.media.normalize import normalize_to_wav
from voxint.media.peaks import peaks_relative
from voxint.pipeline.stages.context import StageContext, StageDataError

# Run-scoped so retries overwrite their own output and runs never collide.
_ARTIFACT_TEMPLATE = "artifacts/{run_id}/normalized.wav"


def run(ctx: StageContext, session: Session, run_id: uuid.UUID) -> None:
    pipeline_run = session.get(PipelineRun, run_id)
    if pipeline_run is None:
        raise StageDataError(f"no pipeline run {run_id}")
    media = session.get(MediaItem, pipeline_run.media_item_id)
    if media is None:
        raise StageDataError(f"run {run_id}: media item missing")

    source = openable_current(ctx.media_root, media)
    if source is None:
        live_path = media.current_path or media.source_path
        raise StageDataError(
            f"source {live_path!r} is not a readable regular file "
            f"under media root {ctx.media_root}"
        )

    relative = _ARTIFACT_TEMPLATE.format(run_id=run_id)
    info = normalize_to_wav(
        source,
        ctx.media_root / relative,
        ffmpeg_bin=ctx.ffmpeg_bin,
        ffprobe_bin=ctx.ffprobe_bin,
    )

    # The normalized stream's duration is canonical — source headers lie.
    media.duration_seconds = info.duration_seconds
    media.size_bytes = source.stat().st_size

    # Also drop any waveform-peaks cache (issue #57): the WAV it described no
    # longer exists, and its row-level source fingerprint would fail anyway —
    # deleting here keeps re-runs from serving a stale envelope even briefly. We
    # unlink the sidecar file too: once its row is gone, media-delete (which
    # plans from rows) could never find it, so leaving it would orphan a ~14 KB
    # file until the next view recomputes over it.
    (ctx.media_root / peaks_relative(run_id)).unlink(missing_ok=True)
    session.execute(
        delete(AudioArtifact).where(
            AudioArtifact.pipeline_run_id == run_id,
            AudioArtifact.kind.in_(
                (
                    ArtifactKind.PREPROCESSED_AUDIO.value,
                    ArtifactKind.WAVEFORM_PEAKS.value,
                )
            ),
        )
    )
    session.add(
        AudioArtifact(
            pipeline_run_id=run_id,
            kind=ArtifactKind.PREPROCESSED_AUDIO.value,
            path=relative,
            meta={
                "sample_rate": info.sample_rate,
                "channels": info.channels,
                "codec": info.codec,
                "duration_seconds": info.duration_seconds,
            },
        )
    )
