"""Prepare stage: manufacture the 16 kHz mono WAV invariant.

Everything downstream — the GPU contracts included — assumes normalized audio;
this stage is where that guarantee is created and recorded.
"""

import uuid

from sqlalchemy import delete
from sqlalchemy.orm import Session

from voxint.db.models import ArtifactKind, AudioArtifact, MediaItem, PipelineRun
from voxint.media.normalize import normalize_to_wav
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

    media_root = ctx.media_root.resolve()
    source = ctx.media_root / media.source_path
    try:
        source.resolve().relative_to(media_root)
    except ValueError:
        raise StageDataError(
            f"source {media.source_path!r} escapes media root {media_root}"
        ) from None
    if not source.is_file():
        raise StageDataError(f"source {media.source_path!r} is not a regular file")

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

    session.execute(
        delete(AudioArtifact).where(
            AudioArtifact.pipeline_run_id == run_id,
            AudioArtifact.kind == ArtifactKind.PREPROCESSED_AUDIO.value,
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
