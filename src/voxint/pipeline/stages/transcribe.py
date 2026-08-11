"""Transcribe stage: ASR over the normalized audio, raw text preserved forever."""

import uuid

from sqlalchemy import delete
from sqlalchemy.orm import Session

from voxint.db.models import TranscriptSegment
from voxint.pipeline.stages.context import StageContext, normalized_audio_path


def run(ctx: StageContext, session: Session, run_id: uuid.UUID) -> None:
    audio = normalized_audio_path(session, run_id, ctx.media_root)
    result = ctx.asr.transcribe(audio)
    session.execute(
        delete(TranscriptSegment).where(TranscriptSegment.pipeline_run_id == run_id)
    )
    for index, segment in enumerate(result.segments):
        session.add(
            TranscriptSegment(
                pipeline_run_id=run_id,
                segment_index=index,
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                raw_text=segment.text,
                suspect=segment.suspect,
            )
        )
