"""Transcribe stage: ASR over the normalized audio, raw text preserved forever."""

import uuid

from sqlalchemy import delete
from sqlalchemy.orm import Session

from voxint.db.models import TranscriptSegment
from voxint.pipeline.stages.context import StageContext, normalized_audio_path

# Whisper biases toward — and can hallucinate from — an over-long initial_prompt,
# so bound it. Terms are dropped whole at the boundary rather than mid-word.
INITIAL_PROMPT_MAX_CHARS = 2000


def _initial_prompt(vocabulary: tuple[str, ...]) -> str | None:
    """Render the run's vocabulary as a bounded whisper ``initial_prompt``.

    ``vocabulary`` is already deduped/ordered upstream; this joins terms with
    ", " up to the char cap (whole terms only) and returns None when empty.
    """
    parts: list[str] = []
    length = 0
    for term in vocabulary:
        addition = len(term) + (2 if parts else 0)  # ", " separator between terms
        if length + addition > INITIAL_PROMPT_MAX_CHARS:
            break
        parts.append(term)
        length += addition
    return ", ".join(parts) if parts else None


def run(ctx: StageContext, session: Session, run_id: uuid.UUID) -> None:
    audio = normalized_audio_path(session, run_id, ctx.media_root)
    result = ctx.asr.transcribe(audio, initial_prompt=_initial_prompt(ctx.vocabulary))
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
