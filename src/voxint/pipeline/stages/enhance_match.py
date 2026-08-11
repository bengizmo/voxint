"""Enhance + match stage.

P3 scope: enhancement runs only when an LLM client is wired (P4); until then
``enhanced_text`` is explicitly reset to NULL so retries after a config change
never leave stale enhancements behind. Speaker matching (cosine over the turn
ledger) is P4.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.db.models import TranscriptSegment
from voxint.pipeline.stages.context import StageContext


def run(ctx: StageContext, session: Session, run_id: uuid.UUID) -> None:
    segments = (
        session.execute(
            select(TranscriptSegment)
            .where(TranscriptSegment.pipeline_run_id == run_id)
            .order_by(TranscriptSegment.segment_index)
        )
        .scalars()
        .all()
    )
    for segment in segments:
        segment.enhanced_text = (
            ctx.llm.enhance(segment.raw_text, context="") if ctx.llm is not None else None
        )
