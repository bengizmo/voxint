"""Finalize stage: the completion checkpoint.

Deliberately a no-op in P3 — transcript export and completeness gates land
with the quality-gate work. Keeping the stage in the canonical order means
adding that logic later is a body change, not a state-machine change.
"""

import uuid

from sqlalchemy.orm import Session

from voxint.pipeline.stages.context import StageContext


def run(ctx: StageContext, session: Session, run_id: uuid.UUID) -> None:
    del ctx, session, run_id
