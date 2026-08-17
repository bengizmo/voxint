"""transcript_segments.confidence — persist ASR per-segment confidence

Issue #53: the whisper service already emits per-segment confidence
(``exp(avg_logprob)`` clamped to [0, 1] — a transformed likelihood, NOT a
calibrated probability; see docs/quality-gates.md), but the app client dropped
it. This adds the nullable column so the review console can flag low-confidence
segments for triage. NULL for runs transcribed before this migration and for any
backend that reports no confidence — the console never fabricates a value for a
NULL. A CHECK mirrors the service's clamp so a bad value is rejected at write.

Revision ID: 0018
Revises: 0017
Create Date: 2026-08-16 22:40:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0018"
down_revision: str | None = "0017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "transcript_segments",
        sa.Column("confidence", sa.Float(), nullable=True),
    )
    op.create_check_constraint(
        "transcript_segments_confidence_range_check",
        "transcript_segments",
        "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
    )


def downgrade() -> None:
    op.drop_constraint(
        "transcript_segments_confidence_range_check",
        "transcript_segments",
        type_="check",
    )
    op.drop_column("transcript_segments", "confidence")
