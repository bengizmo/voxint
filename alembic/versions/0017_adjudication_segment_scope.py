"""adjudication_decisions.transcript_segment_id + INHERIT — per-segment relabel

Issue #54 Phase B: two-scope relabel. A ruling can now target one transcript
segment instead of the whole (run, label). Storage stays the ONE immutable human
ledger — segment scope is a nullable ``transcript_segment_id`` column, not a
second table — so replay, idempotency, and read-time resolution have a single
home. NULL = label scope (the historical grain); non-NULL = this-segment scope.

A new ``inherit`` decision value is the append-only reset for a segment override:
because the ledger is insert-only, "undo this segment's override" is a new row
that falls back to the label's resolution, not an UPDATE. INHERIT is segment-only
(a CHECK enforces it); it carries no speaker (the existing assign-speaker CHECK
already generalises: only ``assign`` has a speaker_id).

The label-scope resolvers keep their exact meaning by filtering
``transcript_segment_id IS NULL`` (done in code); this migration only widens the
schema. The append-only trigger is unaffected — ADD COLUMN is DDL, not a row
UPDATE.

Revision ID: 0017
Revises: 0016
Create Date: 2026-08-16 18:30:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0017"
down_revision: str | None = "0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "adjudication_decisions",
        sa.Column(
            "transcript_segment_id",
            sa.Uuid(),
            sa.ForeignKey("transcript_segments.id"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_adjudication_decisions_run_segment",
        "adjudication_decisions",
        ["pipeline_run_id", "transcript_segment_id"],
    )
    # Widen the allowed decision values to include 'inherit'.
    op.drop_constraint("adjudication_decisions_check", "adjudication_decisions", type_="check")
    op.create_check_constraint(
        "adjudication_decisions_check",
        "adjudication_decisions",
        "decision IN ('assign', 'exclude', 'unknown', 'inherit')",
    )
    # INHERIT is segment-scope only.
    op.create_check_constraint(
        "adjudication_decisions_inherit_segment_check",
        "adjudication_decisions",
        "decision != 'inherit' OR transcript_segment_id IS NOT NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        "adjudication_decisions_inherit_segment_check",
        "adjudication_decisions",
        type_="check",
    )
    op.drop_constraint("adjudication_decisions_check", "adjudication_decisions", type_="check")
    op.create_check_constraint(
        "adjudication_decisions_check",
        "adjudication_decisions",
        "decision IN ('assign', 'exclude', 'unknown')",
    )
    op.drop_index(
        "ix_adjudication_decisions_run_segment", table_name="adjudication_decisions"
    )
    op.drop_column("adjudication_decisions", "transcript_segment_id")
