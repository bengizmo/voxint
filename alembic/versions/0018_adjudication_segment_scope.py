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

Revision ID: 0018
Revises: 0017
Create Date: 2026-08-16 18:30:00.000000
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
    # Segment scope carries only assign or inherit (belt-and-suspenders for the
    # writer, which the route already enforces).
    op.create_check_constraint(
        "adjudication_decisions_segment_decision_check",
        "adjudication_decisions",
        "transcript_segment_id IS NULL OR decision IN ('assign', 'inherit')",
    )


def downgrade() -> None:
    # Dropping the scope column would silently PROMOTE every per-segment override
    # into a whole-label ruling (and 'inherit' rows can't satisfy the pre-0017
    # decision CHECK). Refuse rather than corrupt: the operator must remove the
    # segment-scope rows deliberately first (they are permanent human rulings, so
    # this migration will not delete them for you).
    bind = op.get_bind()
    scoped = bind.execute(
        sa.text(
            "SELECT count(*) FROM adjudication_decisions"
            " WHERE transcript_segment_id IS NOT NULL"
        )
    ).scalar_one()
    if scoped:
        raise RuntimeError(
            f"cannot downgrade past 0017: {scoped} per-segment override ruling(s) "
            "exist. Removing them is a deliberate, destructive act — do it "
            "explicitly before downgrading."
        )
    op.drop_constraint(
        "adjudication_decisions_segment_decision_check",
        "adjudication_decisions",
        type_="check",
    )
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
