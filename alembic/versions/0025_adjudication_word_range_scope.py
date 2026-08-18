"""adjudication_decisions word-range scope — sub-segment reassignment

Issue #59 slice 3: reassign a derived split child (or any word-range of a
segment) to a different speaker. A ruling can now target a half-open
``[start_word_index, end_word_index)`` interval over the parent segment's
immutable ``words`` list, one grain finer than the 0018 whole-segment scope.

Storage stays the ONE immutable human ledger — the range is two nullable columns
on ``adjudication_decisions``, not a new table and not a FK to the disposable
``segment_split_boundaries`` rows — so the coordinates key on the immutable
parent id + word offsets and survive re-split / un-split. Both NULL = the 0018
whole-segment (or 0001 label) grain, unchanged; both set = this sub-segment range.

Purely additive and preserves append-only:
- Two nullable Integer columns with NO server default. An existing row reads
  both NULL ⇒ its scope is exactly what it was pre-0025 (no data backfill, no
  behavior change on upgrade).
- The append-only trigger is unaffected — ADD COLUMN is DDL, not a row UPDATE.
- CHECKs keep the pair coherent: start/end are set together or both NULL; a
  present range must scope a segment and be a well-formed non-empty half-open
  interval (``end > start >= 0``). The contrapositive keeps label-scope rows
  range-NULL: no segment ⇒ no range.

The index ``ix_adjudication_decisions_word_range`` is the batch-load path the
read-time resolver uses to fold word-range overrides over derived children
without an N+1 (mirrors ``ix_adjudication_decisions_run_segment`` at the finer
grain).

Revision ID: 0025
Revises: 0024
Create Date: 2026-08-17 19:10:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0025"
down_revision: str | None = "0024"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "adjudication_decisions",
        sa.Column("start_word_index", sa.Integer(), nullable=True),
    )
    op.add_column(
        "adjudication_decisions",
        sa.Column("end_word_index", sa.Integer(), nullable=True),
    )
    # Set together or both NULL.
    op.create_check_constraint(
        "adjudication_decisions_word_range_pair_check",
        "adjudication_decisions",
        "(start_word_index IS NULL) = (end_word_index IS NULL)",
    )
    # A present range must scope a segment and be a well-formed non-empty
    # half-open interval; no segment ⇒ no range (label scope stays range-NULL).
    op.create_check_constraint(
        "adjudication_decisions_word_range_bounds_check",
        "adjudication_decisions",
        "start_word_index IS NULL OR ("
        "transcript_segment_id IS NOT NULL"
        " AND start_word_index >= 0"
        " AND end_word_index > start_word_index)",
    )
    op.create_index(
        "ix_adjudication_decisions_word_range",
        "adjudication_decisions",
        [
            "pipeline_run_id",
            "transcript_segment_id",
            "start_word_index",
            "end_word_index",
            "created_at",
            "id",
        ],
    )


def downgrade() -> None:
    # Dropping the range columns would silently PROMOTE every sub-segment
    # reassignment into a whole-segment override (its narrower scope lost),
    # changing read-time attribution. Refuse rather than corrupt: word-range
    # rows are permanent human rulings — remove them deliberately first (mirrors
    # 0018's guard for the segment-scope column).
    bind = op.get_bind()
    ranged = bind.execute(
        sa.text(
            "SELECT count(*) FROM adjudication_decisions"
            " WHERE start_word_index IS NOT NULL"
        )
    ).scalar_one()
    if ranged:
        raise RuntimeError(
            f"cannot downgrade past 0024: {ranged} word-range reassignment "
            "ruling(s) exist. Removing them is a deliberate, destructive act — "
            "do it explicitly before downgrading."
        )
    op.drop_index(
        "ix_adjudication_decisions_word_range", table_name="adjudication_decisions"
    )
    op.drop_constraint(
        "adjudication_decisions_word_range_bounds_check",
        "adjudication_decisions",
        type_="check",
    )
    op.drop_constraint(
        "adjudication_decisions_word_range_pair_check",
        "adjudication_decisions",
        type_="check",
    )
    op.drop_column("adjudication_decisions", "end_word_index")
    op.drop_column("adjudication_decisions", "start_word_index")
