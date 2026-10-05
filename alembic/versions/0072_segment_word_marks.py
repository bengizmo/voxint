"""Store append-only, attributed per-word marks (#757).

Revision ID: 0072
Revises: 0071

Downgrade drops the mark ledger and permanently loses its data.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0072"
down_revision: str | None = "0071"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "segment_word_marks",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("seq", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("pipeline_run_id", sa.Uuid(), sa.ForeignKey("pipeline_runs.id"), nullable=False),
        sa.Column(
            "segment_id",
            sa.Uuid(),
            sa.ForeignKey("transcript_segments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("start_word_index", sa.Integer(), nullable=False),
        sa.Column("end_word_index", sa.Integer(), nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column(
            "voids_mark_id",
            sa.Uuid(),
            sa.ForeignKey("segment_word_marks.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("operator", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("idempotency_key", sa.Text(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("seq", name="segment_word_marks_seq_key"),
        sa.UniqueConstraint("idempotency_key", name="segment_word_marks_idempotency_key"),
        sa.CheckConstraint(
            "start_word_index >= 0 AND end_word_index > start_word_index",
            name="segment_word_marks_word_range_bounds_check",
        ),
        sa.CheckConstraint(
            "action IN ('keep', 'omit', 'clear', 'undo')",
            name="segment_word_marks_action_check",
        ),
        sa.CheckConstraint(
            "(action = 'undo') = (voids_mark_id IS NOT NULL)",
            name="segment_word_marks_undo_check",
        ),
        sa.CheckConstraint(
            "voids_mark_id IS NULL OR voids_mark_id <> id",
            name="segment_word_marks_no_self_void_check",
        ),
        sa.CheckConstraint(
            "user_id IS NULL OR operator NOT LIKE 'system:%'",
            name="segment_word_marks_user_not_system_check",
        ),
        sa.CheckConstraint(
            "operator NOT LIKE 'system:%' OR user_id IS NULL",
            name="segment_word_marks_system_not_user_check",
        ),
    )
    op.create_index(
        "ix_segment_word_marks_voids",
        "segment_word_marks",
        ["voids_mark_id"],
        unique=True,
        postgresql_where=sa.text("voids_mark_id IS NOT NULL"),
    )
    op.create_index(
        "ix_segment_word_marks_run_segment_seq",
        "segment_word_marks",
        ["pipeline_run_id", "segment_id", "seq"],
    )
    op.execute("""
        CREATE FUNCTION segment_word_marks_append_only()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF TG_OP = 'UPDATE' THEN
                RAISE EXCEPTION 'segment_word_marks is append-only (UPDATE blocked)'
                    USING ERRCODE = 'P0001';
            END IF;
            IF EXISTS (SELECT 1 FROM transcript_segments WHERE id = OLD.segment_id) THEN
                RAISE EXCEPTION 'segment_word_marks is append-only (DELETE blocked)'
                    USING ERRCODE = 'P0001';
            END IF;
            RETURN OLD;
        END $$
    """)
    op.execute("""
        CREATE TRIGGER segment_word_marks_append_only_trigger
        BEFORE UPDATE OR DELETE ON segment_word_marks
        FOR EACH ROW EXECUTE FUNCTION segment_word_marks_append_only()
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER segment_word_marks_append_only_trigger ON segment_word_marks")
    op.execute("DROP FUNCTION segment_word_marks_append_only()")
    op.drop_table("segment_word_marks")
