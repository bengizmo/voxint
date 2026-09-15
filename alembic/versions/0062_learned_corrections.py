"""Learned corrections: toggle, suggestions, and evidence (#476).

``projects.learn_corrections`` enables observation of operator transcript edits.
``learned_corrections`` holds suggested (and accepted) match/replace pairs with
provenance. ``learned_correction_evidence`` links each suggestion to the
transcript segments whose edits produced it; count is always COUNT(*) over
this table.

Revision ID: 0062
Revises: 0061
Create Date: 2026-09-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0062"
down_revision: str | None = "0061"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column(
            "learn_corrections",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )

    op.create_table(
        "learned_corrections",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "project_id",
            sa.Uuid(),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("match", sa.Text(), nullable=False),
        sa.Column("replace", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("accepted_rule_id", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('suggested', 'accepted')",
            name="learned_corrections_status_check",
        ),
        sa.CheckConstraint(
            "(status = 'accepted') = (accepted_rule_id IS NOT NULL)",
            name="learned_corrections_accepted_rule_id_check",
        ),
        sa.UniqueConstraint(
            "project_id", "match", "replace",
            name="learned_corrections_project_match_replace_key",
        ),
    )
    op.create_index(
        "ix_learned_corrections_project_id",
        "learned_corrections",
        ["project_id"],
    )

    op.create_table(
        "learned_correction_evidence",
        sa.Column(
            "learned_correction_id",
            sa.Uuid(),
            sa.ForeignKey("learned_corrections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "segment_id",
            sa.Uuid(),
            sa.ForeignKey("transcript_segments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("learned_correction_id", "segment_id"),
    )
    op.create_index(
        "ix_learned_correction_evidence_segment_id",
        "learned_correction_evidence",
        ["segment_id"],
    )


def downgrade() -> None:
    op.drop_table("learned_correction_evidence")
    op.drop_table("learned_corrections")
    op.drop_column("projects", "learn_corrections")
