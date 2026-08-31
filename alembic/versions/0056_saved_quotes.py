"""Add saved_quotes table for the quote board (issue #338, Phase 6).

Revision ID: 0056
Revises: 0055
Create Date: 2026-08-30
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0056"
down_revision: str = "0055"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "saved_quotes",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("segment_id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("search_query", sa.Text(), nullable=False),
        sa.Column("left_context", sa.Text(), nullable=False),
        sa.Column("hit", sa.Text(), nullable=False),
        sa.Column("right_context", sa.Text(), nullable=False),
        sa.Column("speaker_name", sa.Text(), nullable=True),
        sa.Column("media_title", sa.Text(), nullable=False),
        sa.Column("start_seconds", sa.Float(), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("operator", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["segment_id"], ["transcript_segments.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["pipeline_runs.id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "project_id",
            "segment_id",
            "search_query",
            name="saved_quotes_project_segment_query_key",
        ),
        sa.CheckConstraint(
            "char_length(search_query) > 0",
            name="saved_quotes_query_nonempty_check",
        ),
        sa.CheckConstraint(
            "note IS NULL OR char_length(note) <= 2000",
            name="saved_quotes_note_len_check",
        ),
    )
    op.create_index(
        "ix_saved_quotes_project_created",
        "saved_quotes",
        ["project_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_saved_quotes_project_created", table_name="saved_quotes")
    op.drop_table("saved_quotes")
