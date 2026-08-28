"""Add benchmark_runs and benchmark_items tables.

DB-backed benchmark results for cross-run comparison of pipeline
transcription accuracy (WER) and throughput.

Revision ID: 0047
Revises: 0046
Create Date: 2026-08-28
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0047"
down_revision: str = "0046"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "benchmark_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tag", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("corpus_version", sa.Integer(), nullable=False),
        sa.Column("protocol_hash", sa.Text(), nullable=False),
        sa.Column("voxint_version", sa.Text(), nullable=False),
        sa.Column("config_snapshot", postgresql.JSONB(), nullable=False),
        sa.Column("system_info", postgresql.JSONB(), nullable=False),
        sa.Column("summary", postgresql.JSONB(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name="benchmark_runs_status_check",
        ),
        sa.CheckConstraint(
            "corpus_version >= 1",
            name="benchmark_runs_corpus_version_positive_check",
        ),
        sa.CheckConstraint(
            "tag IS NULL OR length(tag) <= 60",
            name="benchmark_runs_tag_length_check",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_benchmark_runs_created_at", "benchmark_runs", ["created_at"])

    op.create_table(
        "benchmark_items",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("benchmark_run_id", sa.Uuid(), nullable=False),
        sa.Column("corpus_file_id", sa.Text(), nullable=False),
        sa.Column("pipeline_run_id", sa.Uuid(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("stage_timings", postgresql.JSONB(), nullable=True),
        sa.Column("wer_counts", postgresql.JSONB(), nullable=True),
        sa.Column("hallucination_words", sa.Integer(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'submitted', 'completed', 'failed', 'skipped')",
            name="benchmark_items_status_check",
        ),
        sa.ForeignKeyConstraint(
            ["benchmark_run_id"],
            ["benchmark_runs.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["pipeline_run_id"],
            ["pipeline_runs.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "benchmark_run_id", "corpus_file_id",
            name="uq_benchmark_items_run_file",
        ),
    )


def downgrade() -> None:
    op.drop_table("benchmark_items")
    op.drop_index("ix_benchmark_runs_created_at", table_name="benchmark_runs")
    op.drop_table("benchmark_runs")
