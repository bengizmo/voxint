"""research_jobs — operator-initiated web-research job state (issue #40)

Revision ID: 0011
Revises: 0010
Create Date: 2026-08-15 18:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "research_jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("speaker_id", sa.Uuid(), nullable=False),
        sa.Column("pipeline_run_id", sa.Uuid(), nullable=True),
        sa.Column("status", sa.Text(), server_default=sa.text("'queued'"), nullable=False),
        sa.Column(
            "cancel_requested",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("budget", postgresql.JSONB(), nullable=False),
        sa.Column("operator_note", sa.Text(), nullable=True),
        sa.Column("searches_used", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("reads_used", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("rounds_used", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("producer_run_id", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="research_jobs_status_check",
        ),
        sa.CheckConstraint(
            "searches_used >= 0 AND reads_used >= 0 AND rounds_used >= 0",
            name="research_jobs_counters_check",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(budget) = 'object'",
            name="research_jobs_budget_object_check",
        ),
        sa.CheckConstraint(
            "started_at IS NULL OR started_at >= created_at",
            name="research_jobs_started_after_created_check",
        ),
        sa.CheckConstraint(
            "finished_at IS NULL OR started_at IS NOT NULL",
            name="research_jobs_finished_requires_started_check",
        ),
        sa.ForeignKeyConstraint(["speaker_id"], ["speakers.id"]),
        sa.ForeignKeyConstraint(["pipeline_run_id"], ["pipeline_runs.id"]),
        sa.ForeignKeyConstraint(["producer_run_id"], ["enrichment_producer_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_research_jobs_speaker_id", "research_jobs", ["speaker_id"])
    op.create_index("ix_research_jobs_pipeline_run_id", "research_jobs", ["pipeline_run_id"])
    # DB-enforced "one active job per speaker" — the console's friendly
    # pre-check is racy check-then-insert; this index is the real invariant.
    op.create_index(
        "research_jobs_one_active_per_speaker",
        "research_jobs",
        ["speaker_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('queued', 'running')"),
    )


def downgrade() -> None:
    op.drop_index("research_jobs_one_active_per_speaker", table_name="research_jobs")
    op.drop_index("ix_research_jobs_pipeline_run_id", table_name="research_jobs")
    op.drop_index("ix_research_jobs_speaker_id", table_name="research_jobs")
    op.drop_table("research_jobs")
