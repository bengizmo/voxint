"""Synthdetect plugin tables and AppSettings columns (#145).

Revision ID: 0050
Revises: 0049
"""

import sqlalchemy as sa
from alembic import op

revision: str = "0050"
down_revision: str = "0049"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "synthdetect_jobs",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column(
            "pipeline_run_id",
            sa.UUID(),
            sa.ForeignKey("pipeline_runs.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("inference_space", sa.Text(), nullable=False),
        sa.Column("calibration_policy_id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="queued"),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("total_turns", sa.Integer()),
        sa.Column("scored_turns", sa.Integer()),
        sa.Column("skipped_turns", sa.Integer()),
        sa.Column("mean_risk", sa.Float()),
        sa.Column("max_risk", sa.Float()),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="synthdetect_jobs_status_check",
        ),
        sa.CheckConstraint(
            "started_at IS NULL OR started_at >= created_at",
            name="synthdetect_jobs_started_after_created_check",
        ),
        sa.CheckConstraint(
            "finished_at IS NULL OR started_at IS NOT NULL",
            name="synthdetect_jobs_finished_requires_started_check",
        ),
    )
    op.create_index(
        "synthdetect_jobs_one_active_per_run",
        "synthdetect_jobs",
        ["pipeline_run_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('queued', 'running')"),
    )

    op.create_table(
        "synthdetect_scores",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column(
            "synthdetect_job_id",
            sa.UUID(),
            sa.ForeignKey("synthdetect_jobs.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "pipeline_run_id",
            sa.UUID(),
            sa.ForeignKey("pipeline_runs.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "diarization_turn_id",
            sa.UUID(),
            sa.ForeignKey("diarization_turns.id", ondelete="SET NULL"),
        ),
        sa.Column("speaker_label", sa.Text()),
        sa.Column("raw_logit", sa.Float()),
        sa.Column("calibrated_score", sa.Float()),
        sa.Column("window_count", sa.Integer(), nullable=False),
        sa.Column("skip_reason", sa.Text()),
        sa.Column("inference_space", sa.Text(), nullable=False),
        sa.Column("calibration_policy_id", sa.Text(), nullable=False),
        sa.UniqueConstraint(
            "synthdetect_job_id",
            "diarization_turn_id",
            name="synthdetect_scores_job_turn_key",
        ),
        sa.CheckConstraint(
            "(raw_logit IS NULL) = (skip_reason IS NOT NULL)",
            name="synthdetect_scores_logit_xor_skip_check",
        ),
    )

    op.add_column("app_settings", sa.Column("synthdetect_enabled", sa.Boolean()))
    op.add_column("app_settings", sa.Column("synthdetect_autogenerate", sa.Boolean()))
    op.add_column("app_settings", sa.Column("synthdetect_url", sa.Text()))
    op.add_column("app_settings", sa.Column("synthdetect_http_timeout_seconds", sa.Integer()))


def downgrade() -> None:
    op.drop_column("app_settings", "synthdetect_http_timeout_seconds")
    op.drop_column("app_settings", "synthdetect_url")
    op.drop_column("app_settings", "synthdetect_autogenerate")
    op.drop_column("app_settings", "synthdetect_enabled")
    op.drop_table("synthdetect_scores")
    op.drop_table("synthdetect_jobs")
