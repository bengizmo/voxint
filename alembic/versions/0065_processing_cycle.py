"""Add processing cycles for restart-from-stage (#506).

Revision ID: 0065
Revises: 0064
Create Date: 2026-09-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0065"
down_revision: str | None = "0064"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "pipeline_runs",
        sa.Column("processing_cycle", sa.Integer(), nullable=False, server_default="1"),
    )
    op.create_check_constraint(
        "pipeline_runs_processing_cycle_nonneg_check",
        "pipeline_runs",
        "processing_cycle >= 1",
    )
    op.add_column(
        "stage_runs",
        sa.Column("processing_cycle", sa.Integer(), nullable=False, server_default="1"),
    )
    op.create_check_constraint(
        "stage_runs_processing_cycle_nonneg_check",
        "stage_runs",
        "processing_cycle >= 1",
    )
    op.drop_constraint("stage_runs_attempt_key", "stage_runs", type_="unique")
    op.create_unique_constraint(
        "stage_runs_attempt_key",
        "stage_runs",
        ["pipeline_run_id", "stage", "processing_cycle", "attempt"],
    )


def downgrade() -> None:
    # If any run was restarted (cycle > 1), both cycles may have attempt=1 for
    # the same (run, stage).  The old constraint rejects that.  This downgrade
    # is safe only when no cycle > 1 data exists (pre-release, or before the
    # restart-from-stage service integration ships).
    op.drop_constraint("stage_runs_attempt_key", "stage_runs", type_="unique")
    op.create_unique_constraint(
        "stage_runs_attempt_key",
        "stage_runs",
        ["pipeline_run_id", "stage", "attempt"],
    )
    op.drop_constraint("stage_runs_processing_cycle_nonneg_check", "stage_runs", type_="check")
    op.drop_constraint(
        "pipeline_runs_processing_cycle_nonneg_check", "pipeline_runs", type_="check"
    )
    op.drop_column("stage_runs", "processing_cycle")
    op.drop_column("pipeline_runs", "processing_cycle")
