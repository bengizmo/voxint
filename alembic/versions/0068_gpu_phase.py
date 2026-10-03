"""Persistent shared-GPU phase state (#748).

Revision ID: 0068
Revises: 0067
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0068"
down_revision: str | None = "0067"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "gpu_phase",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=False),
        sa.Column("phase", sa.Text(), nullable=False, server_default=sa.text("'llm'")),
        sa.Column(
            "phase_since", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("lease_id", sa.Text(), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("failures", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("retry_after", sa.DateTime(timezone=True), nullable=True),
        sa.Column("operator_request", sa.Text(), nullable=True),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint("id = 1", name="gpu_phase_singleton_check"),
        sa.CheckConstraint(
            "phase IN ('llm', 'draining_post', 'acquiring', 'starting_services', "
            "'audio', 'draining', 'stopping_services', 'releasing', 'error')",
            name="gpu_phase_phase_check",
        ),
        sa.CheckConstraint("failures >= 0", name="gpu_phase_failures_check"),
        sa.CheckConstraint(
            "operator_request IN ('audio', 'release')", name="gpu_phase_request_check"
        ),
    )
    op.execute("INSERT INTO gpu_phase (id, phase) VALUES (1, 'llm')")


def downgrade() -> None:
    op.drop_table("gpu_phase")
