"""Require a recent language-model answer before post-lane admission (#768).

Revision ID: 0070
Revises: 0069
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0070"
down_revision: str | None = "0069"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "gpu_phase", sa.Column("llm_ready", sa.Boolean(), nullable=False, server_default=sa.false())
    )
    op.add_column(
        "gpu_phase", sa.Column("llm_checked_at", sa.DateTime(timezone=True), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("gpu_phase", "llm_checked_at")
    op.drop_column("gpu_phase", "llm_ready")
