"""Add source hash to synthdetect jobs.

Revision ID: 0060
Revises: 0059
Create Date: 2026-09-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0060"
down_revision: str | None = "0059"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "synthdetect_jobs",
        sa.Column("source_content_hash", sa.Text(), nullable=True),
    )
    op.create_check_constraint(
        "synthdetect_jobs_source_hash_check",
        "synthdetect_jobs",
        "source_content_hash IS NULL OR source_content_hash ~ '^[0-9a-f]{64}$'",
    )


def downgrade() -> None:
    op.drop_constraint(
        "synthdetect_jobs_source_hash_check",
        "synthdetect_jobs",
        type_="check",
    )
    op.drop_column("synthdetect_jobs", "source_content_hash")
