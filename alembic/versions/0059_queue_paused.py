"""Add queue_paused flag to app_settings (#419).

A non-nullable Boolean defaulting false. When true, the worker skips claiming
fresh pipeline runs (run_pipeline lane) and recovery_sweep skips re-dispatching
stale QUEUED runs. In-progress runs (finish_pipeline lane) and auxiliary job
lanes are unaffected.

Revision ID: 0059
Revises: 0058
Create Date: 2026-09-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0059"
down_revision: str | None = "0058"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "app_settings",
        sa.Column(
            "queue_paused",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )


def downgrade() -> None:
    op.drop_column("app_settings", "queue_paused")
