"""Add archived_at to projects (#477).

Nullable TIMESTAMPTZ column for project archive/restore lifecycle.
No index (tiny table, single-operator tool).

Downgrade drops the column: archived projects come back active under the
old code, and a re-upgrade leaves archived_at NULL. Requires a matching
application rollback.

Revision ID: 0064
Revises: 0063
Create Date: 2026-09-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0064"
down_revision: str | None = "0063"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("projects", "archived_at")
