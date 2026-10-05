"""Store operator English filler additions and keeps (#753).

Revision ID: 0071
Revises: 0070
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0071"
down_revision: str | None = "0070"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for name in ("fillers_add", "fillers_keep"):
        op.add_column(
            "app_settings",
            sa.Column(
                name,
                sa.JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql"),
                nullable=True,
            ),
        )


def downgrade() -> None:
    op.drop_column("app_settings", "fillers_keep")
    op.drop_column("app_settings", "fillers_add")
