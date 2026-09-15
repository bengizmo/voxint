"""Watch-folder pickup marker columns on media_items (#478).

``picked_up_by_sweep_at`` (nullable TIMESTAMPTZ) is the sweep batch timestamp;
``picked_up_from_folder`` (nullable TEXT) is the MEDIA_ROOT-relative folder path
frozen at ingest time.  Partial index for the Home feed query.

Revision ID: 0063
Revises: 0062
Create Date: 2026-09-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0063"
down_revision: str | None = "0062"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "media_items",
        sa.Column("picked_up_by_sweep_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "media_items",
        sa.Column("picked_up_from_folder", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_media_items_pickup",
        "media_items",
        [sa.text("picked_up_by_sweep_at DESC")],
        postgresql_where=sa.text("picked_up_by_sweep_at IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_media_items_pickup", table_name="media_items")
    op.drop_column("media_items", "picked_up_from_folder")
    op.drop_column("media_items", "picked_up_by_sweep_at")
