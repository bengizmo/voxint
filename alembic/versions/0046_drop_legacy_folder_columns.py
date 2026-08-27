"""Drop rollback-only app_settings.media_folders / folder_domain_packs columns.

Since #153 (migration 0040) the ``media_folders`` RELATION is authoritative for
registered folders and per-folder domain packs.  The legacy singleton columns on
``app_settings`` were retained for one release as a rollback input; that release
(v0.27.0) has shipped, so the columns are safe to drop.

Revision ID: 0046
Revises: 0045
Create Date: 2026-08-27
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0046"
down_revision: str = "0045"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_column("app_settings", "media_folders")
    op.drop_column("app_settings", "folder_domain_packs")


def downgrade() -> None:
    import sqlalchemy as sa
    from sqlalchemy import text

    op.add_column(
        "app_settings",
        sa.Column(
            "media_folders",
            sa.ARRAY(sa.Text()),
            nullable=False,
            server_default=text("ARRAY[]::text[]"),
        ),
    )
    op.add_column(
        "app_settings",
        sa.Column(
            "folder_domain_packs",
            sa.dialects.postgresql.JSONB(),
            nullable=False,
            server_default=text("'{}'::jsonb"),
        ),
    )
