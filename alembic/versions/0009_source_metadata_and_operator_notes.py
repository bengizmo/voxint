"""media_source_metadata snapshot table + pipeline_runs.operator_notes (issue #36)

Revision ID: 0009
Revises: 0008
Create Date: 2026-08-14 21:30:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "media_source_metadata",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("media_item_id", sa.Uuid(), nullable=False),
        sa.Column("source_kind", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("uploader", sa.Text(), nullable=True),
        sa.Column("uploader_url", sa.Text(), nullable=True),
        sa.Column("channel", sa.Text(), nullable=True),
        sa.Column("channel_url", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("upload_date", sa.Date(), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column(
            "tags",
            postgresql.ARRAY(sa.Text()),
            server_default=sa.text("'{}'::text[]"),
            nullable=False,
        ),
        sa.Column("canonical_url", sa.Text(), nullable=True),
        sa.Column("extractor", sa.Text(), nullable=True),
        sa.Column("extractor_version", sa.Text(), nullable=True),
        sa.Column("raw", postgresql.JSONB(), nullable=True),
        sa.Column("raw_schema_version", sa.Integer(), nullable=False),
        sa.Column("acquired_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "source_kind IN ('ytdlp', 'rss')",
            name="media_source_metadata_kind_check",
        ),
        sa.CheckConstraint(
            "duration_seconds IS NULL OR duration_seconds >= 0",
            name="media_source_metadata_duration_nonneg_check",
        ),
        sa.CheckConstraint(
            "raw_schema_version >= 1",
            name="media_source_metadata_raw_schema_version_check",
        ),
        sa.ForeignKeyConstraint(
            ["media_item_id"],
            ["media_items.id"],
            name="media_source_metadata_media_item_id_fkey",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "media_item_id", name="media_source_metadata_media_item_id_key"
        ),
    )
    op.add_column(
        "pipeline_runs", sa.Column("operator_notes", sa.Text(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("pipeline_runs", "operator_notes")
    op.drop_table("media_source_metadata")
