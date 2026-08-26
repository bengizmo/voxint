"""media operation journal and per-file progress (ADR 0007, issue #155)

Creates the durable journal for byte-touching move, trash, restore, and purge
operations. ``media_operations`` records operation state, retry scheduling, and
leases; ``media_operation_files`` tracks per-file progress for purge inventory
and sidecar bundles. Adds trash and purge timestamps to ``media_items``.

Revision ID: 0044
Revises: 0043
Create Date: 2026-08-26
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0044"
down_revision: str | None = "0043"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OPERATION_TYPES = "'move', 'trash', 'restore', 'purge'"
_OPERATION_STATES = (
    "'planned', 'fs_applied', 'db_applied', 'awaiting_retry', "
    "'completed', 'failed'"
)
_FILE_KINDS = "'source', 'sidecar', 'preprocessed_audio', 'audio_clip', 'peaks'"
_FILE_STATUSES = "'pending', 'done', 'missing', 'failed'"


def upgrade() -> None:
    op.create_table(
        "media_operations",
        sa.Column(
            "id",
            sa.Uuid(),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "media_id",
            sa.Uuid(),
            sa.ForeignKey("media_items.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("operation_type", sa.Text(), nullable=False),
        sa.Column(
            "state",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'planned'"),
        ),
        sa.Column("origin_path", sa.Text(), nullable=True),
        sa.Column("destination_path", sa.Text(), nullable=True),
        sa.Column("origin_digest", sa.Text(), nullable=True),
        sa.Column("error_code", sa.Text(), nullable=True),
        sa.Column(
            "attempt_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("last_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("claimed_by", sa.Text(), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "restores_operation_id",
            sa.Uuid(),
            sa.ForeignKey("media_operations.id"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            f"operation_type IN ({_OPERATION_TYPES})",
            name="media_operations_operation_type_check",
        ),
        sa.CheckConstraint(
            f"state IN ({_OPERATION_STATES})",
            name="media_operations_state_check",
        ),
    )
    op.create_index(
        "uq_media_operations_active_per_item",
        "media_operations",
        ["media_id"],
        unique=True,
        postgresql_where=sa.text("state NOT IN ('completed', 'failed')"),
    )
    op.create_index(
        "ix_media_operations_reconciler",
        "media_operations",
        ["state", "next_attempt_at"],
    )
    op.create_index(
        "ix_media_operations_media_id",
        "media_operations",
        ["media_id"],
    )

    op.create_table(
        "media_operation_files",
        sa.Column(
            "id",
            sa.Uuid(),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "operation_id",
            sa.Uuid(),
            sa.ForeignKey("media_operations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("file_kind", sa.Text(), nullable=False),
        sa.Column(
            "status",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            f"file_kind IN ({_FILE_KINDS})",
            name="media_operation_files_file_kind_check",
        ),
        sa.CheckConstraint(
            f"status IN ({_FILE_STATUSES})",
            name="media_operation_files_status_check",
        ),
    )
    op.create_index(
        "ix_media_operation_files_operation_id",
        "media_operation_files",
        ["operation_id"],
    )

    op.add_column(
        "media_items",
        sa.Column("trashed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "media_items",
        sa.Column("purged_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("media_operation_files")
    op.drop_table("media_operations")
    op.drop_column("media_items", "trashed_at")
    op.drop_column("media_items", "purged_at")
