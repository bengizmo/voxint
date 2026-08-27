"""Widen media_operation_files file_kind CHECK for chunk and export tracking.

Purge enumerates ALL derived files for a media item, including AudioChunk
segments and transcript exports. The original CHECK (0044) covered only the
move/trash/restore manifest kinds.

Revision ID: 0045
Revises: 0044
Create Date: 2026-08-26
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0045"
down_revision: str | None = "0044"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OLD_KINDS = "'source', 'sidecar', 'preprocessed_audio', 'audio_clip', 'peaks'"
_NEW_KINDS = (
    "'source', 'sidecar', 'preprocessed_audio', 'audio_clip', "
    "'peaks', 'chunk', 'transcript_export'"
)


def upgrade() -> None:
    op.drop_constraint(
        "media_operation_files_file_kind_check", "media_operation_files"
    )
    op.create_check_constraint(
        "media_operation_files_file_kind_check",
        "media_operation_files",
        f"file_kind IN ({_NEW_KINDS})",
    )


def downgrade() -> None:
    op.execute(
        "DELETE FROM media_operation_files "
        "WHERE file_kind IN ('chunk', 'transcript_export')"
    )
    op.drop_constraint(
        "media_operation_files_file_kind_check", "media_operation_files"
    )
    op.create_check_constraint(
        "media_operation_files_file_kind_check",
        "media_operation_files",
        f"file_kind IN ({_OLD_KINDS})",
    )
