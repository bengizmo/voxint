"""audio_artifacts reclamation columns — media retention / GC (issue #15)

Adds the reclamation audit stamp to audio_artifacts: the GC sweep unlinks the
normalized-audio intermediate for old terminal runs and records reclaimed_at +
reclaimed_bytes on the row (never deleting the row). A paired-nullability check
forbids half-stamped rows; a partial index keeps the sweep predicate cheap.

Revision ID: 0014
Revises: 0013
Create Date: 2026-08-16 08:10:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0014"
down_revision: str | None = "0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "audio_artifacts",
        sa.Column("reclaimed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "audio_artifacts",
        sa.Column("reclaimed_bytes", sa.BigInteger(), nullable=True),
    )
    op.create_check_constraint(
        "audio_artifacts_reclaimed_shape_check",
        "audio_artifacts",
        "(reclaimed_at IS NULL) = (reclaimed_bytes IS NULL)",
    )
    op.create_check_constraint(
        "audio_artifacts_reclaimed_bytes_nonneg_check",
        "audio_artifacts",
        "reclaimed_bytes IS NULL OR reclaimed_bytes >= 0",
    )
    op.create_index(
        "ix_audio_artifacts_reclaimable",
        "audio_artifacts",
        ["pipeline_run_id"],
        postgresql_where=sa.text("kind = 'preprocessed_audio' AND reclaimed_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_audio_artifacts_reclaimable", table_name="audio_artifacts")
    op.drop_constraint(
        "audio_artifacts_reclaimed_bytes_nonneg_check",
        "audio_artifacts",
        type_="check",
    )
    op.drop_constraint(
        "audio_artifacts_reclaimed_shape_check",
        "audio_artifacts",
        type_="check",
    )
    op.drop_column("audio_artifacts", "reclaimed_bytes")
    op.drop_column("audio_artifacts", "reclaimed_at")
