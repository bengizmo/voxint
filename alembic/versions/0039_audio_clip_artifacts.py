"""Attributed audio-clip artifacts (#88)

Additive, one table touched (``audio_artifacts``). Clips are a new
``ArtifactKind`` stored as ordinary artifact rows, so manual run-media deletion
and the confinement machinery cover them for free. Three changes:

- Widen ``audio_artifacts_kind_check`` to admit ``'audio_clip'``.
- Add a nullable ``idempotency_key`` Text column, present exactly for clips
  (``audio_artifacts_clip_key_shape_check``). It is a content-addressed cache
  key (normalized-artifact id + annotation id + integer sample bounds).
- Add ``uq_audio_artifacts_clip_key``: one LIVE clip row per (run, key). It
  excludes reclaimed tombstones, so a post-reclamation regeneration inserts a
  fresh row rather than reviving the stamped one. And
  ``ix_audio_artifacts_clip_reclaimable``: the GC sweep predicate, so clips age
  by their OWN created_at.

Keep in lockstep with the ``AudioArtifact`` model + ``ArtifactKind`` enum.

Revision ID: 0039
Revises: 0038
Create Date: 2026-08-25 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0039"
down_revision: str | None = "0038"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OLD_KINDS = "('preprocessed_audio', 'chunk', 'transcript_export', 'waveform_peaks')"
_NEW_KINDS = (
    "('preprocessed_audio', 'chunk', 'transcript_export', 'waveform_peaks', "
    "'audio_clip')"
)


def upgrade() -> None:
    op.add_column(
        "audio_artifacts",
        sa.Column("idempotency_key", sa.Text(), nullable=True),
    )
    op.drop_constraint("audio_artifacts_kind_check", "audio_artifacts", type_="check")
    op.create_check_constraint(
        "audio_artifacts_kind_check",
        "audio_artifacts",
        f"kind IN {_NEW_KINDS}",
    )
    op.create_check_constraint(
        "audio_artifacts_clip_key_shape_check",
        "audio_artifacts",
        "(idempotency_key IS NOT NULL) = (kind = 'audio_clip')",
    )
    op.create_index(
        "uq_audio_artifacts_clip_key",
        "audio_artifacts",
        ["pipeline_run_id", "idempotency_key"],
        unique=True,
        postgresql_where=sa.text("kind = 'audio_clip' AND reclaimed_at IS NULL"),
    )
    op.create_index(
        "ix_audio_artifacts_clip_reclaimable",
        "audio_artifacts",
        ["pipeline_run_id"],
        postgresql_where=sa.text("kind = 'audio_clip' AND reclaimed_at IS NULL"),
    )


def downgrade() -> None:
    # Drop clip rows first so the narrowed kind CHECK can be re-applied.
    op.execute("DELETE FROM audio_artifacts WHERE kind = 'audio_clip'")
    op.drop_index("ix_audio_artifacts_clip_reclaimable", table_name="audio_artifacts")
    op.drop_index("uq_audio_artifacts_clip_key", table_name="audio_artifacts")
    op.drop_constraint(
        "audio_artifacts_clip_key_shape_check", "audio_artifacts", type_="check"
    )
    op.drop_constraint("audio_artifacts_kind_check", "audio_artifacts", type_="check")
    op.create_check_constraint(
        "audio_artifacts_kind_check",
        "audio_artifacts",
        f"kind IN {_OLD_KINDS}",
    )
    op.drop_column("audio_artifacts", "idempotency_key")
