"""transcript semantic-search embedding spine — segment_embeddings + embedding_jobs (#121)

The additive embedding producer reads finished transcript text and writes these
two tables. It never touches ASR / diarization / TitaNet, so it does not trip
the numerics parity gate.

- ``segment_embeddings``: one row per embedded transcript chunk (paragraph-
  derived, token-bounded). ``embedding`` is a 384-dim MiniLM text vector, a
  space deliberately separate from the 192-dim TitaNet speaker space; cosine is
  only valid within one ``embedding_space``. Chunks are ephemeral (boundaries
  shift on correction/split/speaker changes) so the span and text live on the
  row. ``generation`` is monotonic per (run, space): a re-embed publishes a
  whole new generation atomically. No ANN index in v1 — exact cosine scan is
  sub-second at single-operator scale.
- ``embedding_jobs``: the dedicated build-attempt lane (not the LLM-coupled
  run-asset family). One active job per (run, space) via a partial unique index;
  ``source_content_hash`` is the staleness detector; a succeeded job stamps the
  ``generation`` it published.

The pgvector extension already exists (created in 0001). Downgrade drops both
tables (disposable derived data; the transcript is untouched).

Revision ID: 0033
Revises: 0032
Create Date: 2026-08-21 23:30:00.000000
"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

revision: str = "0033"
down_revision: str | None = "0032"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "segment_embeddings",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "pipeline_run_id",
            sa.Uuid(),
            sa.ForeignKey("pipeline_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("embedding_space", sa.Text(), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("start_seconds", sa.Float(), nullable=False),
        sa.Column("end_seconds", sa.Float(), nullable=False),
        sa.Column("speaker_label", sa.Text(), nullable=True),
        sa.Column("text_rendering", sa.Text(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(dim=384), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "pipeline_run_id",
            "embedding_space",
            "generation",
            "chunk_index",
            name="segment_embeddings_chunk_key",
        ),
        sa.CheckConstraint(
            "length(trim(embedding_space)) > 0",
            name="segment_embeddings_space_nonempty_check",
        ),
        sa.CheckConstraint("generation >= 1", name="segment_embeddings_generation_check"),
        sa.CheckConstraint("chunk_index >= 0", name="segment_embeddings_chunk_index_check"),
        sa.CheckConstraint(
            "start_seconds >= 0 AND end_seconds >= start_seconds",
            name="segment_embeddings_interval_check",
        ),
        sa.CheckConstraint(
            "content_hash ~ '^[0-9a-f]{64}$'",
            name="segment_embeddings_content_hash_check",
        ),
    )
    op.create_index(
        "ix_segment_embeddings_pipeline_run_id",
        "segment_embeddings",
        ["pipeline_run_id"],
    )
    op.create_index(
        "ix_segment_embeddings_run_space_generation",
        "segment_embeddings",
        ["pipeline_run_id", "embedding_space", "generation"],
    )

    op.create_table(
        "embedding_jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "pipeline_run_id",
            sa.Uuid(),
            sa.ForeignKey("pipeline_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("embedding_space", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False),
        sa.Column("source_content_hash", sa.Text(), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="embedding_jobs_status_check",
        ),
        sa.CheckConstraint(
            "length(trim(embedding_space)) > 0",
            name="embedding_jobs_space_nonempty_check",
        ),
        sa.CheckConstraint(
            "generation IS NULL OR generation >= 1",
            name="embedding_jobs_generation_check",
        ),
        sa.CheckConstraint(
            "source_content_hash ~ '^[0-9a-f]{64}$'",
            name="embedding_jobs_source_hash_check",
        ),
        sa.CheckConstraint(
            "started_at IS NULL OR started_at >= created_at",
            name="embedding_jobs_started_after_created_check",
        ),
        sa.CheckConstraint(
            "finished_at IS NULL OR started_at IS NOT NULL",
            name="embedding_jobs_finished_requires_started_check",
        ),
    )
    op.create_index(
        "ix_embedding_jobs_pipeline_run_id",
        "embedding_jobs",
        ["pipeline_run_id"],
    )
    op.create_index(
        "embedding_jobs_one_active_per_run_space",
        "embedding_jobs",
        ["pipeline_run_id", "embedding_space"],
        unique=True,
        postgresql_where=sa.text("status IN ('queued', 'running')"),
    )


def downgrade() -> None:
    op.drop_index(
        "embedding_jobs_one_active_per_run_space", table_name="embedding_jobs"
    )
    op.drop_index("ix_embedding_jobs_pipeline_run_id", table_name="embedding_jobs")
    op.drop_table("embedding_jobs")
    op.drop_index(
        "ix_segment_embeddings_run_space_generation", table_name="segment_embeddings"
    )
    op.drop_index(
        "ix_segment_embeddings_pipeline_run_id", table_name="segment_embeddings"
    )
    op.drop_table("segment_embeddings")
