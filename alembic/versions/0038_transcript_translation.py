"""Transcript translation: settings columns + job/result tables (#133)

Three pieces, all additive:

- ``app_settings.translation_target_language`` (nullable Text override — NULL/
  blank inherits the env default) and ``app_settings.translation_autogenerate``
  (nullable tri-state Boolean, the semantic-index pattern from migration 0036).
- ``run_translations``: immutable whole-transcript translation generations —
  one row per (run, target language, generation), lines as a versioned JSONB
  snapshot, run-level ``source_content_hash`` as the only freshness authority,
  superseded-not-edited (the ``run_enrichment_assets`` pattern; head uniqueness
  is held by the writer's advisory lock + atomic supersede, not an index).
- ``translation_jobs``: mutable orchestration state (queued → running →
  terminal, the ``run_asset_jobs`` pattern) with a one-active partial unique
  index and the enqueue-time source hash for the executor's source-changed
  race guard.

Revision ID: 0038
Revises: 0037
Create Date: 2026-08-23 15:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0038"
down_revision: str | None = "0037"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "app_settings",
        sa.Column("translation_target_language", sa.Text(), nullable=True),
    )
    op.add_column(
        "app_settings",
        sa.Column("translation_autogenerate", sa.Boolean(), nullable=True),
    )

    op.create_table(
        "run_translations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "pipeline_run_id",
            sa.Uuid(),
            sa.ForeignKey("pipeline_runs.id"),
            nullable=False,
        ),
        sa.Column("target_language", sa.Text(), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("source_language", sa.Text(), nullable=True),
        sa.Column("lines", JSONB(), nullable=False),
        sa.Column("payload_schema_version", sa.Integer(), nullable=False),
        sa.Column("producer", sa.Text(), nullable=False),
        sa.Column("producer_version", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("source_content_hash", sa.Text(), nullable=False),
        sa.Column(
            "superseded_by_translation_id",
            sa.Uuid(),
            sa.ForeignKey("run_translations.id"),
            nullable=True,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "pipeline_run_id",
            "target_language",
            "generation",
            name="run_translations_generation_key",
        ),
        sa.CheckConstraint(
            "length(trim(target_language)) > 0",
            name="run_translations_target_language_nonempty_check",
        ),
        sa.CheckConstraint("generation >= 1", name="run_translations_generation_check"),
        sa.CheckConstraint(
            "jsonb_typeof(lines) = 'array'",
            name="run_translations_lines_array_check",
        ),
        sa.CheckConstraint(
            "payload_schema_version >= 1",
            name="run_translations_payload_version_check",
        ),
        sa.CheckConstraint(
            "length(trim(producer)) > 0",
            name="run_translations_producer_nonempty_check",
        ),
        sa.CheckConstraint(
            "length(trim(producer_version)) > 0",
            name="run_translations_producer_version_nonempty_check",
        ),
        sa.CheckConstraint(
            "length(trim(model)) > 0",
            name="run_translations_model_nonempty_check",
        ),
        sa.CheckConstraint(
            "source_content_hash ~ '^[0-9a-f]{64}$'",
            name="run_translations_source_hash_check",
        ),
        sa.CheckConstraint(
            "completed_at >= started_at",
            name="run_translations_completed_after_started_check",
        ),
    )
    op.create_index(
        "ix_run_translations_pipeline_run_id",
        "run_translations",
        ["pipeline_run_id"],
    )
    # No one-current partial unique index (deliberate, the
    # run_enrichment_assets precedent): the writer's insert-then-supersede
    # transaction transiently holds two unsuperseded rows; head uniqueness is
    # guaranteed by the writer's advisory lock + atomic supersede instead.

    op.create_table(
        "translation_jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "pipeline_run_id",
            sa.Uuid(),
            sa.ForeignKey("pipeline_runs.id"),
            nullable=False,
        ),
        sa.Column("target_language", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False),
        sa.Column("config", JSONB(), nullable=False),
        sa.Column("source_content_hash", sa.Text(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "translation_id",
            sa.Uuid(),
            sa.ForeignKey("run_translations.id"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "length(trim(target_language)) > 0",
            name="translation_jobs_target_language_nonempty_check",
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="translation_jobs_status_check",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(config) = 'object'",
            name="translation_jobs_config_object_check",
        ),
        sa.CheckConstraint(
            "source_content_hash ~ '^[0-9a-f]{64}$'",
            name="translation_jobs_source_hash_check",
        ),
        sa.CheckConstraint(
            "started_at IS NULL OR started_at >= created_at",
            name="translation_jobs_started_after_created_check",
        ),
        sa.CheckConstraint(
            "finished_at IS NULL OR started_at IS NOT NULL",
            name="translation_jobs_finished_requires_started_check",
        ),
    )
    op.create_index(
        "ix_translation_jobs_pipeline_run_id",
        "translation_jobs",
        ["pipeline_run_id"],
    )
    op.create_index(
        "translation_jobs_one_active_per_run_language",
        "translation_jobs",
        ["pipeline_run_id", "target_language"],
        unique=True,
        postgresql_where=sa.text("status IN ('queued', 'running')"),
    )


def downgrade() -> None:
    op.drop_index(
        "translation_jobs_one_active_per_run_language", table_name="translation_jobs"
    )
    op.drop_index("ix_translation_jobs_pipeline_run_id", table_name="translation_jobs")
    op.drop_table("translation_jobs")
    op.drop_index("ix_run_translations_pipeline_run_id", table_name="run_translations")
    op.drop_table("run_translations")
    op.drop_column("app_settings", "translation_autogenerate")
    op.drop_column("app_settings", "translation_target_language")
