"""LLM clean-up text variant: job and result tables (#758)

Two additive tables, mirroring transcript translation (0038 + 0047):

- ``run_cleanups``: immutable clean-up generations, one row per (run,
  generation). ``lines`` is a versioned JSONB snapshot (frozen source text,
  rendered cleaned text, deleted anchor ranges, outcome), ``counts`` the
  writer-derived outcome tally and ``config`` the provenance (prompt version,
  filler-list snapshot, batch knobs). The run-level ``source_content_hash`` is
  the only freshness authority. A trigger rejects DELETE and every UPDATE
  except the one-time ``superseded_by_cleanup_id`` stamp, which must point to
  a newer, still unsuperseded generation of the same run. Head uniqueness is
  held by the writer's advisory lock and atomic supersede, not an index.
- ``cleanup_jobs``: mutable orchestration state (queued -> running ->
  terminal) with a one-active-per-run partial unique index, the enqueue-time
  source hash, and a result link that is set exactly when the job succeeded.

Revision ID: 0073
Revises: 0072
Create Date: 2026-10-06 09:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0073"
down_revision: str | None = "0072"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "run_cleanups",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "pipeline_run_id",
            sa.Uuid(),
            sa.ForeignKey("pipeline_runs.id"),
            nullable=False,
        ),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("lines", JSONB(), nullable=False),
        sa.Column("counts", JSONB(), nullable=False),
        sa.Column("config", JSONB(), nullable=False),
        sa.Column("payload_schema_version", sa.Integer(), nullable=False),
        sa.Column("producer", sa.Text(), nullable=False),
        sa.Column("producer_version", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("source_content_hash", sa.Text(), nullable=False),
        sa.Column("idempotency_key", sa.Text(), nullable=False),
        sa.Column("replay_digest", sa.Text(), nullable=False),
        sa.Column(
            "superseded_by_cleanup_id",
            sa.Uuid(),
            sa.ForeignKey("run_cleanups.id"),
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
            "pipeline_run_id", "generation", name="run_cleanups_generation_key"
        ),
        sa.UniqueConstraint("idempotency_key", name="run_cleanups_idempotency_key"),
        sa.CheckConstraint("generation >= 1", name="run_cleanups_generation_check"),
        sa.CheckConstraint(
            "jsonb_typeof(lines) = 'array'", name="run_cleanups_lines_array_check"
        ),
        sa.CheckConstraint(
            "jsonb_typeof(counts) = 'object'", name="run_cleanups_counts_object_check"
        ),
        sa.CheckConstraint(
            "jsonb_typeof(config) = 'object'", name="run_cleanups_config_object_check"
        ),
        sa.CheckConstraint(
            "payload_schema_version >= 1", name="run_cleanups_payload_version_check"
        ),
        sa.CheckConstraint(
            "length(trim(producer)) > 0", name="run_cleanups_producer_nonempty_check"
        ),
        sa.CheckConstraint(
            "length(trim(producer_version)) > 0",
            name="run_cleanups_producer_version_nonempty_check",
        ),
        sa.CheckConstraint(
            "length(trim(model)) > 0", name="run_cleanups_model_nonempty_check"
        ),
        sa.CheckConstraint(
            "source_content_hash ~ '^[0-9a-f]{64}$'",
            name="run_cleanups_source_hash_check",
        ),
        sa.CheckConstraint(
            "length(trim(idempotency_key)) > 0",
            name="run_cleanups_idempotency_key_nonempty_check",
        ),
        sa.CheckConstraint(
            "replay_digest ~ '^[0-9a-f]{64}$'", name="run_cleanups_replay_digest_check"
        ),
        sa.CheckConstraint(
            "completed_at >= started_at",
            name="run_cleanups_completed_after_started_check",
        ),
    )
    op.create_index(
        "ix_run_cleanups_pipeline_run_id", "run_cleanups", ["pipeline_run_id"]
    )
    # No one-current partial unique index (the run_translations precedent): the
    # writer's insert-then-supersede transaction transiently holds two
    # unsuperseded rows; the advisory lock + atomic supersede hold the head.

    op.execute("""
        CREATE FUNCTION run_cleanups_content_immutable() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            IF TG_OP = 'INSERT' THEN
                IF NEW.superseded_by_cleanup_id IS NOT NULL THEN
                    RAISE EXCEPTION 'run_cleanups rows are born unsuperseded';
                END IF;
                RETURN NEW;
            END IF;
            IF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION 'run_cleanups is immutable (DELETE blocked)';
            END IF;
            IF NEW.id IS DISTINCT FROM OLD.id
               OR NEW.pipeline_run_id IS DISTINCT FROM OLD.pipeline_run_id
               OR NEW.generation IS DISTINCT FROM OLD.generation
               OR NEW.lines IS DISTINCT FROM OLD.lines
               OR NEW.counts IS DISTINCT FROM OLD.counts
               OR NEW.config IS DISTINCT FROM OLD.config
               OR NEW.payload_schema_version IS DISTINCT FROM OLD.payload_schema_version
               OR NEW.producer IS DISTINCT FROM OLD.producer
               OR NEW.producer_version IS DISTINCT FROM OLD.producer_version
               OR NEW.model IS DISTINCT FROM OLD.model
               OR NEW.source_content_hash IS DISTINCT FROM OLD.source_content_hash
               OR NEW.idempotency_key IS DISTINCT FROM OLD.idempotency_key
               OR NEW.replay_digest IS DISTINCT FROM OLD.replay_digest
               OR NEW.started_at IS DISTINCT FROM OLD.started_at
               OR NEW.completed_at IS DISTINCT FROM OLD.completed_at
               OR NEW.created_at IS DISTINCT FROM OLD.created_at
            THEN
                RAISE EXCEPTION
                    'run_cleanups content is immutable (only supersession may be stamped)';
            END IF;
            IF OLD.superseded_by_cleanup_id IS NOT NULL
               AND NEW.superseded_by_cleanup_id
                   IS DISTINCT FROM OLD.superseded_by_cleanup_id
            THEN
                RAISE EXCEPTION 'run_cleanups supersession is write-once';
            END IF;
            IF OLD.superseded_by_cleanup_id IS NULL
               AND NEW.superseded_by_cleanup_id IS NOT NULL
            THEN
                PERFORM 1 FROM run_cleanups c
                 WHERE c.id = NEW.superseded_by_cleanup_id
                   AND c.pipeline_run_id = OLD.pipeline_run_id
                   AND c.generation > OLD.generation
                   AND c.superseded_by_cleanup_id IS NULL;
                IF NOT FOUND THEN
                    RAISE EXCEPTION
                        'run_cleanups supersession must point to a newer, unsuperseded generation of the same run';
                END IF;
            END IF;
            RETURN NEW;
        END $$
    """)
    op.execute("""
        CREATE TRIGGER run_cleanups_content_immutable_trigger
        BEFORE INSERT OR UPDATE OR DELETE ON run_cleanups
        FOR EACH ROW EXECUTE FUNCTION run_cleanups_content_immutable()
    """)

    op.create_table(
        "cleanup_jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "pipeline_run_id",
            sa.Uuid(),
            sa.ForeignKey("pipeline_runs.id"),
            nullable=False,
        ),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False),
        sa.Column("config", JSONB(), nullable=False),
        sa.Column("source_content_hash", sa.Text(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "cleanup_id",
            sa.Uuid(),
            sa.ForeignKey("run_cleanups.id"),
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
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="cleanup_jobs_status_check",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(config) = 'object'", name="cleanup_jobs_config_object_check"
        ),
        sa.CheckConstraint(
            "source_content_hash ~ '^[0-9a-f]{64}$'",
            name="cleanup_jobs_source_hash_check",
        ),
        sa.CheckConstraint(
            "(cleanup_id IS NOT NULL) = (status = 'succeeded')",
            name="cleanup_jobs_result_iff_succeeded_check",
        ),
        sa.CheckConstraint(
            "started_at IS NULL OR started_at >= created_at",
            name="cleanup_jobs_started_after_created_check",
        ),
        sa.CheckConstraint(
            "finished_at IS NULL OR started_at IS NOT NULL",
            name="cleanup_jobs_finished_requires_started_check",
        ),
        sa.CheckConstraint(
            "finished_at IS NULL OR finished_at >= started_at",
            name="cleanup_jobs_finished_after_started_check",
        ),
    )
    op.create_index("ix_cleanup_jobs_pipeline_run_id", "cleanup_jobs", ["pipeline_run_id"])
    op.create_index(
        "cleanup_jobs_one_active_per_run",
        "cleanup_jobs",
        ["pipeline_run_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('queued', 'running')"),
    )


def downgrade() -> None:
    op.drop_index("cleanup_jobs_one_active_per_run", table_name="cleanup_jobs")
    op.drop_index("ix_cleanup_jobs_pipeline_run_id", table_name="cleanup_jobs")
    op.drop_table("cleanup_jobs")
    op.execute("DROP TRIGGER run_cleanups_content_immutable_trigger ON run_cleanups")
    op.execute("DROP FUNCTION run_cleanups_content_immutable()")
    op.drop_index("ix_run_cleanups_pipeline_run_id", table_name="run_cleanups")
    op.drop_table("run_cleanups")
