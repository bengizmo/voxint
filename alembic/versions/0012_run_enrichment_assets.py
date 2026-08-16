"""run_enrichment_assets + run_asset_jobs — run-level assets (issue #41)

Revision ID: 0012
Revises: 0011
Create Date: 2026-08-15 21:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "run_enrichment_assets",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("pipeline_run_id", sa.Uuid(), nullable=False),
        sa.Column("asset_kind", sa.Text(), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("payload_schema_version", sa.Integer(), nullable=False),
        sa.Column("producer", sa.Text(), nullable=False),
        sa.Column("producer_version", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("source_content_hash", sa.Text(), nullable=False),
        sa.Column("config", postgresql.JSONB(), nullable=True),
        sa.Column("config_schema_version", sa.Integer(), nullable=True),
        sa.Column("idempotency_key", sa.Text(), nullable=False),
        sa.Column("superseded_by_asset_id", sa.Uuid(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "asset_kind IN ('summary', 'topics', 'entity_mentions')",
            name="run_enrichment_assets_kind_check",
        ),
        sa.CheckConstraint("generation >= 1", name="run_enrichment_assets_generation_check"),
        sa.CheckConstraint(
            "jsonb_typeof(payload) = 'object'",
            name="run_enrichment_assets_payload_object_check",
        ),
        sa.CheckConstraint(
            "payload_schema_version >= 1",
            name="run_enrichment_assets_payload_version_check",
        ),
        sa.CheckConstraint(
            "length(trim(producer)) > 0",
            name="run_enrichment_assets_producer_nonempty_check",
        ),
        sa.CheckConstraint(
            "length(trim(producer_version)) > 0",
            name="run_enrichment_assets_producer_version_nonempty_check",
        ),
        sa.CheckConstraint(
            "length(trim(model)) > 0",
            name="run_enrichment_assets_model_nonempty_check",
        ),
        sa.CheckConstraint(
            "source_content_hash ~ '^[0-9a-f]{64}$'",
            name="run_enrichment_assets_source_hash_check",
        ),
        sa.CheckConstraint(
            "(config IS NULL) = (config_schema_version IS NULL)",
            name="run_enrichment_assets_config_pair_check",
        ),
        sa.CheckConstraint(
            "config IS NULL OR jsonb_typeof(config) = 'object'",
            name="run_enrichment_assets_config_object_check",
        ),
        sa.CheckConstraint(
            "config_schema_version IS NULL OR config_schema_version >= 1",
            name="run_enrichment_assets_config_version_check",
        ),
        sa.CheckConstraint(
            "completed_at >= started_at",
            name="run_enrichment_assets_completed_after_started_check",
        ),
        sa.ForeignKeyConstraint(["pipeline_run_id"], ["pipeline_runs.id"]),
        sa.ForeignKeyConstraint(["superseded_by_asset_id"], ["run_enrichment_assets.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "pipeline_run_id",
            "asset_kind",
            "generation",
            name="run_enrichment_assets_generation_key",
        ),
        sa.UniqueConstraint("idempotency_key", name="run_enrichment_assets_idempotency_key"),
    )
    op.create_index(
        "ix_run_enrichment_assets_pipeline_run_id",
        "run_enrichment_assets",
        ["pipeline_run_id"],
    )

    op.create_table(
        "run_asset_jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("pipeline_run_id", sa.Uuid(), nullable=False),
        sa.Column("asset_kind", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), server_default=sa.text("'queued'"), nullable=False),
        sa.Column(
            "cancel_requested",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("config", postgresql.JSONB(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("asset_id", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "asset_kind IN ('summary', 'topics', 'entity_mentions')",
            name="run_asset_jobs_kind_check",
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="run_asset_jobs_status_check",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(config) = 'object'",
            name="run_asset_jobs_config_object_check",
        ),
        sa.CheckConstraint(
            "started_at IS NULL OR started_at >= created_at",
            name="run_asset_jobs_started_after_created_check",
        ),
        sa.CheckConstraint(
            "finished_at IS NULL OR started_at IS NOT NULL",
            name="run_asset_jobs_finished_requires_started_check",
        ),
        sa.ForeignKeyConstraint(["pipeline_run_id"], ["pipeline_runs.id"]),
        sa.ForeignKeyConstraint(["asset_id"], ["run_enrichment_assets.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_run_asset_jobs_pipeline_run_id", "run_asset_jobs", ["pipeline_run_id"])
    # DB-enforced "one active job per (run, kind)" — the console's friendly
    # pre-check is racy check-then-insert; this index is the real invariant.
    op.create_index(
        "run_asset_jobs_one_active_per_run_kind",
        "run_asset_jobs",
        ["pipeline_run_id", "asset_kind"],
        unique=True,
        postgresql_where=sa.text("status IN ('queued', 'running')"),
    )

    # Asset content is immutable; the only permitted mutation is stamping the
    # supersession fact once (NULL -> a newer asset id). Jobs may reference an
    # asset, so DELETE is always blocked, and a row can never be born
    # already-superseded (same doctrine as enrichment_candidates).
    op.execute("""
        CREATE FUNCTION run_enrichment_assets_content_immutable() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            IF TG_OP = 'INSERT' THEN
                IF NEW.superseded_by_asset_id IS NOT NULL THEN
                    RAISE EXCEPTION
                        'run_enrichment_assets rows are born unsuperseded';
                END IF;
                RETURN NEW;
            END IF;
            IF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION 'run_enrichment_assets is immutable (DELETE blocked)';
            END IF;
            IF NEW.id IS DISTINCT FROM OLD.id
               OR NEW.pipeline_run_id IS DISTINCT FROM OLD.pipeline_run_id
               OR NEW.asset_kind IS DISTINCT FROM OLD.asset_kind
               OR NEW.generation IS DISTINCT FROM OLD.generation
               OR NEW.payload IS DISTINCT FROM OLD.payload
               OR NEW.payload_schema_version IS DISTINCT FROM OLD.payload_schema_version
               OR NEW.producer IS DISTINCT FROM OLD.producer
               OR NEW.producer_version IS DISTINCT FROM OLD.producer_version
               OR NEW.model IS DISTINCT FROM OLD.model
               OR NEW.source_content_hash IS DISTINCT FROM OLD.source_content_hash
               OR NEW.config IS DISTINCT FROM OLD.config
               OR NEW.config_schema_version IS DISTINCT FROM OLD.config_schema_version
               OR NEW.idempotency_key IS DISTINCT FROM OLD.idempotency_key
               OR NEW.started_at IS DISTINCT FROM OLD.started_at
               OR NEW.completed_at IS DISTINCT FROM OLD.completed_at
               OR NEW.created_at IS DISTINCT FROM OLD.created_at
            THEN
                RAISE EXCEPTION
                    'run_enrichment_assets content is immutable (only supersession may be stamped)';
            END IF;
            IF OLD.superseded_by_asset_id IS NOT NULL
               AND NEW.superseded_by_asset_id
                   IS DISTINCT FROM OLD.superseded_by_asset_id
            THEN
                RAISE EXCEPTION 'run_enrichment_assets supersession is write-once';
            END IF;
            IF OLD.superseded_by_asset_id IS NULL
               AND NEW.superseded_by_asset_id IS NOT NULL
            THEN
                -- The stamp must point at a NEWER generation of the SAME
                -- (run, kind): a valid-FK stamp to itself, another kind, or
                -- another run would silently hide the current asset with a
                -- lineage immutable rows can never repair.
                PERFORM 1 FROM run_enrichment_assets t
                 WHERE t.id = NEW.superseded_by_asset_id
                   AND t.pipeline_run_id = OLD.pipeline_run_id
                   AND t.asset_kind = OLD.asset_kind
                   AND t.generation > OLD.generation;
                IF NOT FOUND THEN
                    RAISE EXCEPTION
                        'run_enrichment_assets supersession must point to a newer generation of the same run and kind';
                END IF;
            END IF;
            RETURN NEW;
        END $$
    """)
    op.execute("""
        CREATE TRIGGER run_enrichment_assets_content_immutable_trigger
        BEFORE INSERT OR UPDATE OR DELETE ON run_enrichment_assets
        FOR EACH ROW EXECUTE FUNCTION run_enrichment_assets_content_immutable()
    """)


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER run_enrichment_assets_content_immutable_trigger ON run_enrichment_assets"
    )
    op.execute("DROP FUNCTION run_enrichment_assets_content_immutable()")
    op.drop_index("run_asset_jobs_one_active_per_run_kind", table_name="run_asset_jobs")
    op.drop_index("ix_run_asset_jobs_pipeline_run_id", table_name="run_asset_jobs")
    op.drop_table("run_asset_jobs")
    op.drop_index("ix_run_enrichment_assets_pipeline_run_id", table_name="run_enrichment_assets")
    op.drop_table("run_enrichment_assets")
