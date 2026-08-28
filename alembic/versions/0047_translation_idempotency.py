"""Add idempotency_key, replay_digest, and immutability trigger to
run_translations, closing the integrity gaps identified in ADR 0008.

Existing rows receive no backfill (completed generations needing no replay
protection). The writer populates both columns for all new rows. The
immutability trigger matches the run_enrichment_assets pattern: reject UPDATE
(except the write-once superseded_by_translation_id stamp) and DELETE.

Revision ID: 0047
Revises: 0046
"""

import sqlalchemy as sa
from alembic import op

revision: str = "0047"
down_revision: str = "0046"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "run_translations",
        sa.Column("idempotency_key", sa.Text(), nullable=True),
    )
    op.create_unique_constraint(
        "run_translations_idempotency_key",
        "run_translations",
        ["idempotency_key"],
    )
    op.add_column(
        "run_translations",
        sa.Column("replay_digest", sa.Text(), nullable=True),
    )

    op.execute("""
        CREATE FUNCTION run_translations_content_immutable() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            IF TG_OP = 'INSERT' THEN
                IF NEW.superseded_by_translation_id IS NOT NULL THEN
                    RAISE EXCEPTION
                        'run_translations rows are born unsuperseded';
                END IF;
                RETURN NEW;
            END IF;
            IF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION 'run_translations is immutable (DELETE blocked)';
            END IF;
            IF NEW.id IS DISTINCT FROM OLD.id
               OR NEW.pipeline_run_id IS DISTINCT FROM OLD.pipeline_run_id
               OR NEW.target_language IS DISTINCT FROM OLD.target_language
               OR NEW.generation IS DISTINCT FROM OLD.generation
               OR NEW.source_language IS DISTINCT FROM OLD.source_language
               OR NEW.lines IS DISTINCT FROM OLD.lines
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
                    'run_translations content is immutable (only supersession may be stamped)';
            END IF;
            IF OLD.superseded_by_translation_id IS NOT NULL
               AND NEW.superseded_by_translation_id
                   IS DISTINCT FROM OLD.superseded_by_translation_id
            THEN
                RAISE EXCEPTION 'run_translations supersession is write-once';
            END IF;
            IF OLD.superseded_by_translation_id IS NULL
               AND NEW.superseded_by_translation_id IS NOT NULL
            THEN
                PERFORM 1 FROM run_translations t
                 WHERE t.id = NEW.superseded_by_translation_id
                   AND t.pipeline_run_id = OLD.pipeline_run_id
                   AND t.target_language = OLD.target_language
                   AND t.generation > OLD.generation;
                IF NOT FOUND THEN
                    RAISE EXCEPTION
                        'run_translations supersession must point to a newer generation of the same run and language';
                END IF;
            END IF;
            RETURN NEW;
        END $$
    """)
    op.execute("""
        CREATE TRIGGER run_translations_content_immutable_trigger
        BEFORE INSERT OR UPDATE OR DELETE ON run_translations
        FOR EACH ROW EXECUTE FUNCTION run_translations_content_immutable()
    """)


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER run_translations_content_immutable_trigger ON run_translations"
    )
    op.execute("DROP FUNCTION run_translations_content_immutable()")
    op.drop_constraint("run_translations_idempotency_key", "run_translations")
    op.drop_column("run_translations", "replay_digest")
    op.drop_column("run_translations", "idempotency_key")
