"""enrichment draft schema: producer runs, candidates, evidence, review trail (issue #37)

Revision ID: 0010
Revises: 0009
Create Date: 2026-08-14 22:30:00.000000

Machine-derived claims about speakers and runs live here as reviewable,
evidence-backed drafts. Four triggers guard the layer's integrity at the
persistence boundary (not just by convention, mirroring adjudication_decisions):

- enrichment_producer_runs is write-once (a completed invocation is history);
- profile_review_decisions is append-only (human trail, terminal per candidate);
- enrichment_candidates content is immutable — rows are born unsuperseded and
  only a write-once supersession stamp may later be applied;
- enrichment_candidate_evidence is append-only.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "enrichment_producer_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("producer", sa.Text(), nullable=False),
        sa.Column("producer_version", sa.Text(), nullable=False),
        sa.Column("target_kind", sa.Text(), nullable=False),
        sa.Column("speaker_id", sa.Uuid(), nullable=True),
        sa.Column("pipeline_run_id", sa.Uuid(), nullable=True),
        sa.Column("diarization_label", sa.Text(), nullable=True),
        sa.Column("covered_fields", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("config", postgresql.JSONB(), nullable=True),
        sa.Column("config_schema_version", sa.Integer(), nullable=True),
        sa.Column("idempotency_key", sa.Text(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "length(trim(producer)) > 0",
            name="enrichment_producer_runs_producer_nonempty_check",
        ),
        sa.CheckConstraint(
            "length(trim(producer_version)) > 0",
            name="enrichment_producer_runs_producer_version_nonempty_check",
        ),
        sa.CheckConstraint(
            "target_kind IN ('speaker', 'run', 'run_label')",
            name="enrichment_producer_runs_target_kind_check",
        ),
        sa.CheckConstraint(
            "target_kind != 'speaker' OR (speaker_id IS NOT NULL"
            " AND pipeline_run_id IS NULL AND diarization_label IS NULL)",
            name="enrichment_producer_runs_speaker_shape_check",
        ),
        sa.CheckConstraint(
            "target_kind != 'run' OR (pipeline_run_id IS NOT NULL"
            " AND speaker_id IS NULL AND diarization_label IS NULL)",
            name="enrichment_producer_runs_run_shape_check",
        ),
        sa.CheckConstraint(
            "target_kind != 'run_label' OR (pipeline_run_id IS NOT NULL"
            " AND diarization_label IS NOT NULL AND speaker_id IS NULL)",
            name="enrichment_producer_runs_run_label_shape_check",
        ),
        sa.CheckConstraint(
            "diarization_label IS NULL OR length(trim(diarization_label)) > 0",
            name="enrichment_producer_runs_label_nonempty_check",
        ),
        sa.CheckConstraint(
            "cardinality(covered_fields) >= 1 AND covered_fields <@ "
            "ARRAY['name', 'bio', 'affiliation', 'link']::text[]",
            name="enrichment_producer_runs_covered_fields_check",
        ),
        sa.CheckConstraint(
            "generation >= 1", name="enrichment_producer_runs_generation_check"
        ),
        sa.CheckConstraint(
            "outcome IN ('found', 'none')",
            name="enrichment_producer_runs_outcome_check",
        ),
        sa.CheckConstraint(
            "(config IS NULL) = (config_schema_version IS NULL)",
            name="enrichment_producer_runs_config_pair_check",
        ),
        sa.CheckConstraint(
            "config_schema_version IS NULL OR config_schema_version >= 1",
            name="enrichment_producer_runs_config_schema_version_check",
        ),
        sa.CheckConstraint(
            "config IS NULL OR jsonb_typeof(config) = 'object'",
            name="enrichment_producer_runs_config_object_check",
        ),
        sa.CheckConstraint(
            "completed_at >= started_at",
            name="enrichment_producer_runs_completed_after_started_check",
        ),
        sa.ForeignKeyConstraint(
            ["speaker_id"],
            ["speakers.id"],
            name="enrichment_producer_runs_speaker_id_fkey",
        ),
        sa.ForeignKeyConstraint(
            ["pipeline_run_id"],
            ["pipeline_runs.id"],
            name="enrichment_producer_runs_pipeline_run_id_fkey",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "idempotency_key", name="enrichment_producer_runs_idempotency_key"
        ),
    )
    op.create_index(
        op.f("ix_enrichment_producer_runs_speaker_id"),
        "enrichment_producer_runs",
        ["speaker_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_enrichment_producer_runs_pipeline_run_id"),
        "enrichment_producer_runs",
        ["pipeline_run_id"],
        unique=False,
    )

    op.create_table(
        "enrichment_candidates",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("producer_run_id", sa.Uuid(), nullable=False),
        sa.Column("target_kind", sa.Text(), nullable=False),
        sa.Column("speaker_id", sa.Uuid(), nullable=True),
        sa.Column("pipeline_run_id", sa.Uuid(), nullable=True),
        sa.Column("diarization_label", sa.Text(), nullable=True),
        sa.Column("field", sa.Text(), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column(
            "score_components",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("superseded_by_producer_run_id", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "target_kind IN ('speaker', 'run', 'run_label')",
            name="enrichment_candidates_target_kind_check",
        ),
        sa.CheckConstraint(
            "target_kind != 'speaker' OR (speaker_id IS NOT NULL"
            " AND pipeline_run_id IS NULL AND diarization_label IS NULL)",
            name="enrichment_candidates_speaker_shape_check",
        ),
        sa.CheckConstraint(
            "target_kind != 'run' OR (pipeline_run_id IS NOT NULL"
            " AND speaker_id IS NULL AND diarization_label IS NULL)",
            name="enrichment_candidates_run_shape_check",
        ),
        sa.CheckConstraint(
            "target_kind != 'run_label' OR (pipeline_run_id IS NOT NULL"
            " AND diarization_label IS NOT NULL AND speaker_id IS NULL)",
            name="enrichment_candidates_run_label_shape_check",
        ),
        sa.CheckConstraint(
            "diarization_label IS NULL OR length(trim(diarization_label)) > 0",
            name="enrichment_candidates_label_nonempty_check",
        ),
        sa.CheckConstraint(
            "field IN ('name', 'bio', 'affiliation', 'link')",
            name="enrichment_candidates_field_check",
        ),
        sa.CheckConstraint(
            "length(trim(value)) > 0 AND char_length(value) <= 4000",
            name="enrichment_candidates_value_check",
        ),
        sa.CheckConstraint(
            "score IS NULL OR (score >= 0 AND score <= 1)",
            name="enrichment_candidates_score_check",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(score_components) = 'object'",
            name="enrichment_candidates_score_components_object_check",
        ),
        sa.ForeignKeyConstraint(
            ["producer_run_id"],
            ["enrichment_producer_runs.id"],
            name="enrichment_candidates_producer_run_id_fkey",
        ),
        sa.ForeignKeyConstraint(
            ["speaker_id"],
            ["speakers.id"],
            name="enrichment_candidates_speaker_id_fkey",
        ),
        sa.ForeignKeyConstraint(
            ["pipeline_run_id"],
            ["pipeline_runs.id"],
            name="enrichment_candidates_pipeline_run_id_fkey",
        ),
        sa.ForeignKeyConstraint(
            ["superseded_by_producer_run_id"],
            ["enrichment_producer_runs.id"],
            name="enrichment_candidates_superseded_by_fkey",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_enrichment_candidates_producer_run_id"),
        "enrichment_candidates",
        ["producer_run_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_enrichment_candidates_speaker_id"),
        "enrichment_candidates",
        ["speaker_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_enrichment_candidates_pipeline_run_id"),
        "enrichment_candidates",
        ["pipeline_run_id"],
        unique=False,
    )

    op.create_table(
        "enrichment_candidate_evidence",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("candidate_id", sa.Uuid(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("source_metadata_id", sa.Uuid(), nullable=True),
        sa.Column("source_field", sa.Text(), nullable=True),
        sa.Column("transcript_segment_id", sa.Uuid(), nullable=True),
        sa.Column("timestamp_seconds", sa.Float(), nullable=True),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("snippet", sa.Text(), nullable=True),
        sa.Column("detail", postgresql.JSONB(), nullable=True),
        sa.Column("detail_schema_version", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "ordinal >= 0", name="enrichment_candidate_evidence_ordinal_nonneg_check"
        ),
        sa.CheckConstraint(
            "kind IN ('metadata_field', 'transcript_segment', 'url')",
            name="enrichment_candidate_evidence_kind_check",
        ),
        sa.CheckConstraint(
            "kind != 'metadata_field' OR (source_metadata_id IS NOT NULL"
            " AND source_field IS NOT NULL AND transcript_segment_id IS NULL"
            " AND timestamp_seconds IS NULL AND url IS NULL AND retrieved_at IS NULL)",
            name="enrichment_candidate_evidence_metadata_shape_check",
        ),
        sa.CheckConstraint(
            "kind != 'transcript_segment' OR (transcript_segment_id IS NOT NULL"
            " AND source_metadata_id IS NULL AND source_field IS NULL"
            " AND url IS NULL AND retrieved_at IS NULL)",
            name="enrichment_candidate_evidence_transcript_shape_check",
        ),
        sa.CheckConstraint(
            "kind != 'url' OR (url IS NOT NULL"
            " AND source_metadata_id IS NULL AND source_field IS NULL"
            " AND transcript_segment_id IS NULL AND timestamp_seconds IS NULL)",
            name="enrichment_candidate_evidence_url_shape_check",
        ),
        sa.CheckConstraint(
            "source_field IS NULL OR length(trim(source_field)) > 0",
            name="enrichment_candidate_evidence_source_field_nonempty_check",
        ),
        sa.CheckConstraint(
            "timestamp_seconds IS NULL OR timestamp_seconds >= 0",
            name="enrichment_candidate_evidence_timestamp_nonneg_check",
        ),
        sa.CheckConstraint(
            "url IS NULL OR char_length(url) <= 2048",
            name="enrichment_candidate_evidence_url_length_check",
        ),
        sa.CheckConstraint(
            "snippet IS NULL OR char_length(snippet) <= 1000",
            name="enrichment_candidate_evidence_snippet_length_check",
        ),
        sa.CheckConstraint(
            "(detail IS NULL) = (detail_schema_version IS NULL)",
            name="enrichment_candidate_evidence_detail_pair_check",
        ),
        sa.CheckConstraint(
            "detail_schema_version IS NULL OR detail_schema_version >= 1",
            name="enrichment_candidate_evidence_detail_schema_version_check",
        ),
        sa.CheckConstraint(
            "detail IS NULL OR jsonb_typeof(detail) = 'object'",
            name="enrichment_candidate_evidence_detail_object_check",
        ),
        sa.ForeignKeyConstraint(
            ["candidate_id"],
            ["enrichment_candidates.id"],
            name="enrichment_candidate_evidence_candidate_id_fkey",
        ),
        sa.ForeignKeyConstraint(
            ["source_metadata_id"],
            ["media_source_metadata.id"],
            name="enrichment_candidate_evidence_source_metadata_id_fkey",
        ),
        sa.ForeignKeyConstraint(
            ["transcript_segment_id"],
            ["transcript_segments.id"],
            name="enrichment_candidate_evidence_transcript_segment_id_fkey",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "candidate_id", "ordinal", name="enrichment_candidate_evidence_ordinal_key"
        ),
    )
    op.create_index(
        op.f("ix_enrichment_candidate_evidence_candidate_id"),
        "enrichment_candidate_evidence",
        ["candidate_id"],
        unique=False,
    )

    op.create_table(
        "profile_review_decisions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("candidate_id", sa.Uuid(), nullable=False),
        sa.Column("decision", sa.Text(), nullable=False),
        sa.Column("operator", sa.Text(), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("idempotency_key", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "decision IN ('accept', 'reject')",
            name="profile_review_decisions_decision_check",
        ),
        sa.CheckConstraint(
            "length(trim(operator)) > 0",
            name="profile_review_decisions_operator_nonempty_check",
        ),
        sa.CheckConstraint(
            "note IS NULL OR char_length(note) <= 2000",
            name="profile_review_decisions_note_length_check",
        ),
        sa.ForeignKeyConstraint(
            ["candidate_id"],
            ["enrichment_candidates.id"],
            name="profile_review_decisions_candidate_id_fkey",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "candidate_id", name="profile_review_decisions_candidate_key"
        ),
        sa.UniqueConstraint(
            "idempotency_key", name="profile_review_decisions_idempotency_key"
        ),
    )

    # A completed invocation is history: its scope, coverage, generation, and
    # outcome anchor candidate lineage and supersession, so nothing may ever
    # rewrite or remove it.
    op.execute("""
        CREATE FUNCTION enrichment_producer_runs_append_only() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION
                'enrichment_producer_runs is append-only (% blocked)', TG_OP;
        END $$
    """)
    op.execute("""
        CREATE TRIGGER enrichment_producer_runs_append_only_trigger
        BEFORE UPDATE OR DELETE ON enrichment_producer_runs
        FOR EACH ROW EXECUTE FUNCTION enrichment_producer_runs_append_only()
    """)

    # The human trail is append-only at the persistence boundary, exactly like
    # adjudication_decisions.
    op.execute("""
        CREATE FUNCTION profile_review_decisions_append_only() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'profile_review_decisions is append-only (% blocked)', TG_OP;
        END $$
    """)
    op.execute("""
        CREATE TRIGGER profile_review_decisions_append_only_trigger
        BEFORE UPDATE OR DELETE ON profile_review_decisions
        FOR EACH ROW EXECUTE FUNCTION profile_review_decisions_append_only()
    """)

    # Claim content is immutable; the only permitted mutation is stamping the
    # supersession fact once (NULL -> a producer-run id). History may be
    # referenced by the review trail, so DELETE is always blocked, and a row
    # can never be born already-superseded.
    op.execute("""
        CREATE FUNCTION enrichment_candidates_content_immutable() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            IF TG_OP = 'INSERT' THEN
                IF NEW.superseded_by_producer_run_id IS NOT NULL THEN
                    RAISE EXCEPTION
                        'enrichment_candidates rows are born unsuperseded';
                END IF;
                RETURN NEW;
            END IF;
            IF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION 'enrichment_candidates is immutable (DELETE blocked)';
            END IF;
            IF NEW.id IS DISTINCT FROM OLD.id
               OR NEW.producer_run_id IS DISTINCT FROM OLD.producer_run_id
               OR NEW.target_kind IS DISTINCT FROM OLD.target_kind
               OR NEW.speaker_id IS DISTINCT FROM OLD.speaker_id
               OR NEW.pipeline_run_id IS DISTINCT FROM OLD.pipeline_run_id
               OR NEW.diarization_label IS DISTINCT FROM OLD.diarization_label
               OR NEW.field IS DISTINCT FROM OLD.field
               OR NEW.value IS DISTINCT FROM OLD.value
               OR NEW.score IS DISTINCT FROM OLD.score
               OR NEW.score_components IS DISTINCT FROM OLD.score_components
               OR NEW.created_at IS DISTINCT FROM OLD.created_at
            THEN
                RAISE EXCEPTION
                    'enrichment_candidates content is immutable (only supersession may be stamped)';
            END IF;
            IF OLD.superseded_by_producer_run_id IS NOT NULL
               AND NEW.superseded_by_producer_run_id
                   IS DISTINCT FROM OLD.superseded_by_producer_run_id
            THEN
                RAISE EXCEPTION 'enrichment_candidates supersession is write-once';
            END IF;
            RETURN NEW;
        END $$
    """)
    op.execute("""
        CREATE TRIGGER enrichment_candidates_content_immutable_trigger
        BEFORE INSERT OR UPDATE OR DELETE ON enrichment_candidates
        FOR EACH ROW EXECUTE FUNCTION enrichment_candidates_content_immutable()
    """)

    # Evidence never changes after the claim is recorded.
    op.execute("""
        CREATE FUNCTION enrichment_candidate_evidence_append_only() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION
                'enrichment_candidate_evidence is append-only (% blocked)', TG_OP;
        END $$
    """)
    op.execute("""
        CREATE TRIGGER enrichment_candidate_evidence_append_only_trigger
        BEFORE UPDATE OR DELETE ON enrichment_candidate_evidence
        FOR EACH ROW EXECUTE FUNCTION enrichment_candidate_evidence_append_only()
    """)


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER IF EXISTS enrichment_candidate_evidence_append_only_trigger"
        " ON enrichment_candidate_evidence"
    )
    op.execute("DROP FUNCTION IF EXISTS enrichment_candidate_evidence_append_only()")
    op.execute(
        "DROP TRIGGER IF EXISTS enrichment_candidates_content_immutable_trigger"
        " ON enrichment_candidates"
    )
    op.execute("DROP FUNCTION IF EXISTS enrichment_candidates_content_immutable()")
    op.execute(
        "DROP TRIGGER IF EXISTS profile_review_decisions_append_only_trigger"
        " ON profile_review_decisions"
    )
    op.execute("DROP FUNCTION IF EXISTS profile_review_decisions_append_only()")
    op.execute(
        "DROP TRIGGER IF EXISTS enrichment_producer_runs_append_only_trigger"
        " ON enrichment_producer_runs"
    )
    op.execute("DROP FUNCTION IF EXISTS enrichment_producer_runs_append_only()")
    op.drop_table("profile_review_decisions")
    op.drop_index(
        op.f("ix_enrichment_candidate_evidence_candidate_id"),
        table_name="enrichment_candidate_evidence",
    )
    op.drop_table("enrichment_candidate_evidence")
    op.drop_index(
        op.f("ix_enrichment_candidates_pipeline_run_id"),
        table_name="enrichment_candidates",
    )
    op.drop_index(
        op.f("ix_enrichment_candidates_speaker_id"), table_name="enrichment_candidates"
    )
    op.drop_index(
        op.f("ix_enrichment_candidates_producer_run_id"),
        table_name="enrichment_candidates",
    )
    op.drop_table("enrichment_candidates")
    op.drop_index(
        op.f("ix_enrichment_producer_runs_pipeline_run_id"),
        table_name="enrichment_producer_runs",
    )
    op.drop_index(
        op.f("ix_enrichment_producer_runs_speaker_id"),
        table_name="enrichment_producer_runs",
    )
    op.drop_table("enrichment_producer_runs")
