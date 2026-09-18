"""Provenance-preserving restart for adjudicated/enriched runs (#507).

Changes FK on adjudication_decisions.transcript_segment_id and
enrichment_candidate_evidence.transcript_segment_id from RESTRICT to
ON DELETE SET NULL, so segment deletion during restart detaches the
reference instead of failing.  The append-only triggers are narrowed to
allow only this specific SET NULL mutation and enforce void-before-detach
at the database level.  New ``detached_at`` and ``original_*`` columns
preserve the historical scope for audit.

Revision ID: 0066
Revises: 0065
Create Date: 2026-09-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0066"
down_revision: str | None = "0065"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── adjudication_decisions: provenance columns ──────────────────────
    op.add_column(
        "adjudication_decisions",
        sa.Column("detached_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "adjudication_decisions",
        sa.Column("original_transcript_segment_id", sa.Uuid(), nullable=True),
    )
    op.add_column(
        "adjudication_decisions",
        sa.Column("original_start_word_index", sa.Integer(), nullable=True),
    )
    op.add_column(
        "adjudication_decisions",
        sa.Column("original_end_word_index", sa.Integer(), nullable=True),
    )
    # Pairing CHECKs: detached_at and original_transcript_segment_id are
    # set together; original word-range only on detached rows that had a range.
    op.create_check_constraint(
        "adjudication_decisions_detach_pair_check",
        "adjudication_decisions",
        "(detached_at IS NULL) = (original_transcript_segment_id IS NULL)",
    )
    op.create_check_constraint(
        "adjudication_decisions_original_range_only_detached_check",
        "adjudication_decisions",
        "detached_at IS NOT NULL OR original_start_word_index IS NULL",
    )
    op.create_check_constraint(
        "adjudication_decisions_original_range_pair_check",
        "adjudication_decisions",
        "(original_start_word_index IS NULL) = (original_end_word_index IS NULL)",
    )

    # ── enrichment_candidate_evidence: provenance columns ───────────────
    op.add_column(
        "enrichment_candidate_evidence",
        sa.Column("detached_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "enrichment_candidate_evidence",
        sa.Column("original_transcript_segment_id", sa.Uuid(), nullable=True),
    )
    op.create_check_constraint(
        "enrichment_candidate_evidence_detach_pair_check",
        "enrichment_candidate_evidence",
        "(detached_at IS NULL) = (original_transcript_segment_id IS NULL)",
    )
    # Index for performant ON DELETE SET NULL cascade (was missing).
    op.create_index(
        "ix_enrichment_candidate_evidence_transcript_segment_id",
        "enrichment_candidate_evidence",
        ["transcript_segment_id"],
    )

    # ── FK: adjudication_decisions.transcript_segment_id → SET NULL ─────
    op.drop_constraint(
        "adjudication_decisions_transcript_segment_id_fkey",
        "adjudication_decisions",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "adjudication_decisions_transcript_segment_id_fkey",
        "adjudication_decisions",
        "transcript_segments",
        ["transcript_segment_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # ── FK: enrichment_candidate_evidence.transcript_segment_id → SET NULL
    op.drop_constraint(
        "enrichment_candidate_evidence_transcript_segment_id_fkey",
        "enrichment_candidate_evidence",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "enrichment_candidate_evidence_transcript_segment_id_fkey",
        "enrichment_candidate_evidence",
        "transcript_segments",
        ["transcript_segment_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # ── CHECK: inherit requires segment (conditional on detach) ─────────
    op.drop_constraint(
        "adjudication_decisions_inherit_segment_check",
        "adjudication_decisions",
        type_="check",
    )
    op.create_check_constraint(
        "adjudication_decisions_inherit_segment_check",
        "adjudication_decisions",
        "detached_at IS NOT NULL OR decision != 'inherit' OR transcript_segment_id IS NOT NULL",
    )

    # ── CHECK: evidence transcript shape (conditional on detach) ────────
    op.drop_constraint(
        "enrichment_candidate_evidence_transcript_shape_check",
        "enrichment_candidate_evidence",
        type_="check",
    )
    op.create_check_constraint(
        "enrichment_candidate_evidence_transcript_shape_check",
        "enrichment_candidate_evidence",
        "detached_at IS NOT NULL"
        " OR kind != 'transcript_segment'"
        " OR (transcript_segment_id IS NOT NULL"
        " AND source_metadata_id IS NULL AND source_field IS NULL"
        " AND url IS NULL AND retrieved_at IS NULL)",
    )

    # ── Trigger: adjudication_decisions (narrow detach path) ────────────
    #
    # The original trigger unconditionally blocks UPDATE and DELETE.  The
    # replacement allows exactly one UPDATE pattern: transcript_segment_id
    # going NOT NULL → NULL, provided a REVOKE already exists for the
    # decision (void-before-detach invariant, enforced in the database).
    # The trigger populates the original_* provenance columns and stamps
    # detached_at atomically.
    op.execute("""
        CREATE OR REPLACE FUNCTION adjudication_decisions_append_only()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION
                    'adjudication_decisions is append-only (DELETE blocked)';
            END IF;

            -- Narrow detach path: FK SET NULL from transcript_segments deletion.
            IF OLD.transcript_segment_id IS NOT NULL
               AND NEW.transcript_segment_id IS NULL
            THEN
                -- Void-before-detach: a REVOKE must exist for this decision.
                IF NOT EXISTS (
                    SELECT 1 FROM adjudication_decisions
                    WHERE decision = 'revoke'
                      AND voids_decision_id = OLD.id
                ) THEN
                    RAISE EXCEPTION
                        'adjudication_decisions: cannot detach decision %'
                        ' without a prior REVOKE', OLD.id;
                END IF;

                -- Content columns must be unchanged (the SET NULL cascade
                -- only touches transcript_segment_id).
                IF NEW.id                IS DISTINCT FROM OLD.id
                   OR NEW.pipeline_run_id IS DISTINCT FROM OLD.pipeline_run_id
                   OR NEW.diarization_label IS DISTINCT FROM OLD.diarization_label
                   OR NEW.decision        IS DISTINCT FROM OLD.decision
                   OR NEW.speaker_id      IS DISTINCT FROM OLD.speaker_id
                   OR NEW.operator        IS DISTINCT FROM OLD.operator
                   OR NEW.user_id         IS DISTINCT FROM OLD.user_id
                   OR NEW.idempotency_key IS DISTINCT FROM OLD.idempotency_key
                   OR NEW.voids_decision_id IS DISTINCT FROM OLD.voids_decision_id
                   OR NEW.created_at      IS DISTINCT FROM OLD.created_at
                THEN
                    RAISE EXCEPTION
                        'adjudication_decisions: detach may only null'
                        ' scope columns; content columns must be unchanged';
                END IF;

                -- Preserve original scope in provenance columns.
                NEW.original_transcript_segment_id := OLD.transcript_segment_id;
                NEW.original_start_word_index      := OLD.start_word_index;
                NEW.original_end_word_index        := OLD.end_word_index;
                -- Clear live scope columns.
                NEW.start_word_index := NULL;
                NEW.end_word_index   := NULL;
                NEW.detached_at      := now();
                RETURN NEW;
            END IF;

            RAISE EXCEPTION
                'adjudication_decisions is append-only (UPDATE blocked)';
        END $$
    """)

    # ── Trigger: enrichment_candidate_evidence (narrow detach path) ─────
    op.execute("""
        CREATE OR REPLACE FUNCTION enrichment_candidate_evidence_append_only()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION
                    'enrichment_candidate_evidence is append-only'
                    ' (DELETE blocked)';
            END IF;

            -- Narrow detach path: FK SET NULL from transcript_segments deletion.
            IF OLD.transcript_segment_id IS NOT NULL
               AND NEW.transcript_segment_id IS NULL
            THEN
                -- All non-FK columns must be unchanged.
                IF NEW.id             IS DISTINCT FROM OLD.id
                   OR NEW.candidate_id IS DISTINCT FROM OLD.candidate_id
                   OR NEW.ordinal      IS DISTINCT FROM OLD.ordinal
                   OR NEW.kind         IS DISTINCT FROM OLD.kind
                   OR NEW.source_metadata_id IS DISTINCT FROM OLD.source_metadata_id
                   OR NEW.source_field IS DISTINCT FROM OLD.source_field
                   OR NEW.timestamp_seconds IS DISTINCT FROM OLD.timestamp_seconds
                   OR NEW.url          IS DISTINCT FROM OLD.url
                   OR NEW.retrieved_at IS DISTINCT FROM OLD.retrieved_at
                   OR NEW.snippet      IS DISTINCT FROM OLD.snippet
                   OR NEW.detail       IS DISTINCT FROM OLD.detail
                   OR NEW.detail_schema_version IS DISTINCT FROM OLD.detail_schema_version
                   OR NEW.created_at   IS DISTINCT FROM OLD.created_at
                THEN
                    RAISE EXCEPTION
                        'enrichment_candidate_evidence: detach may only null'
                        ' transcript_segment_id; all other columns must be'
                        ' unchanged';
                END IF;

                NEW.original_transcript_segment_id := OLD.transcript_segment_id;
                NEW.detached_at := now();
                RETURN NEW;
            END IF;

            RAISE EXCEPTION
                'enrichment_candidate_evidence is append-only'
                ' (UPDATE blocked)';
        END $$
    """)


def downgrade() -> None:
    # Restore original trigger functions (unconditional append-only).
    op.execute("""
        CREATE OR REPLACE FUNCTION adjudication_decisions_append_only()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION
                'adjudication_decisions is append-only (% blocked)', TG_OP;
        END $$
    """)
    op.execute("""
        CREATE OR REPLACE FUNCTION enrichment_candidate_evidence_append_only()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION
                'enrichment_candidate_evidence is append-only (% blocked)',
                TG_OP;
        END $$
    """)

    # Restore original CHECK constraints.
    op.drop_constraint(
        "enrichment_candidate_evidence_transcript_shape_check",
        "enrichment_candidate_evidence",
        type_="check",
    )
    op.create_check_constraint(
        "enrichment_candidate_evidence_transcript_shape_check",
        "enrichment_candidate_evidence",
        "kind != 'transcript_segment' OR (transcript_segment_id IS NOT NULL"
        " AND source_metadata_id IS NULL AND source_field IS NULL"
        " AND url IS NULL AND retrieved_at IS NULL)",
    )
    op.drop_constraint(
        "adjudication_decisions_inherit_segment_check",
        "adjudication_decisions",
        type_="check",
    )
    op.create_check_constraint(
        "adjudication_decisions_inherit_segment_check",
        "adjudication_decisions",
        "decision != 'inherit' OR transcript_segment_id IS NOT NULL",
    )

    # Restore FK constraints to RESTRICT.
    op.drop_constraint(
        "enrichment_candidate_evidence_transcript_segment_id_fkey",
        "enrichment_candidate_evidence",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "enrichment_candidate_evidence_transcript_segment_id_fkey",
        "enrichment_candidate_evidence",
        "transcript_segments",
        ["transcript_segment_id"],
        ["id"],
    )
    op.drop_constraint(
        "adjudication_decisions_transcript_segment_id_fkey",
        "adjudication_decisions",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "adjudication_decisions_transcript_segment_id_fkey",
        "adjudication_decisions",
        "transcript_segments",
        ["transcript_segment_id"],
        ["id"],
    )

    # Drop new index and columns.
    op.drop_index(
        "ix_enrichment_candidate_evidence_transcript_segment_id",
        table_name="enrichment_candidate_evidence",
    )
    op.drop_constraint(
        "enrichment_candidate_evidence_detach_pair_check",
        "enrichment_candidate_evidence",
        type_="check",
    )
    op.drop_column("enrichment_candidate_evidence", "original_transcript_segment_id")
    op.drop_column("enrichment_candidate_evidence", "detached_at")

    op.drop_constraint(
        "adjudication_decisions_original_range_pair_check",
        "adjudication_decisions",
        type_="check",
    )
    op.drop_constraint(
        "adjudication_decisions_original_range_only_detached_check",
        "adjudication_decisions",
        type_="check",
    )
    op.drop_constraint(
        "adjudication_decisions_detach_pair_check",
        "adjudication_decisions",
        type_="check",
    )
    op.drop_column("adjudication_decisions", "original_end_word_index")
    op.drop_column("adjudication_decisions", "original_start_word_index")
    op.drop_column("adjudication_decisions", "original_transcript_segment_id")
    op.drop_column("adjudication_decisions", "detached_at")
