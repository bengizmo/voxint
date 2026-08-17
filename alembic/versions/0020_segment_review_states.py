"""segment_review_states: per-segment verified mark + operator corrected text

Issues #53 (verify-and-advance) and #58 (transcript text correction). One mutable
row per reviewed segment, UPSERTed latest-wins — NOT the append-only
adjudication ledger (verified/corrected is orthogonal to speaker attribution) and
NOT columns on the immutable transcript_segments observation row. ``raw_text``
stays the ASR evidence of record; a correction is written beside it, never over
it. See docs/plans/2026-08-16-2227_transcript-text-correction-provenance.md.

- ``verified_at`` NULL = unverified; a timestamp = operator-confirmed.
- ``corrected_text`` NULL = no correction; non-NULL = operator text taking
  display/export/search precedence (corrected > enhanced > raw). Empty is
  normalized to NULL by the writer, so it is never an empty rendering.
- FK ON DELETE CASCADE: re-transcription mints new segment ids; cascade keeps
  re-ingest from leaking orphan review rows.
- A PARTIAL GIN index makes corrected text full-text-searchable (a third
  rendering, never coalesced with raw/enhanced), sparse and cheap because
  corrected_text is NULL for most rows. Its expression must stay in lockstep
  with voxint.db.search (contract-tested) and the query in api/runs_query.py.

Revision ID: 0020
Revises: 0019
Create Date: 2026-08-16 23:05:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0020"
down_revision: str | None = "0019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Keep in lockstep with voxint.db.models.MAX_CORRECTED_TEXT_CHARS.
_MAX_CORRECTED = 20_000


def upgrade() -> None:
    op.create_table(
        "segment_review_states",
        sa.Column(
            "transcript_segment_id",
            sa.Uuid(),
            sa.ForeignKey("transcript_segments.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "pipeline_run_id",
            sa.Uuid(),
            sa.ForeignKey("pipeline_runs.id"),
            nullable=False,
        ),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("corrected_text", sa.Text(), nullable=True),
        sa.Column("corrected_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "(corrected_text IS NULL) = (corrected_at IS NULL)",
            name="segment_review_states_corrected_pair_check",
        ),
        sa.CheckConstraint(
            f"corrected_text IS NULL OR char_length(corrected_text) <= {_MAX_CORRECTED}",
            name="segment_review_states_corrected_len_check",
        ),
    )
    op.create_index(
        "ix_segment_review_states_run",
        "segment_review_states",
        ["pipeline_run_id"],
    )
    # Partial GIN: only reviewed-and-corrected rows carry lexemes.
    op.execute(
        "CREATE INDEX segment_review_states_corrected_fts_idx "
        "ON segment_review_states USING gin (to_tsvector('english', corrected_text)) "
        "WHERE corrected_text IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX segment_review_states_corrected_fts_idx")
    op.drop_index("ix_segment_review_states_run", table_name="segment_review_states")
    op.drop_table("segment_review_states")
