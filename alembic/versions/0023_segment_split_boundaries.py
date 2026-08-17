"""segment_split_boundaries: operator word-boundary splits (derived children)

Issue #59 (click-to-split / reassign segments), slice 2. An operator splits a
mis-split diarization segment at a WORD boundary. A split is NOT a new
``transcript_segments`` row (that would contaminate the immutable ASR observation
model and force segment_index re-indexing) and NOT a mutable overlay the
append-only ledger would have to point at. Instead a split is stored here as one
append-only CUT: ``word_index`` = "split before word i" on the immutable parent
segment. Children are DERIVED at read time from the cut set ``{0, cuts…,
word_count}`` — the parent row is never mutated.

- ``parent_segment_id`` FK ON DELETE CASCADE: re-transcription mints new segment
  ids; cascade keeps re-ingest from leaking orphan boundaries (mirrors
  ``segment_review_states``).
- ``pipeline_run_id`` denormalized (the one writer derives it from the parent) so
  the read path batch-loads every run's boundaries in one indexed query — no N+1
  when ``attributed_transcript`` expands children.
- ``UNIQUE(parent_segment_id, word_index)`` makes a split STRUCTURALLY idempotent:
  "split before word i" can exist at most once, so a replayed / double-clicked
  split is a no-op (the writer uses ON CONFLICT DO NOTHING) — no client nonce
  needed, unlike the temporal idempotency of the relabel ledger.
- ``word_index >= 1`` CHECK: a cut is INTERIOR — before word 0 (the segment
  start) is not a split. The UPPER bound (``word_index < word_count``) depends on
  the parent's word count, which is cross-table, so it is validated in the writer,
  not a CHECK (Postgres cannot express a cross-table CHECK).

No word content is copied here: children's text/timing are derived from the
parent's immutable ``words`` tokens at read time, never snapshotted.

Downgrade drops the table, discarding operator splits (the underlying ASR
evidence and word timings are untouched, so splits are re-creatable).

Revision ID: 0023
Revises: 0022
Create Date: 2026-08-17 17:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0023"
down_revision: str | None = "0022"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "segment_split_boundaries",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "pipeline_run_id",
            sa.Uuid(),
            sa.ForeignKey("pipeline_runs.id"),
            nullable=False,
        ),
        sa.Column(
            "parent_segment_id",
            sa.Uuid(),
            sa.ForeignKey("transcript_segments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("word_index", sa.Integer(), nullable=False),
        sa.Column("operator", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "word_index >= 1",
            name="segment_split_boundaries_word_index_interior_check",
        ),
        sa.UniqueConstraint(
            "parent_segment_id",
            "word_index",
            name="segment_split_boundaries_parent_word_key",
        ),
    )
    # Batch-load every boundary of a run in one indexed pass (read-path expansion).
    op.create_index(
        "ix_segment_split_boundaries_run",
        "segment_split_boundaries",
        ["pipeline_run_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_segment_split_boundaries_run", table_name="segment_split_boundaries"
    )
    op.drop_table("segment_split_boundaries")
