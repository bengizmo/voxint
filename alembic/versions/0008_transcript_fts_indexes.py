"""transcript full-text search: GIN expression indexes (issue #8)

Two expression indexes over ``transcript_segments`` — no stored column, no
trigger, no backfill. Building the index scans every existing row, and the
planner keeps it fresh across all writers (transcribe's delete-then-insert,
enhancement's per-batch UPDATE, the tutorial seed) with no writer changes.

Both text variants are indexed separately, NOT ``coalesce(enhanced_text,
raw_text)``: enhancement rewrites words (that is its purpose), so a
coalesced document would make the raw ASR rendering of a term unfindable
the moment its batch is enhanced — and the reverse for enhanced-only terms.
Queries OR two ``@@`` predicates; ``to_tsvector('english', NULL)`` is NULL
and NULL ``@@`` is falsy in OR, so no coalesce is needed on the enhanced
side. Two indexes also keep quoted-phrase semantics clean (no false phrase
matches across a raw/enhanced concatenation seam).

Dictionary: ``english`` — stemming recall (compressors→compressor,
leaking→leak) fits the dominant "find the run where we discussed X" query.
Trade-offs accepted and pinned by tests: stopword-only queries match
nothing (empty tsquery); proper nouns stem symmetrically at index and query
time. Switching to ``simple`` later is a cheap index rebuild in a new
migration, not a data migration. The literal expressions here must stay in
lockstep with ``voxint.db.search`` — a contract test pins the agreement.

Revision ID: 0008
Revises: 0007
Create Date: 2026-08-14 20:30:00.000000
"""
from collections.abc import Sequence

from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX transcript_segments_raw_fts_idx "
        "ON transcript_segments USING gin (to_tsvector('english', raw_text))"
    )
    op.execute(
        "CREATE INDEX transcript_segments_enhanced_fts_idx "
        "ON transcript_segments USING gin (to_tsvector('english', enhanced_text))"
    )


def downgrade() -> None:
    op.execute("DROP INDEX transcript_segments_enhanced_fts_idx")
    op.execute("DROP INDEX transcript_segments_raw_fts_idx")
