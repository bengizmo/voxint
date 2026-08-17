"""transcript_segments: nullable per-segment word timings (JSONB)

Issue #59 (click-to-split / reassign segments): a word-boundary split needs
per-word timing, which the whisper service already computes (``word_timestamps=
True``) but voxint historically dropped at the client seam. This adds a nullable
``words`` JSONB column holding the segment's bucketed words — a list of
``{start, end, word, confidence}`` in transcript order.

Nullable, no server default, no backfill: runs transcribed before #59 (and
providers without word timing) keep ``words = NULL``. Word data is derived detail
layered beside the immutable ASR evidence (raw_text/interval remain the numerics
contract) — never written onto existing rows, so this migration touches schema
only, not data.

A ``jsonb_typeof(words) = 'array'`` CHECK backstops the shape, matching the
convention every other JSONB column in this schema uses. The column stores SQL
NULL for a wordless run (none_as_null=True on the model side), an array
(possibly empty) otherwise — never JSONB 'null', which the CHECK would reject.

No index: words are never queried by content, only read alongside their segment.

Downgrade drops the column (and its CHECK), discarding any captured word timings
(they are recomputable by re-transcribing; the ASR text/timing that IS the
contract is untouched).

Revision ID: 0022
Revises: 0021
Create Date: 2026-08-17 16:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0022"
down_revision: str | None = "0021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "transcript_segments",
        sa.Column("words", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_check_constraint(
        "transcript_segments_words_array_check",
        "transcript_segments",
        "words IS NULL OR jsonb_typeof(words) = 'array'",
    )


def downgrade() -> None:
    # DROP COLUMN drops the column's CHECK constraint with it.
    op.drop_column("transcript_segments", "words")
