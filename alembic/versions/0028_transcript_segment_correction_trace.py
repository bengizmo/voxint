"""transcript_segments: deterministic correction trace + corrector version

Issue #82 (compose domain-pack corrections with enhancement, under the #78
non-LLM-correction epic). Adds two columns to ``transcript_segments`` so the
``enhance_match`` dual pass can persist an auditable, versioned correction trail
beside ``enhanced_text`` (never over ``raw_text``, which stays ASR evidence):

- ``correction_trace`` (JSONB, NOT NULL, server_default ``'[]'``) — either the
  empty array ``[]`` (no material correction, or a re-enhance reset) or the
  envelope ``{"version": int, "input_base": "raw"|"llm",
  "entries": [{"id","from","to","span":[s,e]}]}``. A NON-EMPTY ``entries`` list
  is the authoritative "a rule fired" signal the split machinery reads. The
  server default backfills every existing row to ``[]`` on upgrade, so the
  NOT-NULL constraint holds without a data-migration pass.
- ``corrector_version`` (Integer, nullable) — stamped with ``CORRECTOR_VERSION``
  when a material correction/enhancement is persisted. NULL = legacy pre-#82
  ``enhanced_text`` (rendered "enhanced (unversioned)", never recomputed) OR a
  row with no persisted enhanced output.

No shape CHECK: the value is a union (``[]`` array or an envelope object), so an
array-only check (as on ``words``) would be wrong and a union check adds no
safety over the app-controlled writes.

Revision ID: 0028
Revises: 0027
Create Date: 2026-08-18 19:45:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0028"
down_revision: str | None = "0027"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "transcript_segments",
        sa.Column(
            "correction_trace",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.add_column(
        "transcript_segments",
        sa.Column("corrector_version", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("transcript_segments", "corrector_version")
    op.drop_column("transcript_segments", "correction_trace")
