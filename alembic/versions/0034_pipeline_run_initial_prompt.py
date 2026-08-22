"""pipeline_runs: record the whisper initial_prompt each run decoded with

Issue #123 (project glossary). Adds one nullable text column to ``pipeline_runs``:

- ``initial_prompt`` — the bounded whisper ``initial_prompt`` this run actually
  transcribed with, stamped by the transcribe stage. It is the rendered join of
  the effective vocabulary (the selected domain pack's words plus the operator's
  live glossary, deduped and capped). The frozen ``domain_pack`` snapshot records
  only the pack's words; the operator's glossary (``app_settings.vocabulary``) is
  unioned LIVE at run start, so without this column the proper nouns a run was
  actually told about are unrecoverable.

Nullable, no default: NULL means a run with no vocabulary (the prompt was empty)
or a legacy run transcribed before this column existed. This is provenance only;
it changes nothing the model decodes.

Revision ID: 0034
Revises: 0033
Create Date: 2026-08-22 09:30:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0034"
down_revision: str | None = "0033"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "pipeline_runs",
        sa.Column("initial_prompt", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("pipeline_runs", "initial_prompt")
