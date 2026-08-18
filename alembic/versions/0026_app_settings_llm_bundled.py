"""app_settings: bundled local-LLM enablement flag

Issue #67 (scoped Qwen bundle). Adds one nullable column,
``app_settings.llm_bundled_enabled``, so the settings console can turn the
optional bundled local LLM on/off at runtime, DB-row-wins-over-env (the
#74/#10 precedent). Purely additive and tri-state, exactly like the #74
feature-flag columns (``0024``):

- NULLable with NO server default: NULL = "inherit env ``LLM_BUNDLED_ENABLED``",
  non-NULL = row overrides. An existing row is unaffected (reads NULL ⇒ env still
  governs — the NULL-parity invariant), so no backfill and no behavior change on
  upgrade. When effective AND a bundled base URL is compose-injected, transcript
  enhancement + run-asset summary/entities route to the keyless bundled endpoint;
  agentic research + the LLM name pass stay on the BYO endpoint.

Revision ID: 0026
Revises: 0025
Create Date: 2026-08-18 16:00:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0026"
down_revision: str | None = "0025"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "app_settings",
        sa.Column("llm_bundled_enabled", sa.Boolean(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("app_settings", "llm_bundled_enabled")
