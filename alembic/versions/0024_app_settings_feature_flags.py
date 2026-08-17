"""app_settings: in-UI feature-flag + external-sources columns

Issue #74 (keystone of the #47 settings-overhaul arc). Adds one nullable column
per in-UI-editable feature flag to the singleton ``app_settings`` row so the
settings console can override the env default at runtime, DB-row-wins-over-env
(the #10/#11/#16 precedent). Purely additive:

- Every column is NULLable with NO server default, so tri-state semantics hold:
  NULL = "inherit the env default", non-NULL = row overrides. An existing row is
  unaffected (all columns read NULL ⇒ env still governs — the NULL-parity
  invariant), so there is no data backfill and no behavior change on upgrade.
- ``web_search_api_key`` is a credential (plaintext at rest, like
  ``llm_api_key``); the others are non-secret.

This is the ONLY migration in the whole arc — every downstream child (#62/#5/#61/
#63) is pure app+template code touching zero schema.

Revision ID: 0024
Revises: 0023
Create Date: 2026-08-17 15:20:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0024"
down_revision: str | None = "0023"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Kept in lockstep with voxint.db.models.AppSettings (issue #74).
_BOOL_FLAGS = (
    "enrichment_names_enabled",
    "enrichment_names_llm_enabled",
    "enrichment_run_assets_enabled",
    "enrichment_run_assets_autogenerate",
    "voxint_web_research",
    "enrichment_web_research_enabled",
    "ytdlp_enabled",
)
_TEXT_FLAGS = (
    "source_authority_domains",
    "web_search_base_url",
    "web_search_api_key",
)


def upgrade() -> None:
    for name in _BOOL_FLAGS:
        op.add_column("app_settings", sa.Column(name, sa.Boolean(), nullable=True))
    for name in _TEXT_FLAGS:
        op.add_column("app_settings", sa.Column(name, sa.Text(), nullable=True))


def downgrade() -> None:
    for name in reversed(_TEXT_FLAGS):
        op.drop_column("app_settings", name)
    for name in reversed(_BOOL_FLAGS):
        op.drop_column("app_settings", name)
