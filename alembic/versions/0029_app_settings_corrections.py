"""app_settings: operator-authored correction rules

Issue #84 (console corrections-authoring surface, final child of the #78
non-LLM-correction epic). Adds one column to ``app_settings`` so a non-technical
operator can author correction rules from the console instead of hand-editing a
domain-pack ``manifest.yaml``:

- ``corrections`` (JSONB, NOT NULL, server_default ``'[]'``) — a list of rule
  mappings ``{id, match, replace, case_sensitive, whole_word}``, each validated
  at author time through the same #80 gate a pack gets. At submit-time freeze
  these are unioned onto the selected pack's own corrections and stored in
  ``pipeline_runs.domain_pack`` (see ``ingest.service._run_domain_pack_snapshot``)
  — so the #82 corrector and #83 provenance read them off the frozen per-run
  snapshot with no changes. The server default backfills every existing
  single-row ``app_settings`` to ``[]`` on upgrade, so the NOT-NULL constraint
  holds without a data-migration pass.

Mirrors the ``folder_domain_packs`` JSONB column added in #11.

Revision ID: 0029
Revises: 0028
Create Date: 2026-08-19 00:40:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0029"
down_revision: str | None = "0028"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "app_settings",
        sa.Column(
            "corrections",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("app_settings", "corrections")
