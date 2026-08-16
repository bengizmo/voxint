"""app_settings.llm_api_key — in-UI LLM API key storage (issue #10, slice 1)

Adds a nullable ``llm_api_key`` TEXT column to the singleton ``app_settings``
row so a non-technical operator can set/replace/remove the optional LLM API key
from the setup wizard and Settings page instead of hand-editing ``.env`` and
restarting the worker. Precedence mirrors ``llm_enabled``: a non-blank row value
WINS, and the env ``LLM_API_KEY`` is the seed/fallback (NULL/blank ⇒ env).

The key is stored **plaintext at rest**. That is an accepted trade-off for
Voxint's single-operator, local-first deployment (the DB is not a shared,
multi-tenant store); a SQL dump/backup necessarily contains it. It is never
rendered back to the UI, logged, or exported — see the ``AppSettings`` model
docstring and ``app_settings.resolve_effective_llm_api_key``.

Revision ID: 0016
Revises: 0015
Create Date: 2026-08-16 12:05:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0016"
down_revision: str | None = "0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "app_settings",
        sa.Column("llm_api_key", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("app_settings", "llm_api_key")
