"""app_settings: semantic-index feature flags (#121)

Two nullable tri-state columns backing the transcript semantic-search spine, in
the same NULL-inherits-env / non-NULL-overrides shape as the other feature flags
(migration 0024). NULL means "inherit the env default" (config.Settings); a
non-NULL value is a UI override. Both depend only on each other
(``semantic_index_autogenerate`` ⇒ ``semantic_index_enabled``), never on
``llm_enabled`` — the embedder is a local ONNX graph with no LLM and no egress.

Revision ID: 0034
Revises: 0033
Create Date: 2026-08-22 08:30:00.000000
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
        "app_settings",
        sa.Column("semantic_index_enabled", sa.Boolean(), nullable=True),
    )
    op.add_column(
        "app_settings",
        sa.Column("semantic_index_autogenerate", sa.Boolean(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("app_settings", "semantic_index_autogenerate")
    op.drop_column("app_settings", "semantic_index_enabled")
