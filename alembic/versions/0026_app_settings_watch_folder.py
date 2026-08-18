"""app_settings: watch-folder ingest runtime override + last-sweep status

Issue #60 (watch-folder ingest, under the #47 console-UX arc). Adds two nullable
columns to the singleton ``app_settings`` row:

- ``watch_folder_enabled`` (Boolean, nullable) — the tri-state runtime override
  for the ingest gate, DB-row-wins-over-env (the #74 feature-flag precedent):
  NULL = "inherit the env default" (``config.watch_folder_enabled``, off),
  non-NULL = row overrides. Lets the Settings folders panel enable/disable the
  watcher with no restart, and clearing it reverts to the installation setting.
- ``watch_folder_last_sweep`` (JSONB, nullable) — the ONLY watch-sweep state
  persisted: the latest sweep summary (counts + caps + completed_at) for the
  plain-language Settings status line. NULL = the sweep has never run.

Purely additive and NULL by default, so the NULL-parity invariant holds (an
existing row is unaffected — env still governs, no sweep summary yet) and there
is no data backfill or behavior change on upgrade.

Revision ID: 0026
Revises: 0025
Create Date: 2026-08-18 15:00:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0026"
down_revision: str | None = "0025"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column(
        "app_settings",
        sa.Column("watch_folder_enabled", sa.Boolean(), nullable=True),
    )
    op.add_column(
        "app_settings",
        sa.Column("watch_folder_last_sweep", _JSON, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("app_settings", "watch_folder_last_sweep")
    op.drop_column("app_settings", "watch_folder_enabled")
