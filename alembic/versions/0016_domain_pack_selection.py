"""per-run/per-folder domain pack selection (issue #11)

Adds two columns for domain-pack depth:

* ``pipeline_runs.domain_pack`` (JSONB, nullable) — the frozen manifest snapshot
  the run was submitted with. Stamped write-once at submit; the pipeline worker
  and enrichment producers read it instead of the mutable global env, so a run
  and its late enrichment always see the exact pack it was transcribed with.
  NULL = a legacy run created before this feature (resolve the current default at
  execution time).
* ``app_settings.folder_domain_packs`` (JSONB, NOT NULL, default ``{}``) — the
  ``{media_folder -> pack_name}`` mapping the submit path consults to pick each
  run's pack. Empty ⇒ every folder uses the default pack (pre-#11 behavior).

No backfill: existing runs keep ``domain_pack = NULL`` and fall back to the
configured default at execution time, exactly reproducing their prior behavior.

Revision ID: 0016
Revises: 0015
Create Date: 2026-08-16 13:30:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0016"
down_revision: str | None = "0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column(
        "pipeline_runs",
        sa.Column("domain_pack", _JSON, nullable=True),
    )
    op.add_column(
        "app_settings",
        sa.Column(
            "folder_domain_packs",
            _JSON,
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
    )


def downgrade() -> None:
    op.drop_column("app_settings", "folder_domain_packs")
    op.drop_column("pipeline_runs", "domain_pack")
