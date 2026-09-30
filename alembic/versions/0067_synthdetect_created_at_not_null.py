"""Make synthdetect_jobs.created_at NOT NULL (#692).

Migration 0050 created the column nullable with a ``now()`` server default,
while the ORM model (and every other ``created_at`` in the schema) declares it
NOT NULL. This brings the schema in line with the model.

The only application writer relies on the server default, so a NULL can only
come from an explicit NULL insert or a manual edit. Such rows are repaired
rather than failing the upgrade: ``created_at`` becomes ``started_at`` when the
job started (which keeps ``started_at >= created_at`` true), else ``now()``.
A NULL ``started_at`` implies a NULL ``finished_at`` (CHECK
``synthdetect_jobs_finished_requires_started_check``), so no other timestamp
constrains the fallback. The repaired row count is logged as a warning.

Downgrade drops NOT NULL only; backfilled values stay.

Revision ID: 0067
Revises: 0066
Create Date: 2026-09-30
"""

import logging
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0067"
down_revision: str | None = "0066"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

log = logging.getLogger("alembic.runtime.migration")


def upgrade() -> None:
    repaired = (
        op.get_bind()
        .execute(
            sa.text(
                "UPDATE synthdetect_jobs SET created_at = COALESCE(started_at, now())"
                " WHERE created_at IS NULL"
            )
        )
        .rowcount
    )
    if repaired:
        log.warning(
            "0067: backfilled created_at on %d synthdetect_jobs row(s) that had none",
            repaired,
        )
    op.alter_column(
        "synthdetect_jobs",
        "created_at",
        existing_type=sa.DateTime(timezone=True),
        existing_server_default=sa.text("now()"),
        nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "synthdetect_jobs",
        "created_at",
        existing_type=sa.DateTime(timezone=True),
        existing_server_default=sa.text("now()"),
        nullable=True,
    )
