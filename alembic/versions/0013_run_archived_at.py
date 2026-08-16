"""pipeline_runs.archived_at — soft-archive runs (issue #5, slice 2)

Adds a nullable ``archived_at`` timestamp to ``pipeline_runs``. A non-NULL value
soft-archives the run: it is hidden from ``/runs`` and the ``/review`` queue but
every row (including the append-only adjudication ledger) stays intact, and the
action is reversible (un-archive sets it back to NULL). Deliberately a timestamp
column, not a new ``RunStatus`` — archive is operator-visibility metadata that
stays orthogonal to pipeline status (mirrors ``operator_notes``).

Revision ID: 0013
Revises: 0012
Create Date: 2026-08-16 10:30:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "pipeline_runs",
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("pipeline_runs", "archived_at")
