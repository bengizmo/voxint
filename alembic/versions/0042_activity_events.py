"""activity_events — console activity outbox (issue #162, Console 2.0 P7)

The append-only outbox behind the console activity indicator (completion toasts
and a Jobs badge). One row per operator-facing event is inserted in the SAME
transaction as the change it announces (a run reaching COMPLETED, written from
``cas_update_run``), so an event is emitted iff the change committed. The browser
polls the table directly, keyed on the monotonic ``BIGINT`` identity ``id``; a
denormalized ``title``/``href`` snapshot is frozen at emission so the poll never
re-scans the source tables. ``occurrence_key`` (``run:{id}:completed``) makes
emission idempotent under a retried transition. Retention is a bounded newest-N
prune (``voxint.activity_prune``).

Only the ``run_completed`` kind exists here; the deferred speaker-event
follow-up widens the CHECK and adds typed speaker provenance in its own
migration rather than shipping an unconstrained schema now.

Revision ID: 0042
Revises: 0041
Create Date: 2026-08-26 10:20:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0042"
down_revision: str | None = "0041"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_KINDS = "'run_completed'"


def upgrade() -> None:
    op.create_table(
        "activity_events",
        # BigInteger + no explicit sequence => BIGSERIAL/identity in Postgres:
        # a gapless-enough monotonic id that is the browser's poll cursor.
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column(
            "pipeline_run_id",
            sa.Uuid(),
            sa.ForeignKey("pipeline_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("href", sa.Text(), nullable=False),
        sa.Column("occurrence_key", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(f"kind IN ({_KINDS})", name="activity_events_kind_check"),
        sa.CheckConstraint("char_length(title) <= 500", name="activity_events_title_len_check"),
        sa.CheckConstraint("char_length(href) <= 500", name="activity_events_href_len_check"),
        sa.CheckConstraint(
            "char_length(occurrence_key) <= 200",
            name="activity_events_occurrence_key_len_check",
        ),
        sa.UniqueConstraint("occurrence_key", name="uq_activity_events_occurrence_key"),
    )
    op.create_index(
        "ix_activity_events_pipeline_run_id",
        "activity_events",
        ["pipeline_run_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_activity_events_pipeline_run_id", table_name="activity_events")
    op.drop_table("activity_events")
