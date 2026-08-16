"""notification_deliveries outbox — run webhooks (issue #12)

Adds the transactional outbox that backs completion/failure/awaiting webhooks.
One row per notifiable transition arrival is inserted in the same transaction as
the run's state change (atomic at-least-once); a beat sweep later claims due rows
under a lease and POSTs a signed payload. Keyed by (pipeline_run_id,
transition_revision) so a requeued run that fails again is a distinct arrival.

Revision ID: 0015
Revises: 0014
Create Date: 2026-08-16 12:45:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0015"
down_revision: str | None = "0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_EVENTS = "'awaiting_adjudication', 'completed', 'failed'"
_STATUSES = "'pending', 'in_flight', 'delivered', 'dead', 'suppressed'"


def upgrade() -> None:
    op.create_table(
        "notification_deliveries",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "pipeline_run_id",
            sa.Uuid(),
            sa.ForeignKey("pipeline_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("transition_revision", sa.Integer(), nullable=False),
        sa.Column("event", sa.Text(), nullable=False),
        sa.Column(
            "payload",
            sa.JSON().with_variant(sa.dialects.postgresql.JSONB(), "postgresql"),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Text(),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "next_attempt_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            f"event IN ({_EVENTS})", name="notification_deliveries_event_check"
        ),
        sa.CheckConstraint(
            f"status IN ({_STATUSES})", name="notification_deliveries_status_check"
        ),
        sa.CheckConstraint("attempts >= 0", name="notification_deliveries_attempts_nonneg_check"),
        sa.CheckConstraint(
            "(status = 'delivered') = (delivered_at IS NOT NULL)",
            name="notification_deliveries_delivered_shape_check",
        ),
        sa.UniqueConstraint(
            "pipeline_run_id",
            "transition_revision",
            name="uq_notification_deliveries_run_revision",
        ),
    )
    op.create_index(
        "ix_notification_deliveries_pipeline_run_id",
        "notification_deliveries",
        ["pipeline_run_id"],
    )
    op.create_index(
        "ix_notification_deliveries_due",
        "notification_deliveries",
        ["next_attempt_at"],
        postgresql_where=sa.text("status IN ('pending', 'in_flight')"),
    )


def downgrade() -> None:
    op.drop_index("ix_notification_deliveries_due", table_name="notification_deliveries")
    op.drop_index(
        "ix_notification_deliveries_pipeline_run_id",
        table_name="notification_deliveries",
    )
    op.drop_table("notification_deliveries")
