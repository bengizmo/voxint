"""Run notifications / webhooks (issue #12).

Two halves, deliberately decoupled:

- :func:`record_transition` — the **emission** side. Persistence only: it
  inserts one ``notification_deliveries`` outbox row in the caller's existing
  transaction (invoked from ``cas_update_run`` after a notifiable transition
  commits atomically with the state change). No HTTP, no Celery, no clock beyond
  the occurrence timestamp — so importing it into the low-level transition
  primitive stays cheap and cycle-free.
- The **delivery** side (a beat sweep that claims rows under a lease and POSTs a
  signed payload) lives in the worker; it never runs inside a request/transition.

See ``docs/plans/2026-08-16-1240_run-notifications-webhooks.md``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy.dialects.postgresql import insert as pg_insert

from voxint.db.models import NotifiableEvent, NotificationDelivery, RunStatus

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from voxint.config import Settings

# Bumped only on a breaking change to the payload shape; receivers pin on it.
PAYLOAD_SCHEMA_VERSION = 1


def _notifiable(status: RunStatus) -> NotifiableEvent | None:
    """The subset of terminal-ish statuses an operator is notified about."""
    try:
        return NotifiableEvent(status.value)
    except ValueError:
        return None  # RUNNING / QUEUED / CANCELLED — not news


def build_payload(
    *, run_id: uuid.UUID, event: NotifiableEvent, transition_revision: int, delivery_id: uuid.UUID
) -> dict[str, Any]:
    """The immutable, versioned webhook body for one arrival.

    Minimal by design: enough for a receiver to identify the run, the event, and
    the exact arrival (``transition_revision``), and to deduplicate retries
    (``delivery_id``). The run's ``error`` text is deliberately omitted — it can
    carry sensitive detail and is not needed to act on a failure.
    """
    return {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "event": event.value,
        "run_id": str(run_id),
        "transition_revision": transition_revision,
        "occurred_at": datetime.now(tz=UTC).isoformat(),
        "delivery_id": str(delivery_id),
    }


def record_transition(
    session: Session,
    *,
    run_id: uuid.UUID,
    status: RunStatus,
    transition_revision: int,
    settings: Settings,
) -> None:
    """Insert an outbox row for a notifiable transition, in the caller's tx.

    No-ops unless webhooks are enabled and ``status`` is notifiable. Keyed by
    ``(run_id, transition_revision)`` with ``ON CONFLICT DO NOTHING`` so a
    concurrent or retried emission of the same arrival never doubles a row. A
    FAILED arrival is held for ``notify_failed_initial_delay_seconds`` before its
    first delivery attempt, giving a synchronous requeue (recovery / stage retry)
    time to advance the run so the delivery sweep can suppress it rather than
    send a misleading "failed".

    Runs in the SAME transaction as the ``cas_update_run`` UPDATE: if the caller
    rolls back, the outbox row rolls back with it (never emit a transition that
    did not commit).
    """
    if not settings.notify_enabled:
        return
    event = _notifiable(status)
    if event is None:
        return

    delivery_id = uuid.uuid4()
    now = datetime.now(tz=UTC)
    next_attempt_at = now
    if event is NotifiableEvent.FAILED and settings.notify_failed_initial_delay_seconds > 0:
        next_attempt_at = now + timedelta(seconds=settings.notify_failed_initial_delay_seconds)

    payload = build_payload(
        run_id=run_id,
        event=event,
        transition_revision=transition_revision,
        delivery_id=delivery_id,
    )
    stmt = (
        pg_insert(NotificationDelivery)
        .values(
            id=delivery_id,
            pipeline_run_id=run_id,
            transition_revision=transition_revision,
            event=event.value,
            payload=payload,
            next_attempt_at=next_attempt_at,
        )
        .on_conflict_do_nothing(constraint="uq_notification_deliveries_run_revision")
    )
    session.execute(stmt)
