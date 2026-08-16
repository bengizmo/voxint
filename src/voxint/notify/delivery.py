"""Webhook delivery sweep (issue #12, phase C).

The **delivery** half of the transactional outbox: a beat-driven sweep that
claims due ``notification_deliveries`` rows and POSTs each as a signed webhook,
recording the outcome. It is the mirror of the emission side
(:mod:`voxint.notify`), and — like :func:`voxint.media.reclaim` — is pure and
Celery-free so it is unit-testable against real Postgres without a broker.

The load-bearing correctness rule is **never hold a DB transaction across
network I/O**. Each sweep is three separable moves:

1. **Claim** (one short tx): ``SELECT ... FOR UPDATE SKIP LOCKED`` the oldest due
   rows (``pending`` past ``next_attempt_at``, or ``in_flight`` whose lease has
   lapsed — crash reclaim), flip them to ``in_flight`` under a fresh lease, and
   commit. Overlapping sweeps never double-claim; a sweep that dies mid-delivery
   loses its lease and the next sweep redelivers (at-least-once; the receiver
   dedups on ``delivery_id``).
2. **Deliver** (OUTSIDE any tx): for a ``failed`` arrival, re-read the run and
   *suppress* if it has advanced past the row's ``transition_revision`` (a
   requeue settled) rather than send a misleading "failed". Otherwise serialize
   the frozen payload to deterministic bytes, sign them, and POST via the
   address-pinned transport (revalidated public every attempt — DNS rebinding),
   ``trust_env=False``, no redirects, bounded timeout.
3. **Record** (one short tx per row): 2xx → ``delivered``; else bump ``attempts``
   and either reschedule with capped exponential backoff + jitter or, at the
   attempt ceiling, mark ``dead``. ``last_error`` is redacted and length-capped —
   never the URL, secret, or payload.

See ``docs/plans/2026-08-16-1240_run-notifications-webhooks.md``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import random
import socket
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

import httpx
from sqlalchemy import CursorResult, and_, delete, or_, select

from voxint.db.models import (
    NotifiableEvent,
    NotificationDelivery,
    NotificationStatus,
    PipelineRun,
)
from voxint.media.netcheck import (
    HostNotPublicError,
    Resolver,
    UrlPolicyError,
    parse_http_url,
    resolve_public_addresses,
)
from voxint.media.redaction import cap_length, redact

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker

    from voxint.config import Settings

logger = logging.getLogger(__name__)

_USER_AGENT = "voxint-webhook/1"

# A fresh client per network attempt (pool isolation — a shared keepalive pool
# could reuse a connection pinned for one hostname to serve another that pins to
# the same IP). Injectable so tests supply httpx.MockTransport without sockets.
ClientFactory = Callable[[httpx.Timeout], httpx.Client]


def _default_client_factory(timeout: httpx.Timeout) -> httpx.Client:
    # trust_env=False: an ambient HTTP(S)_PROXY must never silently reroute an
    # outbound webhook. Redirects are refused (follow_redirects=False) — a 3xx is
    # a non-2xx answer that retries, never a hop we chase to a fresh host.
    return httpx.Client(follow_redirects=False, trust_env=False, timeout=timeout)


@dataclass(frozen=True)
class DeliverySummary:
    """Per-sweep tally, returned and structured-logged."""

    claimed: int = 0
    delivered: int = 0
    suppressed: int = 0
    retried: int = 0
    dead: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "claimed": self.claimed,
            "delivered": self.delivered,
            "suppressed": self.suppressed,
            "retried": self.retried,
            "dead": self.dead,
        }


@dataclass(frozen=True)
class _Claimed:
    """A row's fields captured at claim time, so delivery touches no live ORM
    object (the claim session is closed before any network I/O)."""

    id: uuid.UUID
    run_id: uuid.UUID
    event: str
    transition_revision: int
    payload: dict[str, Any]
    attempts: int


@dataclass(frozen=True)
class _PostResult:
    ok: bool
    detail: str  # already redacted + capped; names the host at most


def serialize_payload(payload: dict[str, Any]) -> bytes:
    """The exact bytes signed and transmitted for a row.

    Deterministic (sorted keys, no whitespace) so the signature covers precisely
    what crosses the wire — a JSONB round-trip is not byte-stable, so the frozen
    ``payload`` is re-serialized here rather than trusted as stored text."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign(secret: str, timestamp: str, body: bytes) -> str:
    """HMAC-SHA256 over ``timestamp + "." + body`` — the receiver recomputes this
    and compares in constant time (the timestamp binds the signature to a moment,
    so a captured body cannot be replayed outside the receiver's skew window)."""
    mac = hmac.new(secret.encode("utf-8"), f"{timestamp}.".encode() + body, hashlib.sha256)
    return mac.hexdigest()


def deliver_due(
    factory: sessionmaker[Session],
    settings: Settings,
    *,
    now: datetime | None = None,
    client_factory: ClientFactory | None = None,
    resolver: Resolver = socket.getaddrinfo,
) -> DeliverySummary:
    """Claim and deliver one batch of due webhook rows.

    Returns a per-sweep tally. Claiming is a single short transaction; each
    claimed row is then delivered and recorded independently outside any tx, so a
    slow or failing receiver never blocks the database or the other rows."""
    now = now or datetime.now(tz=UTC)
    make_client = client_factory or _default_client_factory
    claimed = _claim_due(factory, settings, now)
    if not claimed:
        return DeliverySummary()

    delivered = suppressed = retried = dead = 0
    for row in claimed:
        outcome = _deliver_one(
            factory, settings, row, now=now, make_client=make_client, resolver=resolver
        )
        if outcome == "delivered":
            delivered += 1
        elif outcome == "suppressed":
            suppressed += 1
        elif outcome == "dead":
            dead += 1
        else:
            retried += 1
    return DeliverySummary(
        claimed=len(claimed), delivered=delivered, suppressed=suppressed, retried=retried, dead=dead
    )


def purge_expired_deliveries(
    factory: sessionmaker[Session], settings: Settings, *, now: datetime | None = None
) -> int:
    """Delete settled delivery rows older than the retention horizon; return the
    count removed.

    Reaps only ``delivered`` and ``suppressed`` rows (fully resolved and no longer
    actionable) past ``notify_retention_seconds`` — ``dead`` rows are kept until an
    operator acts on them, and ``pending``/``in_flight`` are still live. Bounded by
    ``notify_batch_limit`` per call (oldest first) so a large backlog drains over
    several sweeps rather than in one unbounded DELETE. Age is measured from
    ``created_at`` (both terminal statuses have one; ``suppressed`` never gets a
    ``delivered_at``)."""
    now = now or datetime.now(tz=UTC)
    cutoff = now - timedelta(seconds=settings.notify_retention_seconds)
    victims = (
        select(NotificationDelivery.id)
        .where(
            NotificationDelivery.status.in_(
                (NotificationStatus.DELIVERED.value, NotificationStatus.SUPPRESSED.value)
            ),
            NotificationDelivery.created_at < cutoff,
        )
        .order_by(NotificationDelivery.created_at.asc())
        .limit(settings.notify_batch_limit)
    )
    with factory() as session:
        result = cast(
            "CursorResult[Any]",
            session.execute(
                delete(NotificationDelivery).where(
                    NotificationDelivery.id.in_(victims.scalar_subquery())
                )
            ),
        )
        session.commit()
        return int(result.rowcount or 0)


def _claim_due(
    factory: sessionmaker[Session], settings: Settings, now: datetime
) -> list[_Claimed]:
    """Lock and lease the oldest due rows in one short tx; return plain snapshots."""
    lease_until = now + timedelta(seconds=settings.notify_lease_seconds)
    stmt = (
        select(NotificationDelivery)
        .where(
            or_(
                and_(
                    NotificationDelivery.status == NotificationStatus.PENDING.value,
                    NotificationDelivery.next_attempt_at <= now,
                ),
                and_(
                    NotificationDelivery.status == NotificationStatus.IN_FLIGHT.value,
                    NotificationDelivery.lease_expires_at < now,
                ),
            )
        )
        .order_by(NotificationDelivery.next_attempt_at.asc())
        .limit(settings.notify_batch_limit)
        .with_for_update(skip_locked=True)
    )
    claimed: list[_Claimed] = []
    with factory() as session:
        rows = session.execute(stmt).scalars().all()
        for row in rows:
            row.status = NotificationStatus.IN_FLIGHT.value
            row.lease_expires_at = lease_until
            claimed.append(
                _Claimed(
                    id=row.id,
                    run_id=row.pipeline_run_id,
                    event=row.event,
                    transition_revision=row.transition_revision,
                    payload=dict(row.payload),
                    attempts=row.attempts,
                )
            )
        session.commit()
    return claimed


def _deliver_one(
    factory: sessionmaker[Session],
    settings: Settings,
    row: _Claimed,
    *,
    now: datetime,
    make_client: ClientFactory,
    resolver: Resolver,
) -> str:
    """Deliver one claimed row outside any tx; record its outcome. Returns one of
    ``delivered`` / ``suppressed`` / ``retried`` / ``dead``."""
    if row.event == NotifiableEvent.FAILED.value and _run_advanced_past(factory, row):
        _record_suppressed(factory, row.id)
        logger.info(
            "notify_delivery id=%s run=%s event=failed outcome=suppressed", row.id, row.run_id
        )
        return "suppressed"

    result = _post(settings, row, now=now, make_client=make_client, resolver=resolver)
    if result.ok:
        _record_delivered(factory, row.id, now=now, attempts=row.attempts + 1)
        logger.info(
            "notify_delivery id=%s run=%s event=%s outcome=delivered",
            row.id, row.run_id, row.event,
        )
        return "delivered"

    attempts = row.attempts + 1
    if attempts >= settings.notify_max_attempts:
        _record_dead(factory, row.id, attempts=attempts, last_error=result.detail)
        logger.warning(
            "notify_delivery id=%s run=%s event=%s outcome=dead attempts=%d",
            row.id, row.run_id, row.event, attempts,
        )
        return "dead"

    next_at = now + timedelta(seconds=_backoff_with_jitter(attempts, settings))
    _record_retry(
        factory, row.id, attempts=attempts, next_attempt_at=next_at, last_error=result.detail
    )
    logger.info(
        "notify_delivery id=%s run=%s event=%s outcome=retry attempts=%d",
        row.id, row.run_id, row.event, attempts,
    )
    return "retried"


def _backoff_with_jitter(attempts: int, settings: Settings) -> float:
    """Capped exponential in completed attempts, plus up to 10% jitter — the
    stage-retry idiom (``worker.tasks.backoff_seconds``) on the notify knobs."""
    base = settings.notify_backoff_base_seconds
    delay = float(min(base * 2 ** max(attempts - 1, 0), settings.notify_backoff_max_seconds))
    return delay + random.uniform(0, delay * 0.1)


def _run_advanced_past(factory: sessionmaker[Session], row: _Claimed) -> bool:
    """True if the run no longer sits at the FAILED arrival's revision — either it
    was requeued (revision advanced) or it is gone entirely — so this "failed"
    would be stale news and must be suppressed rather than sent."""
    with factory() as session:
        run = session.get(PipelineRun, row.run_id)
        if run is None:
            return True
        return run.revision > row.transition_revision


def _post(
    settings: Settings,
    row: _Claimed,
    *,
    now: datetime,
    make_client: ClientFactory,
    resolver: Resolver,
) -> _PostResult:
    """POST the signed payload to the configured endpoint over the pinned-address
    transport. Any non-2xx answer, refused address, or transport failure is a
    (redacted) failure the caller retries; only a 2xx is success."""
    url = settings.notify_webhook_url
    secret = settings.notify_webhook_secret
    body = serialize_payload(row.payload)
    timestamp = str(int(now.timestamp()))
    signature = sign(secret, timestamp, body)
    headers = {
        "Content-Type": "application/json",
        "User-Agent": _USER_AGENT,
        "X-Voxint-Delivery": str(row.id),
        "X-Voxint-Timestamp": timestamp,
        "X-Voxint-Signature": f"sha256={signature}",
    }

    def refusal(detail: str) -> _PostResult:
        # extra_secrets scrubs the configured URL and secret verbatim even where
        # they appear as prose no structural rule would catch.
        return _PostResult(ok=False, detail=cap_length(redact(detail, extra_secrets=(url, secret))))

    try:
        gate = parse_http_url(url)
    except UrlPolicyError as exc:
        return refusal(f"webhook url refused: {exc}")

    origin = httpx.URL(gate.url)
    host = gate.host
    authority = origin.netloc.decode("ascii")
    try:
        if gate.ip is not None:
            vetted = [str(gate.ip)]
        else:
            vetted = [str(ip) for ip in resolve_public_addresses(host, resolver=resolver)]
    except HostNotPublicError as exc:
        return refusal(f"host {host!r} refused: {exc}")

    per_attempt = settings.notify_timeout_seconds
    timeout = httpx.Timeout(
        min(10.0, per_attempt), read=per_attempt, write=per_attempt, pool=per_attempt
    )
    request_headers = {**headers, "Host": authority}
    extensions: dict[str, str] = {"sni_hostname": host} if gate.scheme == "https" else {}

    last_transport = "no vetted address attempted"
    for address in vetted:
        pinned = origin.copy_with(host=address)
        try:
            with make_client(timeout) as client:
                request = client.build_request(
                    "POST", pinned, content=body, headers=request_headers, extensions=extensions
                )
                response = client.send(request)
                response.close()
        except httpx.HTTPError as exc:
            # Connect/protocol/timeout against THIS vetted address — try the next
            # one (never re-resolving). Exception text can echo the pinned URL.
            last_transport = f"{type(exc).__name__}: {exc}"
            continue
        status = response.status_code
        if 200 <= status < 300:
            return _PostResult(ok=True, detail="")
        return refusal(f"host {host!r} answered HTTP {status}")
    return refusal(f"host {host!r}: {last_transport}")


def _record_delivered(
    factory: sessionmaker[Session], row_id: uuid.UUID, *, now: datetime, attempts: int
) -> None:
    with factory() as session:
        row = session.get(NotificationDelivery, row_id)
        if row is None:
            return
        row.status = NotificationStatus.DELIVERED.value
        row.delivered_at = now
        row.attempts = attempts
        row.lease_expires_at = None
        row.last_error = None
        session.commit()


def _record_suppressed(factory: sessionmaker[Session], row_id: uuid.UUID) -> None:
    with factory() as session:
        row = session.get(NotificationDelivery, row_id)
        if row is None:
            return
        row.status = NotificationStatus.SUPPRESSED.value
        row.lease_expires_at = None
        session.commit()


def _record_dead(
    factory: sessionmaker[Session], row_id: uuid.UUID, *, attempts: int, last_error: str
) -> None:
    with factory() as session:
        row = session.get(NotificationDelivery, row_id)
        if row is None:
            return
        row.status = NotificationStatus.DEAD.value
        row.attempts = attempts
        row.lease_expires_at = None
        row.last_error = last_error
        session.commit()


def _record_retry(
    factory: sessionmaker[Session],
    row_id: uuid.UUID,
    *,
    attempts: int,
    next_attempt_at: datetime,
    last_error: str,
) -> None:
    with factory() as session:
        row = session.get(NotificationDelivery, row_id)
        if row is None:
            return
        row.status = NotificationStatus.PENDING.value
        row.attempts = attempts
        row.next_attempt_at = next_attempt_at
        row.lease_expires_at = None
        row.last_error = last_error
        session.commit()
