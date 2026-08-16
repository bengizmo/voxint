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
from sqlalchemy import CursorResult, and_, delete, or_, select, update

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

# Cap on vetted addresses tried per POST. resolve_public_addresses can return
# several A/AAAA records; without a bound, a host with many blackholed addresses
# would let one attempt's wall-time (each up to notify_timeout_seconds + connect)
# run arbitrarily past notify_lease_seconds. Bounding it keeps a claim's lease a
# meaningful ownership window (the lease-guarded record writes make an over-run
# safe regardless, but this keeps duplicate deliveries rare rather than routine).
_MAX_POST_ADDRESSES = 3

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
    # The lease this sweep wrote at claim time. Every outcome write CASes on it
    # (status still in_flight AND lease unchanged), so a sweep whose lease lapsed
    # and was reclaimed by another sweep can never clobber the new owner's result.
    lease_until: datetime


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
    clock: Callable[[], datetime] | None = None,
    client_factory: ClientFactory | None = None,
    resolver: Resolver = socket.getaddrinfo,
) -> DeliverySummary:
    """Claim and deliver one batch of due webhook rows.

    Returns a per-sweep tally. Claiming is a single short transaction; each
    claimed row is then delivered and recorded independently outside any tx, so a
    slow or failing receiver never blocks the database or the other rows.

    ``now`` fixes the claim instant (due-predicate + lease). ``clock`` supplies a
    FRESH timestamp per row at delivery time — the signature timestamp and the
    retry/delivered anchors must reflect when a row is actually sent, not
    batch-start, or a tail row of a slow batch would be signed with a stale
    timestamp a receiver's replay window rejects. Both default to wall-clock;
    tests inject them for determinism."""
    tick = clock or (lambda: datetime.now(tz=UTC))
    now = now or tick()
    make_client = client_factory or _default_client_factory
    claimed = _claim_due(factory, settings, now)
    if not claimed:
        return DeliverySummary()

    delivered = suppressed = retried = dead = 0
    for row in claimed:
        try:
            outcome = _deliver_one(
                factory, settings, row, clock=tick, make_client=make_client, resolver=resolver
            )
        except Exception:
            # A per-row bug (never a receiver's HTTP behavior — _post catches
            # httpx errors) must not abandon the rest of the batch. The row stays
            # in_flight and its lease reclaim redelivers it. Log without echoing
            # any URL/secret (the traceback frames carry neither).
            logger.exception(
                "notify_delivery id=%s run=%s crashed; left for reclaim", row.id, row.run_id
            )
            retried += 1
            continue
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
                    lease_until=lease_until,
                )
            )
        session.commit()
    return claimed


def _deliver_one(
    factory: sessionmaker[Session],
    settings: Settings,
    row: _Claimed,
    *,
    clock: Callable[[], datetime],
    make_client: ClientFactory,
    resolver: Resolver,
) -> str:
    """Deliver one claimed row outside any tx; record its outcome. Returns one of
    ``delivered`` / ``suppressed`` / ``retried`` / ``dead``.

    Every outcome write is lease-guarded (:func:`_finish`): if this sweep's lease
    lapsed and another sweep reclaimed the row, the write no-ops rather than
    clobbering the new owner's result — logged as ``ownership_lost``."""
    attempt_now = clock()
    if row.event == NotifiableEvent.FAILED.value and _run_advanced_past(factory, row):
        _finish(factory, row, status=NotificationStatus.SUPPRESSED.value, lease_expires_at=None)
        logger.info(
            "notify_delivery id=%s run=%s event=failed outcome=suppressed", row.id, row.run_id
        )
        return "suppressed"

    result = _post(settings, row, now=attempt_now, make_client=make_client, resolver=resolver)
    if result.ok:
        owned = _finish(
            factory,
            row,
            status=NotificationStatus.DELIVERED.value,
            delivered_at=attempt_now,
            attempts=row.attempts + 1,
            lease_expires_at=None,
            last_error=None,
        )
        logger.info(
            "notify_delivery id=%s run=%s event=%s outcome=delivered%s",
            row.id, row.run_id, row.event, "" if owned else " (ownership_lost)",
        )
        return "delivered"

    attempts = row.attempts + 1
    if attempts >= settings.notify_max_attempts:
        _finish(
            factory,
            row,
            status=NotificationStatus.DEAD.value,
            attempts=attempts,
            lease_expires_at=None,
            last_error=result.detail,
        )
        logger.warning(
            "notify_delivery id=%s run=%s event=%s outcome=dead attempts=%d",
            row.id, row.run_id, row.event, attempts,
        )
        return "dead"

    next_at = attempt_now + timedelta(seconds=_backoff_with_jitter(attempts, settings))
    _finish(
        factory,
        row,
        status=NotificationStatus.PENDING.value,
        attempts=attempts,
        next_attempt_at=next_at,
        lease_expires_at=None,
        last_error=result.detail,
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
    # connect is set explicitly (not left to the positional default) so a future
    # all-keyword refactor cannot silently uncap it. Capped at 10s like read/write.
    timeout = httpx.Timeout(
        connect=min(10.0, per_attempt), read=per_attempt, write=per_attempt, pool=per_attempt
    )
    request_headers = {**headers, "Host": authority}
    extensions: dict[str, str] = {"sni_hostname": host} if gate.scheme == "https" else {}

    last_transport = "no vetted address attempted"
    for address in vetted[:_MAX_POST_ADDRESSES]:
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


def _finish(factory: sessionmaker[Session], claimed: _Claimed, **values: Any) -> bool:
    """Write a claimed row's terminal/retry outcome in one short tx, guarded by
    the lease this sweep holds. Returns True if the write landed (this sweep still
    owned the row), False if the lease had lapsed and another sweep reclaimed it.

    The CAS predicate (``status='in_flight' AND lease_expires_at = <our lease>``)
    is what makes at-least-once delivery safe under lease takeover: a stale
    sweeper that delivered slowly cannot move a row another sweep already
    delivered back to ``pending`` (which would also trip the delivered-shape CHECK
    into an IntegrityError and abort the batch) — its write simply matches no
    row."""
    with factory() as session:
        result = cast(
            "CursorResult[Any]",
            session.execute(
                update(NotificationDelivery)
                .where(
                    NotificationDelivery.id == claimed.id,
                    NotificationDelivery.status == NotificationStatus.IN_FLIGHT.value,
                    NotificationDelivery.lease_expires_at == claimed.lease_until,
                )
                .values(**values)
            ),
        )
        session.commit()
        return (result.rowcount or 0) == 1
