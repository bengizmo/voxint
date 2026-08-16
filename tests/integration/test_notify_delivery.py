"""Webhook delivery sweep (issue #12, phase C), against real Postgres + a
monkeypatched pinned transport.

Covers the delivery side: claim-under-lease (SKIP LOCKED, crash reclaim), 2xx →
delivered, non-2xx / transport error → retry with backoff, dead at the attempt
ceiling, FAILED suppression when the run advanced, deterministic body ↔
signature, redirects refused, non-public host refused, and the invariant that no
secret or URL ever lands in ``last_error``. Emission (record_transition) is a
separate phase; here rows are seeded directly.
"""

import socket
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import httpx
from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from voxint.config import Settings
from voxint.db.models import NotifiableEvent, NotificationDelivery, NotificationStatus
from voxint.notify import build_payload
from voxint.notify.delivery import (
    _Claimed,
    _finish,
    deliver_due,
    purge_expired_deliveries,
    serialize_payload,
    sign,
)

_SECRET = "a-sufficiently-long-secret"
_URL = "https://hooks.example.com/voxint"
_PUBLIC_IP = "93.184.216.34"


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "notify_enabled": True,
        "notify_webhook_url": _URL,
        "notify_webhook_secret": _SECRET,
        "notify_max_attempts": 8,
        "notify_batch_limit": 50,
        "notify_lease_seconds": 60,
        "notify_backoff_base_seconds": 10.0,
        "notify_backoff_max_seconds": 600.0,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _public_resolver(host: str, *_a: object, **_k: object) -> list[tuple[object, ...]]:
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (_PUBLIC_IP, 0))]


def _private_resolver(host: str, *_a: object, **_k: object) -> list[tuple[object, ...]]:
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))]


def _capturing_factory(
    handler: Callable[[httpx.Request], httpx.Response],
    captured: list[httpx.Request] | None = None,
) -> Callable[[httpx.Timeout], httpx.Client]:
    def wrapped(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(request)
        return handler(request)

    def factory(timeout: httpx.Timeout) -> httpx.Client:
        return httpx.Client(
            transport=httpx.MockTransport(wrapped),
            follow_redirects=False,
            trust_env=False,
            timeout=timeout,
        )

    return factory


def _ok(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text="ok")


def _seed_run(session: Session, *, revision: int = 0, status: str = "failed") -> uuid.UUID:
    rid = uuid.uuid4()
    mid = uuid.uuid4()
    session.execute(
        text("INSERT INTO media_items (id, source_path) VALUES (:mid, :sp)"),
        {"mid": mid, "sp": f"incoming/{mid}/source"},
    )
    session.execute(
        text(
            "INSERT INTO pipeline_runs (id, media_item_id, status, current_stage,"
            " revision, created_at, updated_at)"
            " VALUES (:rid, :mid, :st, 'finalize', :rev, now(), now())"
        ),
        {"rid": rid, "mid": mid, "st": status, "rev": revision},
    )
    return rid


def _seed_delivery(
    session: Session,
    run_id: uuid.UUID,
    *,
    event: NotifiableEvent = NotifiableEvent.COMPLETED,
    transition_revision: int = 1,
    status: NotificationStatus = NotificationStatus.PENDING,
    next_attempt_at: datetime | None = None,
    lease_expires_at: datetime | None = None,
    attempts: int = 0,
    created_at: datetime | None = None,
) -> uuid.UUID:
    did = uuid.uuid4()
    payload = build_payload(
        run_id=run_id,
        event=event,
        transition_revision=transition_revision,
        delivery_id=did,
    )
    row = NotificationDelivery(
        id=did,
        pipeline_run_id=run_id,
        transition_revision=transition_revision,
        event=event.value,
        payload=payload,
        status=status.value,
        attempts=attempts,
        next_attempt_at=next_attempt_at or (datetime.now(tz=UTC) - timedelta(seconds=1)),
        lease_expires_at=lease_expires_at,
        # The delivered-shape check constraint requires delivered_at iff DELIVERED.
        delivered_at=(created_at or datetime.now(tz=UTC))
        if status is NotificationStatus.DELIVERED
        else None,
    )
    if created_at is not None:
        row.created_at = created_at
    session.add(row)
    return did


def _row(session: Session, did: uuid.UUID) -> NotificationDelivery:
    return session.get_one(NotificationDelivery, did)


def test_delivers_2xx_marks_delivered(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        rid = _seed_run(session, status="completed")
        did = _seed_delivery(session, rid, event=NotifiableEvent.COMPLETED)
        session.commit()

    now = datetime.now(tz=UTC)
    summary = deliver_due(
        session_factory,
        _settings(),
        now=now,
        client_factory=_capturing_factory(_ok),
        resolver=_public_resolver,
    )
    assert summary.as_dict() == {
        "claimed": 1, "delivered": 1, "suppressed": 0, "retried": 0, "dead": 0
    }
    with session_factory() as session:
        row = _row(session, did)
        assert row.status == NotificationStatus.DELIVERED.value
        assert row.delivered_at is not None
        assert row.attempts == 1
        assert row.lease_expires_at is None
        assert row.last_error is None


def test_signature_and_headers_over_exact_bytes(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        rid = _seed_run(session, status="completed")
        did = _seed_delivery(session, rid, event=NotifiableEvent.COMPLETED)
        session.commit()
        payload = dict(_row(session, did).payload)

    captured: list[httpx.Request] = []
    now = datetime.now(tz=UTC)
    deliver_due(
        session_factory,
        _settings(),
        now=now,
        clock=lambda: now,  # pin the per-attempt signing clock for a deterministic ts
        client_factory=_capturing_factory(_ok, captured),
        resolver=_public_resolver,
    )
    assert len(captured) == 1
    req = captured[0]
    body = serialize_payload(payload)
    assert req.content == body
    # Host header carries the hostname; the request URL is pinned to the vetted IP.
    assert req.headers["Host"] == "hooks.example.com"
    assert req.url.host == _PUBLIC_IP
    assert req.headers["Content-Type"] == "application/json"
    assert req.headers["X-Voxint-Delivery"] == str(did)
    ts = req.headers["X-Voxint-Timestamp"]
    assert ts == str(int(now.timestamp()))
    assert req.headers["X-Voxint-Signature"] == f"sha256={sign(_SECRET, ts, body)}"


def test_non_2xx_retries_with_backoff(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        rid = _seed_run(session, status="completed")
        did = _seed_delivery(session, rid, event=NotifiableEvent.COMPLETED)
        session.commit()

    now = datetime.now(tz=UTC)
    summary = deliver_due(
        session_factory,
        _settings(),
        now=now,
        client_factory=_capturing_factory(lambda r: httpx.Response(500, text="boom")),
        resolver=_public_resolver,
    )
    assert summary.retried == 1
    with session_factory() as session:
        row = _row(session, did)
        assert row.status == NotificationStatus.PENDING.value
        assert row.attempts == 1
        assert row.lease_expires_at is None
        assert row.next_attempt_at > now  # backoff pushed it forward
        assert row.last_error is not None and "500" in row.last_error


def test_transport_error_retries(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        rid = _seed_run(session, status="completed")
        did = _seed_delivery(session, rid, event=NotifiableEvent.COMPLETED)
        session.commit()

    def boom(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    summary = deliver_due(
        session_factory,
        _settings(),
        now=datetime.now(tz=UTC),
        client_factory=_capturing_factory(boom),
        resolver=_public_resolver,
    )
    assert summary.retried == 1
    with session_factory() as session:
        row = _row(session, did)
        assert row.status == NotificationStatus.PENDING.value
        assert row.attempts == 1


def test_dead_after_max_attempts(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        rid = _seed_run(session, status="completed")
        did = _seed_delivery(
            session, rid, event=NotifiableEvent.COMPLETED, attempts=7  # max_attempts=8
        )
        session.commit()

    summary = deliver_due(
        session_factory,
        _settings(notify_max_attempts=8),
        now=datetime.now(tz=UTC),
        client_factory=_capturing_factory(lambda r: httpx.Response(503)),
        resolver=_public_resolver,
    )
    assert summary.dead == 1
    with session_factory() as session:
        row = _row(session, did)
        assert row.status == NotificationStatus.DEAD.value
        assert row.attempts == 8


def test_failed_suppressed_when_run_advanced(session_factory: sessionmaker[Session]) -> None:
    # Row was emitted at revision 1; the run has since been requeued to revision 2.
    with session_factory() as session:
        rid = _seed_run(session, revision=2, status="queued")
        did = _seed_delivery(
            session, rid, event=NotifiableEvent.FAILED, transition_revision=1
        )
        session.commit()

    captured: list[httpx.Request] = []
    summary = deliver_due(
        session_factory,
        _settings(),
        now=datetime.now(tz=UTC),
        client_factory=_capturing_factory(_ok, captured),
        resolver=_public_resolver,
    )
    assert summary.suppressed == 1
    assert captured == []  # never POSTed a stale "failed"
    with session_factory() as session:
        row = _row(session, did)
        assert row.status == NotificationStatus.SUPPRESSED.value


def test_failed_delivered_when_run_still_at_revision(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        rid = _seed_run(session, revision=1, status="failed")
        did = _seed_delivery(
            session, rid, event=NotifiableEvent.FAILED, transition_revision=1
        )
        session.commit()

    summary = deliver_due(
        session_factory,
        _settings(),
        now=datetime.now(tz=UTC),
        client_factory=_capturing_factory(_ok),
        resolver=_public_resolver,
    )
    assert summary.delivered == 1
    with session_factory() as session:
        assert _row(session, did).status == NotificationStatus.DELIVERED.value


def test_future_pending_not_claimed(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        rid = _seed_run(session, status="completed")
        did = _seed_delivery(
            session,
            rid,
            next_attempt_at=datetime.now(tz=UTC) + timedelta(minutes=5),
        )
        session.commit()

    summary = deliver_due(
        session_factory,
        _settings(),
        now=datetime.now(tz=UTC),
        client_factory=_capturing_factory(_ok),
        resolver=_public_resolver,
    )
    assert summary.claimed == 0
    with session_factory() as session:
        assert _row(session, did).status == NotificationStatus.PENDING.value


def test_lapsed_in_flight_lease_is_reclaimed(session_factory: sessionmaker[Session]) -> None:
    # A prior sweep crashed mid-delivery: row is in_flight with an expired lease.
    with session_factory() as session:
        rid = _seed_run(session, status="completed")
        did = _seed_delivery(
            session,
            rid,
            status=NotificationStatus.IN_FLIGHT,
            lease_expires_at=datetime.now(tz=UTC) - timedelta(seconds=1),
        )
        session.commit()

    summary = deliver_due(
        session_factory,
        _settings(),
        now=datetime.now(tz=UTC),
        client_factory=_capturing_factory(_ok),
        resolver=_public_resolver,
    )
    assert summary.delivered == 1
    with session_factory() as session:
        assert _row(session, did).status == NotificationStatus.DELIVERED.value


def test_unexpired_in_flight_lease_not_reclaimed(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        rid = _seed_run(session, status="completed")
        did = _seed_delivery(
            session,
            rid,
            status=NotificationStatus.IN_FLIGHT,
            lease_expires_at=datetime.now(tz=UTC) + timedelta(minutes=5),
        )
        session.commit()

    summary = deliver_due(
        session_factory,
        _settings(),
        now=datetime.now(tz=UTC),
        client_factory=_capturing_factory(_ok),
        resolver=_public_resolver,
    )
    assert summary.claimed == 0
    with session_factory() as session:
        assert _row(session, did).status == NotificationStatus.IN_FLIGHT.value


def test_concurrent_claim_skips_locked_row(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        rid = _seed_run(session, status="completed")
        _seed_delivery(session, rid)
        session.commit()

    # Hold a FOR UPDATE lock on the only due row in a separate uncommitted tx;
    # the sweep's SKIP LOCKED claim must find nothing rather than block.
    holder = session_factory()
    try:
        locked = holder.execute(
            select(NotificationDelivery)
            .where(NotificationDelivery.status == NotificationStatus.PENDING.value)
            .with_for_update(skip_locked=True)
        ).scalars().all()
        assert len(locked) == 1  # the holder owns it

        summary = deliver_due(
            session_factory,
            _settings(),
            now=datetime.now(tz=UTC),
            client_factory=_capturing_factory(_ok),
            resolver=_public_resolver,
        )
        assert summary.claimed == 0
    finally:
        holder.rollback()
        holder.close()


def test_batch_limit_bounds_claim(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        for _ in range(3):
            rid = _seed_run(session, status="completed")
            _seed_delivery(session, rid)
        session.commit()

    summary = deliver_due(
        session_factory,
        _settings(notify_batch_limit=2),
        now=datetime.now(tz=UTC),
        client_factory=_capturing_factory(_ok),
        resolver=_public_resolver,
    )
    assert summary.claimed == 2
    assert summary.delivered == 2


def test_redirect_refused_not_followed(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        rid = _seed_run(session, status="completed")
        did = _seed_delivery(session, rid)
        session.commit()

    captured: list[httpx.Request] = []

    def redirect(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "https://evil.example.com/x"})

    summary = deliver_due(
        session_factory,
        _settings(),
        now=datetime.now(tz=UTC),
        client_factory=_capturing_factory(redirect, captured),
        resolver=_public_resolver,
    )
    # A 3xx is a non-2xx answer: one request made, no hop chased, row retried.
    assert len(captured) == 1
    assert summary.retried == 1
    with session_factory() as session:
        row = _row(session, did)
        assert row.status == NotificationStatus.PENDING.value
        assert "evil.example.com" not in (row.last_error or "")


def test_non_public_host_refused_no_post(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        rid = _seed_run(session, status="completed")
        did = _seed_delivery(session, rid)
        session.commit()

    captured: list[httpx.Request] = []
    summary = deliver_due(
        session_factory,
        _settings(),
        now=datetime.now(tz=UTC),
        client_factory=_capturing_factory(_ok, captured),
        resolver=_private_resolver,  # resolves to loopback → refused
    )
    assert captured == []  # SSRF target never contacted
    assert summary.retried == 1
    with session_factory() as session:
        row = _row(session, did)
        assert row.status == NotificationStatus.PENDING.value
        assert "hooks.example.com" in (row.last_error or "")


def test_secret_and_url_never_in_last_error(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        rid = _seed_run(session, status="completed")
        did = _seed_delivery(session, rid)
        session.commit()

    # A handler whose error text tries to echo the secret and full URL back.
    def leaky(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"failed talking to {_URL} with {_SECRET}")

    deliver_due(
        session_factory,
        _settings(),
        now=datetime.now(tz=UTC),
        client_factory=_capturing_factory(leaky),
        resolver=_public_resolver,
    )
    with session_factory() as session:
        last_error = _row(session, did).last_error or ""
        assert _SECRET not in last_error
        assert "/voxint" not in last_error  # the URL path is redacted away


def test_purge_reaps_only_old_settled_rows(session_factory: sessionmaker[Session]) -> None:
    now = datetime.now(tz=UTC)
    old = now - timedelta(days=30)
    with session_factory() as session:
        rid = _seed_run(session, status="completed")
        old_delivered = _seed_delivery(
            session, rid, transition_revision=1, status=NotificationStatus.DELIVERED, created_at=old
        )
        old_suppressed = _seed_delivery(
            session,
            rid,
            transition_revision=2,
            status=NotificationStatus.SUPPRESSED,
            created_at=old,
        )
        old_dead = _seed_delivery(
            session, rid, transition_revision=3, status=NotificationStatus.DEAD, created_at=old
        )
        fresh_delivered = _seed_delivery(
            session, rid, transition_revision=4, status=NotificationStatus.DELIVERED, created_at=now
        )
        old_pending = _seed_delivery(
            session, rid, transition_revision=5, status=NotificationStatus.PENDING, created_at=old
        )
        session.commit()

    purged = purge_expired_deliveries(
        session_factory, _settings(notify_retention_seconds=3600), now=now
    )
    assert purged == 2  # only the old delivered + suppressed rows
    with session_factory() as session:
        surviving = {
            r.id for r in session.execute(select(NotificationDelivery)).scalars().all()
        }
    assert old_delivered not in surviving
    assert old_suppressed not in surviving
    assert old_dead in surviving  # kept until an operator acts
    assert fresh_delivered in surviving  # within retention
    assert old_pending in surviving  # still live, never a purge target


def test_finish_is_lease_guarded_against_stale_sweeper(
    session_factory: sessionmaker[Session],
) -> None:
    # An in_flight row leased to sweep B; sweep A (crashed-slow, lease L_old) must
    # not clobber it. _finish CASes on (in_flight AND lease == our lease).
    lease_b = datetime.now(tz=UTC) + timedelta(seconds=60)
    lease_a_stale = lease_b - timedelta(seconds=120)
    with session_factory() as session:
        rid = _seed_run(session, status="completed")
        did = _seed_delivery(
            session,
            rid,
            status=NotificationStatus.IN_FLIGHT,
            lease_expires_at=lease_b,
        )
        session.commit()

    def _claimed(lease: datetime) -> _Claimed:
        return _Claimed(
            id=did,
            run_id=rid,
            event=NotifiableEvent.COMPLETED.value,
            transition_revision=1,
            payload={},
            attempts=0,
            lease_until=lease,
        )

    # Stale owner: write does not land, row untouched.
    landed = _finish(
        session_factory,
        _claimed(lease_a_stale),
        status=NotificationStatus.PENDING.value,
        lease_expires_at=None,
    )
    assert landed is False
    with session_factory() as session:
        assert _row(session, did).status == NotificationStatus.IN_FLIGHT.value

    # Current owner (lease matches): write lands.
    landed = _finish(
        session_factory,
        _claimed(lease_b),
        status=NotificationStatus.DELIVERED.value,
        delivered_at=datetime.now(tz=UTC),
        attempts=1,
        lease_expires_at=None,
        last_error=None,
    )
    assert landed is True
    with session_factory() as session:
        assert _row(session, did).status == NotificationStatus.DELIVERED.value


def test_multi_address_failover(session_factory: sessionmaker[Session]) -> None:
    # First vetted address connect-fails; the second succeeds. The row delivers
    # and the successful POST is pinned to the working address.
    dead_ip, good_ip = "93.184.216.34", "93.184.216.35"

    def resolver(host: str, *_a: object, **_k: object) -> list[tuple[object, ...]]:
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", (dead_ip, 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", (good_ip, 0)),
        ]

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == dead_ip:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200)

    with session_factory() as session:
        rid = _seed_run(session, status="completed")
        did = _seed_delivery(session, rid)
        session.commit()

    summary = deliver_due(
        session_factory,
        _settings(),
        now=datetime.now(tz=UTC),
        client_factory=_capturing_factory(handler, captured),
        resolver=resolver,
    )
    assert summary.delivered == 1
    assert [r.url.host for r in captured] == [dead_ip, good_ip]  # tried dead, then good
    with session_factory() as session:
        assert _row(session, did).status == NotificationStatus.DELIVERED.value


def test_signature_timestamp_is_fresh_per_row(session_factory: sessionmaker[Session]) -> None:
    # Two rows in one sweep must be signed with per-attempt timestamps, not a
    # single batch-start time (a slow batch would otherwise sign tail rows stale).
    with session_factory() as session:
        for i in range(2):
            rid = _seed_run(session, status="completed")
            _seed_delivery(session, rid, transition_revision=i + 1)
        session.commit()

    t1 = datetime(2026, 1, 1, 0, 0, 10, tzinfo=UTC)  # row 1 attempt
    t2 = datetime(2026, 1, 1, 0, 5, 0, tzinfo=UTC)  # row 2 attempt (5 min later)
    ticks = iter([t1, t2])  # claim uses the explicit `now=`, so clock only signs
    captured: list[httpx.Request] = []
    deliver_due(
        session_factory,
        _settings(),
        now=datetime.now(tz=UTC),  # real, so both seeded rows are due
        clock=lambda: next(ticks),
        client_factory=_capturing_factory(_ok, captured),
        resolver=_public_resolver,
    )
    stamps = sorted(int(r.headers["X-Voxint-Timestamp"]) for r in captured)
    assert stamps == [int(t1.timestamp()), int(t2.timestamp())]


def test_purge_bounded_by_batch_limit(session_factory: sessionmaker[Session]) -> None:
    now = datetime.now(tz=UTC)
    old = now - timedelta(days=30)
    with session_factory() as session:
        rid = _seed_run(session, status="completed")
        for i in range(5):
            _seed_delivery(
                session,
                rid,
                transition_revision=i + 1,
                status=NotificationStatus.DELIVERED,
                created_at=old,
            )
        session.commit()

    purged = purge_expired_deliveries(
        session_factory, _settings(notify_retention_seconds=3600, notify_batch_limit=2), now=now
    )
    assert purged == 2  # one bounded batch; the rest drain on later sweeps
