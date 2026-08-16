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
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta

import httpx
from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from voxint.config import Settings
from voxint.db.models import NotifiableEvent, NotificationDelivery, NotificationStatus
from voxint.notify import build_payload
from voxint.notify.delivery import deliver_due, serialize_payload, sign

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
    )
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
