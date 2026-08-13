"""POST /runs/{id}/requeue — the browser requeue of a FAILED run, end to end.

The requeue *service* (CAS, revision guard, not-FAILED/missing-stage refusals) is
covered at the ingest layer in ``test_ingest_service.py``; here we test the ROUTE:
its HTTP status mapping, the exact-revision form guard, commit-before-publish, the
broker-down degradation it shares with /submit, and the failed-only Requeue button.
"""

import re
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_FETCH, CSRF_REQUEUE, mint_csrf_token
from voxint.config import Settings
from voxint.db.models import STAGE_ORDER, PipelineRun, RunStatus, Stage
from voxint.ingest import submit_media_item
from voxint.pipeline.transitions import (
    RunSnapshot,
    cas_update_run,
    next_stage,
    snapshot,
)

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "requeue-api-test-csrf-key"  # low-entropy; a known secret lets tests mint


def _rd(**kwargs: str) -> dict[str, str]:
    """Form fields (revision, …) with a valid /requeue CSRF token merged in — the
    real requeue form carries one; posting without it is 403."""
    return {"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_REQUEUE), **kwargs}


@pytest.fixture()
def client(session_factory: sessionmaker[Session]) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0], voxint_password=CREDS[1], csrf_secret=_CSRF_KEY
    )
    test_client = TestClient(create_app(settings=settings, session_factory=session_factory))
    test_client.auth = CREDS
    seed_onboarded(session_factory)
    return test_client


@pytest.fixture()
def published(monkeypatch: pytest.MonkeyPatch) -> list[uuid.UUID]:
    """Capture commit-before-publish enqueues without a live broker."""
    calls: list[uuid.UUID] = []
    monkeypatch.setattr("voxint.api.app._publish_run", calls.append)
    return calls


def _drive_to_failed(session: Session, run_id: uuid.UUID, stage: Stage) -> RunSnapshot:
    """Walk a QUEUED run to FAILED at ``stage`` through real CAS transitions, so
    the run's ``revision`` reflects a state the machine can genuinely produce."""
    held = snapshot(session.get(PipelineRun, run_id))  # type: ignore[arg-type]
    held = cas_update_run(
        session, held, status=RunStatus.RUNNING, current_stage=held.current_stage or STAGE_ORDER[0]
    )
    while held.current_stage is not stage:
        nxt = next_stage(held.current_stage)
        assert nxt is not None, f"{stage!r} not reachable in STAGE_ORDER"
        held = cas_update_run(session, held, status=RunStatus.RUNNING, current_stage=nxt)
    held = cas_update_run(
        session, held, status=RunStatus.FAILED, current_stage=stage, error="boom"
    )
    session.commit()
    return held


def _make_failed_run(
    session_factory: sessionmaker[Session], *, stage: Stage = Stage.TRANSCRIBE
) -> tuple[uuid.UUID, int]:
    """Seed a FAILED run at ``stage``; return its id and current revision."""
    with session_factory() as session:
        run_id = submit_media_item(session, f"incoming/{uuid.uuid4()}.wav").id
        session.commit()
        failed = _drive_to_failed(session, run_id, stage)
    return run_id, failed.revision


# --- happy path ---------------------------------------------------------------


def test_requeue_failed_run_returns_queued_and_publishes(
    client: TestClient,
    session_factory: sessionmaker[Session],
    published: list[uuid.UUID],
) -> None:
    run_id, revision = _make_failed_run(session_factory, stage=Stage.TRANSCRIBE)

    resp = client.post(
        f"/runs/{run_id}/requeue",
        data=_rd(revision=str(revision)),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == f"/runs/{run_id}"  # no deferred marker

    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.QUEUED.value
        assert run.current_stage == Stage.TRANSCRIBE.value  # requeued at its failed stage
    assert published == [run_id]  # commit-before-publish fired exactly once


# --- exact-revision CAS guard -------------------------------------------------


def test_requeue_with_stale_revision_conflicts_and_leaves_run_failed(
    client: TestClient,
    session_factory: sessionmaker[Session],
    published: list[uuid.UUID],
) -> None:
    # A stale browser tab submitting an older revision must 409 and change nothing.
    run_id, revision = _make_failed_run(session_factory)

    resp = client.post(
        f"/runs/{run_id}/requeue",
        data=_rd(revision=str(revision - 1)),
        follow_redirects=False,
    )
    assert resp.status_code == 409

    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.FAILED.value  # untouched
    assert published == []  # a rejected requeue never publishes


def test_requeue_non_failed_run_conflicts(
    client: TestClient,
    session_factory: sessionmaker[Session],
    published: list[uuid.UUID],
) -> None:
    # A QUEUED run carries a real revision but is not requeuable → 409, not 5xx.
    with session_factory() as session:
        run = submit_media_item(session, "incoming/queued.wav")
        session.commit()
        run_id, revision = run.id, run.revision

    resp = client.post(
        f"/runs/{run_id}/requeue",
        data=_rd(revision=str(revision)),
        follow_redirects=False,
    )
    assert resp.status_code == 409
    assert published == []


def test_requeue_missing_run_is_404(
    client: TestClient, published: list[uuid.UUID]
) -> None:
    resp = client.post(
        f"/runs/{uuid.uuid4()}/requeue",
        data=_rd(revision="1"),
        follow_redirects=False,
    )
    assert resp.status_code == 404
    assert published == []


# --- CSRF ---------------------------------------------------------------------


def test_requeue_rejected_without_csrf_token(
    client: TestClient,
    session_factory: sessionmaker[Session],
    published: list[uuid.UUID],
) -> None:
    # No csrf_token ⇒ 403 before the CAS runs; the run stays FAILED, untouched.
    run_id, revision = _make_failed_run(session_factory)
    resp = client.post(
        f"/runs/{run_id}/requeue",
        data={"revision": str(revision)},  # NB: no csrf_token
        follow_redirects=False,
    )
    assert resp.status_code == 403
    assert published == []
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.FAILED.value


def test_requeue_rejected_with_wrong_action_token(
    client: TestClient,
    session_factory: sessionmaker[Session],
    published: list[uuid.UUID],
) -> None:
    # A token minted for /fetch is not valid on /requeue (action binding).
    run_id, revision = _make_failed_run(session_factory)
    resp = client.post(
        f"/runs/{run_id}/requeue",
        data={
            "revision": str(revision),
            "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_FETCH),
        },
        follow_redirects=False,
    )
    assert resp.status_code == 403
    assert published == []


def test_requeue_form_renders_valid_csrf_token(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    from voxint.api.csrf import verify_csrf_token

    run_id, _revision = _make_failed_run(session_factory)
    body = client.get(f"/runs/{run_id}").text
    match = re.search(
        r'action="/runs/[^"]+/requeue".*?name="csrf_token" value="([^"]+)"',
        body,
        re.DOTALL,
    )
    assert match is not None
    assert verify_csrf_token(_CSRF_KEY, CSRF_REQUEUE, match.group(1))


# --- broker-down degradation --------------------------------------------------


def test_broker_down_requeue_stays_queued_and_flags_banner(
    client: TestClient,
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from celery.exceptions import OperationalError

    run_id, revision = _make_failed_run(session_factory)

    def _broker_down(_run_id: uuid.UUID) -> None:
        raise OperationalError("Error 111 connecting to redis. Connection refused.")

    monkeypatch.setattr("voxint.api.app._publish_run", _broker_down)

    resp = client.post(
        f"/runs/{run_id}/requeue",
        data=_rd(revision=str(revision)),
        follow_redirects=False,
    )
    # The CAS already moved the run to QUEUED and committed; only the enqueue was
    # deferred, so the request degrades cleanly rather than 500ing.
    assert resp.status_code == 303
    assert resp.headers["location"] == f"/runs/{run_id}?enqueue=deferred"

    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.QUEUED.value


# --- the Requeue button is failed-only ----------------------------------------


def test_requeue_button_shown_only_for_failed_run(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, revision = _make_failed_run(session_factory)
    body = client.get(f"/runs/{run_id}").text
    assert f'action="/runs/{run_id}/requeue"' in body
    assert 'name="revision"' in body
    assert f'value="{revision}"' in body  # carries the rendered revision for the CAS


def test_no_requeue_button_for_queued_run(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        run_id = submit_media_item(session, "incoming/notfailed.wav").id
        session.commit()
    body = client.get(f"/runs/{run_id}").text
    assert "/requeue" not in body


# --- deferred banner is gated on the live status ------------------------------


def test_deferred_banner_suppressed_once_run_left_queued(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # A bookmarked/refreshed ?enqueue=deferred URL must not keep claiming the run
    # "is queued and will start" after the sweep republished it (dual-review
    # finding). The banner is gated on run.status == "queued"; a FAILED run is
    # past that, so it must not render even with the marker present.
    run_id, _revision = _make_failed_run(session_factory)
    body = client.get(f"/runs/{run_id}?enqueue=deferred").text
    assert "enqueue was deferred" not in body
