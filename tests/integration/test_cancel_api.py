"""POST /runs/{id}/cancel — the browser cancel of a live run, end to end (#5).

The cancel *service* (CAS, revision guard, idempotency, non-cancellable refusals)
is covered at the ingest layer in ``test_ingest_service.py``; here we test the
ROUTE: its HTTP status mapping, the exact-revision form guard, no-publish
(cancel is pure DB state), the idempotent double-cancel, and the live-only Cancel
button.
"""

import re
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_CANCEL, CSRF_REQUEUE, mint_csrf_token
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
_CSRF_KEY = "cancel-api-test-csrf-key"  # low-entropy; a known secret lets tests mint


def _cd(**kwargs: str) -> dict[str, str]:
    """Form fields (revision, …) with a valid /cancel CSRF token merged in."""
    return {"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_CANCEL), **kwargs}


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
    """Capture any enqueue — cancel must publish NOTHING (it is pure DB state)."""
    calls: list[uuid.UUID] = []
    monkeypatch.setattr(
        "voxint.api.routers.deps._publish_run", lambda run_id, **_kwargs: calls.append(run_id)
    )

    from voxint.ingest.service import SubmissionResult

    def _record_publish(self: SubmissionResult) -> bool:
        calls.append(self.run_id)
        return True

    monkeypatch.setattr(SubmissionResult, "publish", _record_publish)
    return calls


def _drive_to_running(session: Session, run_id: uuid.UUID, stage: Stage) -> RunSnapshot:
    held = snapshot(session.get(PipelineRun, run_id))  # type: ignore[arg-type]
    held = cas_update_run(
        session, held, status=RunStatus.RUNNING, current_stage=held.current_stage or STAGE_ORDER[0]
    )
    while held.current_stage is not stage:
        nxt = next_stage(held.current_stage)
        assert nxt is not None
        held = cas_update_run(session, held, status=RunStatus.RUNNING, current_stage=nxt)
    session.commit()
    return held


def _make_queued_run(session_factory: sessionmaker[Session]) -> tuple[uuid.UUID, int]:
    with session_factory() as session:
        result = submit_media_item(session, f"incoming/{uuid.uuid4()}.wav")
        session.commit()
        run = session.get(PipelineRun, result.run_id)
        return run.id, run.revision


def _make_running_run(
    session_factory: sessionmaker[Session], *, stage: Stage = Stage.TRANSCRIBE
) -> tuple[uuid.UUID, int]:
    run_id, _ = _make_queued_run(session_factory)
    with session_factory() as session:
        held = _drive_to_running(session, run_id, stage)
    return run_id, held.revision


# --- happy path ---------------------------------------------------------------


def test_cancel_running_run_redirects_and_cancels(
    client: TestClient,
    session_factory: sessionmaker[Session],
    published: list[uuid.UUID],
) -> None:
    run_id, revision = _make_running_run(session_factory)

    resp = client.post(
        f"/runs/{run_id}/cancel",
        data=_cd(revision=str(revision)),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == f"/runs/{run_id}"

    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.CANCELLED.value
        assert run.current_stage == Stage.TRANSCRIBE.value  # stage kept
    assert published == []  # cancel never enqueues


def test_cancel_queued_run_cancels(
    client: TestClient,
    session_factory: sessionmaker[Session],
    published: list[uuid.UUID],
) -> None:
    run_id, revision = _make_queued_run(session_factory)

    resp = client.post(
        f"/runs/{run_id}/cancel",
        data=_cd(revision=str(revision)),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.CANCELLED.value
    assert published == []


def test_cancel_awaiting_adjudication_run(
    client: TestClient,
    session_factory: sessionmaker[Session],
    published: list[uuid.UUID],
) -> None:
    # A human-paused run (no live worker) is cancellable through the route.
    run_id, _ = _make_running_run(session_factory, stage=Stage.DIARIZE_EMBED)
    with session_factory() as session:
        held = snapshot(session.get(PipelineRun, run_id))  # type: ignore[arg-type]
        held = cas_update_run(
            session,
            held,
            status=RunStatus.AWAITING_ADJUDICATION,
            current_stage=Stage.DIARIZE_EMBED,
        )
        session.commit()
        revision = held.revision

    resp = client.post(
        f"/runs/{run_id}/cancel",
        data=_cd(revision=str(revision)),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.CANCELLED.value
    assert published == []


def test_cancel_completed_run_conflicts(
    client: TestClient,
    session_factory: sessionmaker[Session],
    published: list[uuid.UUID],
) -> None:
    # COMPLETED is terminal → 409, not 5xx.
    run_id, _ = _make_running_run(session_factory, stage=STAGE_ORDER[-1])
    with session_factory() as session:
        held = snapshot(session.get(PipelineRun, run_id))  # type: ignore[arg-type]
        held = cas_update_run(session, held, status=RunStatus.COMPLETED, current_stage=None)
        session.commit()
        revision = held.revision

    resp = client.post(
        f"/runs/{run_id}/cancel",
        data=_cd(revision=str(revision)),
        follow_redirects=False,
    )
    assert resp.status_code == 409
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.COMPLETED.value


# --- exact-revision CAS guard -------------------------------------------------


def test_cancel_with_stale_revision_conflicts_and_leaves_run_running(
    client: TestClient,
    session_factory: sessionmaker[Session],
    published: list[uuid.UUID],
) -> None:
    run_id, revision = _make_running_run(session_factory)

    resp = client.post(
        f"/runs/{run_id}/cancel",
        data=_cd(revision=str(revision - 1)),
        follow_redirects=False,
    )
    assert resp.status_code == 409

    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.RUNNING.value  # untouched


def test_cancel_failed_run_conflicts(
    client: TestClient,
    session_factory: sessionmaker[Session],
    published: list[uuid.UUID],
) -> None:
    # FAILED is requeueable, not cancellable → 409, not 5xx.
    run_id, _ = _make_running_run(session_factory)
    with session_factory() as session:
        held = snapshot(session.get(PipelineRun, run_id))  # type: ignore[arg-type]
        held = cas_update_run(
            session, held, status=RunStatus.FAILED, current_stage=held.current_stage, error="boom"
        )
        session.commit()
        revision = held.revision

    resp = client.post(
        f"/runs/{run_id}/cancel",
        data=_cd(revision=str(revision)),
        follow_redirects=False,
    )
    assert resp.status_code == 409
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.FAILED.value


def test_cancel_already_cancelled_is_idempotent_303(
    client: TestClient,
    session_factory: sessionmaker[Session],
    published: list[uuid.UUID],
) -> None:
    # A second cancel from a stale tab must be an idempotent success, not a 409.
    run_id, revision = _make_queued_run(session_factory)
    first = client.post(
        f"/runs/{run_id}/cancel", data=_cd(revision=str(revision)), follow_redirects=False
    )
    assert first.status_code == 303

    # The stale tab still holds the pre-cancel revision; cancel again → 303.
    again = client.post(
        f"/runs/{run_id}/cancel", data=_cd(revision=str(revision)), follow_redirects=False
    )
    assert again.status_code == 303
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.CANCELLED.value


def test_cancel_missing_run_is_404(
    client: TestClient, published: list[uuid.UUID]
) -> None:
    resp = client.post(
        f"/runs/{uuid.uuid4()}/cancel",
        data=_cd(revision="1"),
        follow_redirects=False,
    )
    assert resp.status_code == 404
    assert published == []


# --- CSRF ---------------------------------------------------------------------


def test_cancel_rejected_without_csrf_token(
    client: TestClient,
    session_factory: sessionmaker[Session],
    published: list[uuid.UUID],
) -> None:
    run_id, revision = _make_running_run(session_factory)
    resp = client.post(
        f"/runs/{run_id}/cancel",
        data={"revision": str(revision)},  # NB: no csrf_token
        follow_redirects=False,
    )
    assert resp.status_code == 403
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.RUNNING.value


def test_cancel_rejected_with_wrong_action_token(
    client: TestClient,
    session_factory: sessionmaker[Session],
    published: list[uuid.UUID],
) -> None:
    # A token minted for /requeue is not valid on /cancel (action binding).
    run_id, revision = _make_running_run(session_factory)
    resp = client.post(
        f"/runs/{run_id}/cancel",
        data={
            "revision": str(revision),
            "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_REQUEUE),
        },
        follow_redirects=False,
    )
    assert resp.status_code == 403


# --- the Cancel button is live-only -------------------------------------------


def test_cancel_button_shown_for_running_run(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, revision = _make_running_run(session_factory)
    body = client.get(f"/runs/{run_id}").text
    assert f'action="/runs/{run_id}/cancel"' in body
    assert 'name="revision"' in body
    assert f'value="{revision}"' in body


def test_cancel_button_shown_for_queued_run(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, _ = _make_queued_run(session_factory)
    body = client.get(f"/runs/{run_id}").text
    assert f'action="/runs/{run_id}/cancel"' in body


def test_no_cancel_button_for_cancelled_run(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id, revision = _make_queued_run(session_factory)
    client.post(f"/runs/{run_id}/cancel", data=_cd(revision=str(revision)))
    body = client.get(f"/runs/{run_id}").text
    assert "/cancel" not in body


def test_cancel_form_renders_valid_csrf_token(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    from voxint.api.csrf import verify_csrf_token

    run_id, _ = _make_running_run(session_factory)
    body = client.get(f"/runs/{run_id}").text
    match = re.search(
        r'action="/runs/[^"]+/cancel".*?name="csrf_token" value="([^"]+)"',
        body,
        re.DOTALL,
    )
    assert match is not None
    assert verify_csrf_token(_CSRF_KEY, CSRF_CANCEL, match.group(1))
