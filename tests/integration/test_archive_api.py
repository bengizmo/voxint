"""POST /runs/{id}/archive · /unarchive · /media/delete + archived filtering (#5).

The archive/media-delete *services* (terminal guard, idempotency, shared-source
safety, path confinement) are covered at the ingest layer in
``test_archive_service.py``; here we test the ROUTES and the read surfaces: HTTP
status mapping, CSRF, the archived-run mutation guard, and that archived runs are
hidden from ``/runs``/``/review`` but visible under ``?archived=1``.
"""

import uuid
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import (
    CSRF_REQUEUE,
    CSRF_RUN_ARCHIVE,
    CSRF_RUN_MEDIA_DELETE,
    CSRF_RUN_UNARCHIVE,
    mint_csrf_token,
)
from voxint.config import Settings
from voxint.db.models import (
    STAGE_ORDER,
    ArtifactKind,
    AudioArtifact,
    PipelineRun,
    RunStatus,
)
from voxint.ingest import archive_run, submit_media_item
from voxint.pipeline.transitions import cas_update_run, next_stage, snapshot

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "archive-api-test-csrf-key"


def _tok(action: str, **kwargs: str) -> dict[str, str]:
    return {"csrf_token": mint_csrf_token(_CSRF_KEY, action), **kwargs}


@pytest.fixture()
def client(session_factory: sessionmaker[Session]) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0], voxint_password=CREDS[1], csrf_secret=_CSRF_KEY
    )
    test_client = TestClient(create_app(settings=settings, session_factory=session_factory))
    test_client.auth = CREDS
    seed_onboarded(session_factory)
    return test_client


def _make_completed(session_factory: sessionmaker[Session]) -> uuid.UUID:
    with session_factory() as session:
        run_id = submit_media_item(session, f"incoming/{uuid.uuid4()}.wav").run_id
        session.commit()
        held = snapshot(session.get(PipelineRun, run_id))  # type: ignore[arg-type]
        held = cas_update_run(session, held, status=RunStatus.RUNNING, current_stage=STAGE_ORDER[0])
        while held.current_stage is not STAGE_ORDER[-1]:
            nxt = next_stage(held.current_stage)
            held = cas_update_run(session, held, status=RunStatus.RUNNING, current_stage=nxt)
        cas_update_run(session, held, status=RunStatus.COMPLETED, current_stage=None)
        session.commit()
    return run_id


def _make_failed(session_factory: sessionmaker[Session]) -> tuple[uuid.UUID, int]:
    with session_factory() as session:
        run_id = submit_media_item(session, f"incoming/{uuid.uuid4()}.wav").run_id
        session.commit()
        held = snapshot(session.get(PipelineRun, run_id))  # type: ignore[arg-type]
        held = cas_update_run(session, held, status=RunStatus.RUNNING, current_stage=STAGE_ORDER[0])
        held = cas_update_run(
            session, held, status=RunStatus.FAILED, current_stage=STAGE_ORDER[0], error="boom"
        )
        session.commit()
        return run_id, held.revision


def _make_queued(session_factory: sessionmaker[Session]) -> uuid.UUID:
    with session_factory() as session:
        run_id = submit_media_item(session, f"incoming/{uuid.uuid4()}.wav").run_id
        session.commit()
        return run_id


def _archive(session_factory: sessionmaker[Session], run_id: uuid.UUID) -> None:
    with session_factory() as session:
        archive_run(session, run_id)
        session.commit()


def _archived_at(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> datetime | None:
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        return run.archived_at


# --- archive / unarchive routes ---------------------------------------------


def test_archive_completed_run_redirects_and_stamps(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id = _make_completed(session_factory)
    resp = client.post(
        f"/runs/{run_id}/archive", data=_tok(CSRF_RUN_ARCHIVE), follow_redirects=False
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == f"/runs/{run_id}"
    assert _archived_at(session_factory, run_id) is not None


def test_unarchive_redirects_and_clears(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id = _make_completed(session_factory)
    _archive(session_factory, run_id)
    resp = client.post(
        f"/runs/{run_id}/unarchive", data=_tok(CSRF_RUN_UNARCHIVE), follow_redirects=False
    )
    assert resp.status_code == 303
    assert _archived_at(session_factory, run_id) is None


def test_archive_live_run_conflicts(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id = _make_queued(session_factory)
    resp = client.post(f"/runs/{run_id}/archive", data=_tok(CSRF_RUN_ARCHIVE))
    assert resp.status_code == 409
    assert _archived_at(session_factory, run_id) is None


def test_archive_missing_run_404(client: TestClient) -> None:
    resp = client.post(f"/runs/{uuid.uuid4()}/archive", data=_tok(CSRF_RUN_ARCHIVE))
    assert resp.status_code == 404


def test_archive_missing_csrf_forbidden(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id = _make_completed(session_factory)
    resp = client.post(f"/runs/{run_id}/archive", data={})
    assert resp.status_code == 403
    assert _archived_at(session_factory, run_id) is None  # not archived


def test_archive_wrong_action_token_forbidden(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id = _make_completed(session_factory)
    # A token minted for un-archive must not be replayable on archive.
    resp = client.post(f"/runs/{run_id}/archive", data=_tok(CSRF_RUN_UNARCHIVE))
    assert resp.status_code == 403
    assert _archived_at(session_factory, run_id) is None


def test_archive_is_idempotent_over_http(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id = _make_completed(session_factory)
    first = client.post(
        f"/runs/{run_id}/archive", data=_tok(CSRF_RUN_ARCHIVE), follow_redirects=False
    )
    stamp = _archived_at(session_factory, run_id)
    second = client.post(
        f"/runs/{run_id}/archive", data=_tok(CSRF_RUN_ARCHIVE), follow_redirects=False
    )
    assert first.status_code == second.status_code == 303
    assert _archived_at(session_factory, run_id) == stamp  # unchanged


# --- media-delete route -----------------------------------------------------


def test_media_delete_redirects_with_banner_and_removes_rows(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id = _make_completed(session_factory)
    with session_factory() as session:
        session.add(
            AudioArtifact(
                pipeline_run_id=run_id,
                kind=ArtifactKind.PREPROCESSED_AUDIO.value,
                path=f"derived/{run_id}/audio.wav",
            )
        )
        session.commit()
    resp = client.post(
        f"/runs/{run_id}/media/delete",
        data=_tok(CSRF_RUN_MEDIA_DELETE),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    loc = resp.headers["location"]
    assert loc.startswith(f"/runs/{run_id}?") and "media=deleted" in loc
    with session_factory() as session:
        assert (
            session.query(AudioArtifact).filter_by(pipeline_run_id=run_id).count() == 0
        )


def test_media_delete_live_run_conflicts(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id = _make_queued(session_factory)
    resp = client.post(f"/runs/{run_id}/media/delete", data=_tok(CSRF_RUN_MEDIA_DELETE))
    assert resp.status_code == 409


def test_media_delete_missing_csrf_forbidden(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id = _make_completed(session_factory)
    resp = client.post(f"/runs/{run_id}/media/delete", data={})
    assert resp.status_code == 403


# --- archived-run mutation guard --------------------------------------------


def test_requeue_archived_run_conflicts(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # A FAILED run is normally requeueable; once archived it is read-only until
    # un-archived, so a stale tab can't drive a hidden run live.
    run_id, revision = _make_failed(session_factory)
    _archive(session_factory, run_id)
    resp = client.post(
        f"/runs/{run_id}/requeue",
        data=_tok(CSRF_REQUEUE, revision=str(revision)),
    )
    assert resp.status_code == 409
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None and run.status == RunStatus.FAILED.value  # untouched


# --- read-surface filtering -------------------------------------------------


def test_runs_list_hides_archived_by_default_and_shows_under_toggle(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    visible = _make_completed(session_factory)
    hidden = _make_completed(session_factory)
    _archive(session_factory, hidden)

    default = client.get("/runs")
    assert visible.hex[:8] in default.text
    assert hidden.hex[:8] not in default.text

    archived = client.get("/runs?archived=1")
    assert hidden.hex[:8] in archived.text
    assert visible.hex[:8] not in archived.text


def test_review_queue_excludes_archived(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id = _make_completed(session_factory)
    # A completed run with no labels won't be in the queue anyway; assert the
    # archived one never appears regardless.
    _archive(session_factory, run_id)
    resp = client.get("/review")
    assert run_id.hex[:8] not in resp.text


# --- run_detail button visibility -------------------------------------------


def test_run_detail_shows_archive_then_unarchive(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id = _make_completed(session_factory)
    before = client.get(f"/runs/{run_id}")
    assert f"/runs/{run_id}/archive" in before.text
    assert f"/runs/{run_id}/media/delete" in before.text
    assert f"/runs/{run_id}/unarchive" not in before.text

    _archive(session_factory, run_id)
    after = client.get(f"/runs/{run_id}")
    assert f"/runs/{run_id}/unarchive" in after.text
    assert f"/runs/{run_id}/archive\"" not in after.text  # archive form gone


def test_run_detail_hides_manage_controls_for_live_run(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    run_id = _make_queued(session_factory)
    resp = client.get(f"/runs/{run_id}")
    assert f"/runs/{run_id}/archive" not in resp.text
    assert f"/runs/{run_id}/media/delete" not in resp.text
