"""Live-row fragment transitions and bounded polling (#496)."""

import re
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.test_runs_canonical import _make_run
from tests.integration.test_runs_canonical import client as client
from voxint.api.csrf import CSRF_REQUEUE, verify_csrf_token
from voxint.api.routers.legacy_runs import RUNS_LIVE_ROWS_MAX
from voxint.db.models import PipelineRun, RunStatus, StageRun


def _row(body: str, run_id: uuid.UUID) -> str:
    """The rendered row for ``run_id``: its opening tag through to the next row or the end."""
    start = body.rindex("<div ", 0, body.index(f'id="run-row-{run_id}"'))
    end = body.find('<div class="gt-row', start + 1)
    return body[start:] if end == -1 else body[start:end]


def test_running_stage_and_elapsed(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        run_id = _make_run(session, status=RunStatus.RUNNING)
        run = session.get(PipelineRun, run_id)
        assert run is not None
        run.current_stage = "transcribe"
        session.add(
            StageRun(
                pipeline_run_id=run_id,
                stage="transcribe",
                status="running",
                attempt=1,
                started_at=datetime.now(UTC) - timedelta(seconds=65),
                lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
            )
        )
        session.commit()
    response = client.get(f"/runs/live-rows?ids={run_id}")
    body = response.text
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert 'hx-trigger="every 5s"' in body
    poller = body[: body.index('<div class="gt-row')]
    assert 'id="runs-live"' in poller
    assert "hx-swap-oob" not in poller
    assert 'hx-swap-oob="true"' in _row(body, run_id)
    assert "pill running" in body
    assert "Transcrib" in body
    assert "Processing so far, as of last refresh" in body
    assert "1m" in body
    assert "is-just-finished" not in body


def test_mixed_completion(client: TestClient, session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        done = _make_run(session, labels=("A",))
        live = _make_run(session, status=RunStatus.RUNNING)
    body = client.get(f"/runs/live-rows?ids={done},{live}").text
    assert f'hx-get="/runs/live-rows?ids={live}"' in body
    assert body.count('hx-swap-oob="true"') == 2
    done_row, live_row = _row(body, done), _row(body, live)
    assert "is-just-finished" in done_row
    assert "needs review" in done_row.lower()
    assert 'href="/review"' in done_row
    assert "is-just-finished" not in live_row
    assert 'class="chip chip-info ">Running</span>' in live_row


@pytest.mark.parametrize(
    "status",
    [RunStatus.FAILED, RunStatus.CANCELLED, RunStatus.PAUSED, RunStatus.AWAITING_ADJUDICATION],
)
def test_non_live_stops(
    client: TestClient, session_factory: sessionmaker[Session], status: RunStatus
) -> None:
    with session_factory() as session:
        run_id = _make_run(session, status=status)
        run = session.get(PipelineRun, run_id)
        assert run is not None
        run.revision = 7
        session.commit()
    body = client.get(f"/runs/live-rows?ids={run_id}").text
    assert '<div id="runs-live" hidden></div>' in body
    assert "hx-trigger" not in body
    assert f'id="run-row-{run_id}"' in body
    assert ("is-just-finished" in body) == (status in (RunStatus.FAILED, RunStatus.CANCELLED))
    if status == RunStatus.FAILED:
        assert f'action="/runs/{run_id}/requeue"' in body
        assert 'name="revision" value="7"' in body
        token = re.search(r'name="csrf_token" value="([^"]+)"', body)
        assert token is not None
        assert verify_csrf_token(client.app.state.csrf_secret, CSRF_REQUEUE, token[1])
    if status == RunStatus.CANCELLED:
        assert '<span class="oc-muted">—</span>' in body
        assert "Processing time" not in body
        assert "Processing so far" not in body


def test_running_to_queued_adapts_interval(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        run_id = _make_run(session, status=RunStatus.RUNNING)
    assert "every 5s" in client.get(f"/runs/live-rows?ids={run_id}").text
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        run.status = RunStatus.QUEUED.value
        session.commit()
    assert "every 15s" in client.get(f"/runs/live-rows?ids={run_id}").text


def test_missing_archived_and_duplicate_ids(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        archived = _make_run(session)
        run = session.get(PipelineRun, archived)
        assert run is not None
        run.archived_at = datetime.now(UTC)
        session.commit()
        live = _make_run(session, status=RunStatus.QUEUED)
    missing = uuid.uuid4()
    body = client.get(f"/runs/live-rows?ids={archived},{missing},{live},{live}").text
    assert f'id="run-row-{archived}"' not in body
    assert f'id="run-row-{missing}"' not in body
    assert body.count(f'id="run-row-{live}"') == 1
    assert f'hx-get="/runs/live-rows?ids={live}"' in body


@pytest.mark.parametrize(
    "ids",
    [
        "bad",
        str(uuid.UUID(int=1)).replace("-", ""),
        ",".join(str(uuid.UUID(int=i)) for i in range(RUNS_LIVE_ROWS_MAX + 1)),
    ],
)
def test_invalid_ids(client: TestClient, ids: str) -> None:
    assert client.get("/runs/live-rows", params={"ids": ids}).status_code == 400


@pytest.mark.parametrize("suffix", ["", "?ids="])
def test_empty_stops_without_row_queries(client: TestClient, suffix: str) -> None:
    with (
        patch("voxint.api.routers.legacy_runs.list_runs") as listing,
        patch("voxint.api.routers.legacy_runs.pipeline_dashboard_state") as dashboard,
    ):
        response = client.get(f"/runs/live-rows{suffix}")
    listing.assert_not_called()
    dashboard.assert_not_called()
    assert response.status_code == 200
    assert response.text.strip() == '<div id="runs-live" hidden></div>'


def test_unauthenticated_matches_strip(client: TestClient) -> None:
    assert (
        client.get("/runs/live-rows", auth=("bad", "bad")).status_code
        == client.get("/runs/progress-strip", auth=("bad", "bad")).status_code
        == 401
    )
