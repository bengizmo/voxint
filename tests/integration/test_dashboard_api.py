"""The operator dashboard (issue #13) end to end against real Postgres.

The route is a thin HTML view over ``stats_query.collect_stats`` (already
covered in ``tests/unit/test_stats_query.py``); these tests pin the wiring the
unit tests cannot see — auth, that the rendered page and the htmx fragment carry
the aggregated numbers, the ``?since=`` throughput window, and that a malformed
``?since=`` degrades to the 24h default instead of 500-ing.
"""

import re
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.config import Settings
from voxint.db.models import (
    MediaItem,
    PipelineRun,
    RunStatus,
    Speaker,
    Stage,
    StageRun,
    StageStatus,
)

CREDS = ("reviewer", "s3cret")


@pytest.fixture()
def client(session_factory: sessionmaker[Session], tmp_path: Path) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path,
    )
    test_client = TestClient(create_app(settings=settings, session_factory=session_factory))
    test_client.auth = CREDS
    seed_onboarded(session_factory)
    return test_client


def make_run(
    session: Session,
    *,
    status: RunStatus,
    created_at: datetime | None = None,
    stages: Iterable[dict[str, object]] = (),
) -> uuid.UUID:
    """Seed one media item + run, optionally with StageRun attempts."""
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=status.value)
    if created_at is not None:
        run.created_at = created_at
    session.add(run)
    session.flush()
    for spec in stages:
        session.add(StageRun(pipeline_run_id=run.id, **spec))
    session.commit()
    return run.id


def seed_snapshot(session_factory: sessionmaker[Session]) -> None:
    """A fixed spread of runs/stages/speakers for the dashboard to aggregate."""
    now = datetime.now(UTC)
    old = now - timedelta(days=3)
    with session_factory() as session:
        # Two awaiting adjudication -> review backlog == 2.
        make_run(session, status=RunStatus.AWAITING_ADJUDICATION)
        make_run(session, status=RunStatus.AWAITING_ADJUDICATION)
        # One completed run with a finished transcribe (30s) and a failed diarize.
        make_run(
            session,
            status=RunStatus.COMPLETED,
            stages=[
                {
                    "stage": Stage.TRANSCRIBE.value,
                    "status": StageStatus.COMPLETED.value,
                    "started_at": now - timedelta(seconds=30),
                    "finished_at": now,
                },
                {
                    "stage": Stage.DIARIZE_EMBED.value,
                    "status": StageStatus.FAILED.value,
                    "started_at": now - timedelta(seconds=5),
                    "finished_at": now,
                },
            ],
        )
        # One old run outside a short window (for the ?since= narrowing test).
        make_run(session, status=RunStatus.FAILED, created_at=old)
        # Two enrolled speakers -> roster size == 2.
        session.add(Speaker(display_name="Alice"))
        session.add(Speaker(display_name="Bob"))
        session.commit()


def test_dashboard_renders_aggregated_numbers(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    seed_snapshot(session_factory)
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    body = resp.text
    # Full page: nav chrome present, active tab marked.
    assert 'href="/dashboard" aria-current="page"' in body
    # Review backlog (2 awaiting_adjudication) and roster (2 speakers) surface.
    assert "2 run(s) awaiting adjudication" in body
    assert "2 enrolled speaker(s)" in body
    # Status table zero-fills the enum: a status with no runs renders a 0 row.
    assert re.search(r"queued</span></td>\s*<td>0</td>", body)
    # Stage timing binds the seeded values, not just the stage names: transcribe
    # ran 30s, the failed diarize_embed attempt 5s (finished attempts count for
    # duration regardless of terminal status).
    assert "transcribe" in body
    assert "30.00s" in body
    assert "5.00s" in body
    # The one seeded diarize_embed failure renders in the failures table.
    assert re.search(r"diarize_embed</td>\s*<td>1</td>", body)


def test_dashboard_htmx_returns_fragment_without_chrome(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    seed_snapshot(session_factory)
    resp = client.get("/dashboard", headers={"HX-Request": "true"})
    assert resp.status_code == 200
    body = resp.text
    # Fragment carries the numbers but not the page chrome / nav.
    assert "2 run(s) awaiting adjudication" in body
    assert "<nav" not in body
    assert "<h1>Dashboard</h1>" not in body
    # The fragment must NOT re-emit the polling container or its hx-get: the
    # outer #dashboard-metrics div lives in the full page and persists across
    # innerHTML swaps, so a nested one would duplicate the id and the 15s timer.
    assert "hx-get" not in body
    assert 'id="dashboard-metrics"' not in body


def test_dashboard_since_narrows_created_window(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    seed_snapshot(session_factory)
    # Default (24h) excludes the 3-day-old run: 3 recent runs created.
    default_body = client.get("/dashboard").text
    assert "Runs created in window" in default_body
    # A 7-day window includes the old run too (all 4).
    wide_body = client.get("/dashboard", params={"since": "7d"}).text
    # The window line differs; assert the wide window reports one more run than
    # the default by checking both counts render distinctly.
    assert _created_count(default_body) == 3
    assert _created_count(wide_body) == 4
    # The 15s poll URL carries the active window so it survives the refresh; the
    # default page polls the bare route.
    assert 'hx-get="/dashboard?since=7d"' in wide_body
    assert 'hx-get="/dashboard"' in default_body


def test_dashboard_malformed_since_shows_notice_and_drops_poll_param(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    seed_snapshot(session_factory)
    body = client.get("/dashboard", params={"since": "not-a-window"}).text
    # Operator is told the window was ignored, and the rejected value is not
    # echoed into the poll URL (each poll would just re-take the fallback).
    assert "Unrecognized" in body
    assert 'hx-get="/dashboard"' in body
    assert "not-a-window" not in body


def test_dashboard_matches_metrics_snapshot(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """The core invariant: dashboard numbers agree with /metrics on one seed."""
    seed_snapshot(session_factory)
    dash = client.get("/dashboard").text
    metrics = client.get("/metrics").text
    # runs_created (default 24h window) — same figure on both surfaces.
    assert _created_count(dash) == 3
    assert "voxint_runs_created_24h 3" in metrics
    # roster + backlog line up with their Prometheus gauges.
    assert "voxint_roster_speakers 2" in metrics
    assert "2 enrolled speaker(s)" in dash
    assert 'voxint_runs{status="awaiting_adjudication"} 2' in metrics
    assert "2 run(s) awaiting adjudication" in dash


def test_dashboard_malformed_since_degrades_to_default(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    seed_snapshot(session_factory)
    resp = client.get("/dashboard", params={"since": "not-a-window"})
    assert resp.status_code == 200
    # Falls back to the 24h default (3 recent runs), not a 500 or the wide count.
    assert _created_count(resp.text) == 3


def _created_count(body: str) -> int:
    """Pull the runs-created count out of the rendered HTML via its stable hook."""
    match = re.search(r'data-metric="runs-created"[^>]*>\s*(\d+)', body)
    assert match is not None, "runs-created metric hook missing from dashboard render"
    return int(match.group(1))
