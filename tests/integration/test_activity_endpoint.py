"""The /activity/events poll endpoint + shell chrome, end to end (#162).

Pins the dark-ship wiring: operator auth runs before the flag-off 404, the
bootstrap and since responses have the documented shape, the badge equals the
shared jobs_badge_count the /jobs page renders, and the shell mounts the toast
region + badge + poller only when activity AND Jobs discovery are both on.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.activity import record_activity_event
from voxint.api.app import create_app
from voxint.config import Settings
from voxint.db.models import ActivityKind, MediaItem, PipelineRun, RunStatus

CREDS = ("reviewer", "s3cret")


def _client(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
    *,
    activity_enabled: bool = False,
    jobs_enabled: bool = False,
    authenticated: bool = True,
) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path,
        console_activity_enabled=activity_enabled,
        console_jobs_enabled=jobs_enabled,
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    if authenticated:
        client.auth = CREDS
    seed_onboarded(session_factory)
    return client


def _seed_run(session_factory: sessionmaker[Session], *, status: RunStatus) -> uuid.UUID:
    with session_factory() as session:
        media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
        session.add(media)
        session.flush()
        run = PipelineRun(media_item_id=media.id, status=status.value)
        session.add(run)
        session.commit()
        return run.id


def _seed_events(session_factory: sessionmaker[Session], run_id: uuid.UUID, n: int) -> None:
    with session_factory() as session:
        for i in range(n):
            record_activity_event(
                session,
                kind=ActivityKind.RUN_COMPLETED,
                occurrence_key=f"k-{run_id}-{i}",
                pipeline_run_id=run_id,
                title=f"clip {i}",
                href=f"/jobs/{run_id}",
            )
        session.commit()


def test_unauthenticated_is_401_even_when_disabled(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    # Auth runs before the flag-off 404 (the invariant: everything but /healthz
    # authenticates first).
    client = _client(session_factory, tmp_path, activity_enabled=False, authenticated=False)
    assert client.get("/activity/events").status_code == 401


def test_flag_off_returns_404(session_factory: sessionmaker[Session], tmp_path: Path) -> None:
    client = _client(session_factory, tmp_path, activity_enabled=False)
    assert client.get("/activity/events").status_code == 404


def test_bootstrap_baselines_without_backlog(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _client(session_factory, tmp_path, activity_enabled=True, jobs_enabled=True)
    rid = _seed_run(session_factory, status=RunStatus.COMPLETED)
    _seed_events(session_factory, rid, 3)
    resp = client.get("/activity/events")  # no since => bootstrap
    assert resp.status_code == 200
    body = resp.json()
    assert body["events"] == []
    assert body["has_more"] is False
    assert body["next_cursor"] == body["high_water"] > 0


def test_since_returns_new_events_ascending(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _client(session_factory, tmp_path, activity_enabled=True, jobs_enabled=True)
    rid = _seed_run(session_factory, status=RunStatus.COMPLETED)
    _seed_events(session_factory, rid, 3)
    body = client.get("/activity/events?since=0").json()
    ids = [e["id"] for e in body["events"]]
    assert ids == sorted(ids)
    assert len(ids) == 3
    assert body["next_cursor"] == ids[-1]
    assert body["has_more"] is False
    assert body["events"][0]["kind"] == "run_completed"
    assert body["events"][0]["href"] == f"/jobs/{rid}"


def test_badge_matches_jobs_page(session_factory: sessionmaker[Session], tmp_path: Path) -> None:
    client = _client(session_factory, tmp_path, activity_enabled=True, jobs_enabled=True)
    _seed_run(session_factory, status=RunStatus.QUEUED)
    _seed_run(session_factory, status=RunStatus.QUEUED)
    _seed_run(session_factory, status=RunStatus.RUNNING)
    _seed_run(session_factory, status=RunStatus.COMPLETED)  # terminal, not counted

    badge = client.get("/activity/events").json()["badge"]
    assert badge == 3
    jobs_page = client.get("/jobs").text
    assert "3 jobs in progress" in jobs_page


def test_shell_chrome_absent_when_off(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _client(session_factory, tmp_path, activity_enabled=False, jobs_enabled=True)
    home = client.get("/").text
    assert "data-toast-region" not in home
    assert "data-activity-badge" not in home
    assert "/activity/events" not in home


def test_shell_chrome_present_when_activity_and_jobs_on(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _client(session_factory, tmp_path, activity_enabled=True, jobs_enabled=True)
    home = client.get("/").text
    assert "data-toast-region" in home
    assert "data-activity-badge" in home
    assert "/activity/events" in home


def test_shell_chrome_absent_when_jobs_discovery_off(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    # Nav-honesty guard: activity depends on Jobs discovery. With jobs off the
    # sidebar Jobs entry points at /runs, so the badge/toasts must stay hidden.
    client = _client(session_factory, tmp_path, activity_enabled=True, jobs_enabled=False)
    home = client.get("/").text
    assert "data-toast-region" not in home
    assert "data-activity-badge" not in home


def test_activity_flag_defaults_off() -> None:
    assert Settings(_env_file=None).console_activity_enabled is False
