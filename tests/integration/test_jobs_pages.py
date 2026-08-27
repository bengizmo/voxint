"""The Jobs area pages end to end against real Postgres (Console 2.0 P5, #160).

Pins the dark-ship wiring the unit/query tests cannot see: both pages are
reachable directly regardless of the flag (no area gate), the recent-runs table
and stage strip render, the sidebar Jobs entry follows ``console_jobs_enabled``
(off keeps it on the ``/runs`` placeholder, on repoints it at ``/jobs`` with
``aria-current``), and ``/jobs/{id}`` renders the shared run-detail sections.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.config import Settings
from voxint.db.models import MediaItem, PipelineRun, RunStatus, Stage

CREDS = ("reviewer", "s3cret")


def _client(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
    *,
    jobs_enabled: bool = False,
) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path,
        console_jobs_enabled=jobs_enabled,
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory)
    return client


def _make_run(
    session_factory: sessionmaker[Session],
    *,
    status: RunStatus,
    current_stage: str | None = None,
) -> uuid.UUID:
    with session_factory() as session:
        media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
        session.add(media)
        session.flush()
        run = PipelineRun(
            media_item_id=media.id, status=status.value, current_stage=current_stage
        )
        session.add(run)
        session.commit()
        return run.id


@pytest.fixture()
def client_flag_off(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> TestClient:
    return _client(session_factory, tmp_path, jobs_enabled=False)


@pytest.fixture()
def client_flag_on(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> TestClient:
    return _client(session_factory, tmp_path, jobs_enabled=True)


def test_jobs_page_reachable_directly_with_flag_off(
    client_flag_off: TestClient,
) -> None:
    """Dark-ship: /jobs is registered and reachable directly even when the flag
    is off — the flag is rollout control, not authorization."""
    resp = client_flag_off.get("/jobs")
    assert resp.status_code == 200
    assert "Pipeline activity" in resp.text


def test_jobs_page_requires_auth(session_factory: sessionmaker[Session], tmp_path: Path) -> None:
    """Like every console page, /jobs is behind operator auth."""
    settings = Settings(
        voxint_user=CREDS[0], voxint_password=CREDS[1], media_root=tmp_path
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    seed_onboarded(session_factory)
    assert client.get("/jobs").status_code == 401


def test_jobs_page_lists_recent_runs(
    client_flag_off: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """The recent-runs table links each run to its /jobs/{id} detail page."""
    run_id = _make_run(session_factory, status=RunStatus.COMPLETED)
    resp = client_flag_off.get("/jobs")
    assert resp.status_code == 200
    assert f"/jobs/{run_id}" in resp.text


def test_jobs_strip_reconciles_with_status_counts(
    client_flag_off: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """The stage strip totals equal the queued/running status counts the page
    headline (and voxint stats) report — the acceptance criterion for #160."""
    from voxint.api.jobs_query import stage_activity
    from voxint.api.stats_query import run_status_counts

    _make_run(session_factory, status=RunStatus.QUEUED, current_stage=Stage.TRANSCRIBE.value)
    _make_run(session_factory, status=RunStatus.QUEUED, current_stage=None)

    assert client_flag_off.get("/jobs").status_code == 200
    with session_factory() as session:
        activity = stage_activity(session)
        counts = run_status_counts(session)
    assert sum(a.queued for a in activity) == counts.get(RunStatus.QUEUED.value, 0) == 2


def test_job_detail_renders_run_sections(
    client_flag_off: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """/jobs/{id} absorbs the run-detail page; its forms post to the legacy
    /runs/{id}/... action endpoints (reused, not aliased)."""
    run_id = _make_run(session_factory, status=RunStatus.FAILED)
    resp = client_flag_off.get(f"/jobs/{run_id}")
    assert resp.status_code == 200
    assert f"Run <code>{run_id.hex[:8]}</code>" in resp.text
    # The requeue form (rendered for a FAILED run) posts to the legacy endpoint.
    assert f'action="/runs/{run_id}/requeue"' in resp.text


def test_job_detail_unknown_run_is_404(client_flag_off: TestClient) -> None:
    assert client_flag_off.get(f"/jobs/{uuid.uuid4()}").status_code == 404


def test_job_detail_suppresses_the_tutorial_banner(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """The one intentional behavioral delta of the extraction: /runs/{id} shows
    the guided-tour banner in tutorial mode, /jobs/{id} suppresses it (the page
    is dark-shipped and not in the tutorial's route map)."""
    from voxint.tutorial.seed import seed_tutorial_run

    settings = Settings(
        voxint_user=CREDS[0], voxint_password=CREDS[1], media_root=tmp_path
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory)
    with session_factory() as session:
        run_id = seed_tutorial_run(
            session, media_root=settings.media_root, settings=settings
        )
        session.commit()

    banner = 'aria-label="Guided tutorial"'
    on_runs = client.get(f"/runs/{run_id}?tutorial=run")
    assert on_runs.status_code == 200
    assert banner in on_runs.text
    on_jobs = client.get(f"/jobs/{run_id}?tutorial=run")
    assert on_jobs.status_code == 200
    assert banner not in on_jobs.text


def test_sidebar_jobs_entry_points_at_runs_when_flag_off(
    client_flag_off: TestClient,
) -> None:
    """With the flag off (shipped default) the icon rail Jobs entry still points
    at the /runs placeholder, and no /jobs nav link."""
    home = client_flag_off.get("/")
    assert home.status_code == 200
    assert 'href="/runs"' in home.text
    assert 'aria-label="Jobs"' in home.text
    assert '<a href="/jobs"' not in home.text


def test_sidebar_jobs_entry_points_at_jobs_when_flag_on(
    client_flag_on: TestClient,
) -> None:
    """With the flag on, the icon rail Jobs entry repoints at /jobs; the entry
    carries aria-current on the Jobs page itself."""
    home = client_flag_on.get("/")
    assert home.status_code == 200
    assert 'href="/jobs"' in home.text
    assert 'aria-label="Jobs"' in home.text
    # On /jobs, the entry is the current page.
    jobs = client_flag_on.get("/jobs")
    assert 'href="/jobs"' in jobs.text
    assert 'aria-current="page"' in jobs.text
