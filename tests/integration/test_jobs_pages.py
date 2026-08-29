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
    assert "pipeline-board" in resp.text


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
    # V3: run id is in the command bar breadcrumb, not an <h1>.
    assert f"<strong>{run_id.hex[:8]}</strong>" in resp.text
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


def test_running_run_shows_elapsed_duration(
    client_flag_off: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """#244 Delta 2: running runs show elapsed time, not '...'."""
    _make_run(session_factory, status=RunStatus.RUNNING, current_stage=Stage.TRANSCRIBE.value)
    resp = client_flag_off.get("/jobs")
    assert resp.status_code == 200
    import re
    assert re.search(r"\d+[smh]", resp.text), "expected a duration token in the page"


def test_failed_run_shows_humanized_error(
    client_flag_off: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """#244 Delta 7: failed runs show plain-language error, not raw exception."""
    with session_factory() as session:
        media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
        session.add(media)
        session.flush()
        run = PipelineRun(
            media_item_id=media.id,
            status=RunStatus.FAILED.value,
            error="interrupted: lease expired",
        )
        session.add(run)
        session.commit()
    resp = client_flag_off.get("/jobs")
    assert resp.status_code == 200
    assert "worker timed out" in resp.text
    assert "interrupted: lease expired" not in resp.text


def test_pipeline_summary_includes_gpu_busy(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """#244 Delta 8: GPU busy appears in the pipeline summary."""
    from voxint.api.resource_status import GpuActivity, ResourceStripView
    from voxint.api.routers.jobs import _pipeline_summary

    strip = ResourceStripView(
        telemetry_present=True,
        gpus=(GpuActivity(
            gpu_uuid="GPU-abc", short_uuid="abc",
            state="busy", utilization_percent=95, services=("transcription",),
        ),),
        warnings=(),
        unavailable_services=(),
        collected_age_seconds=1.0,
    )
    summary = _pipeline_summary({"running": 1}, 0, resource_strip=strip)
    assert "GPU busy" in summary


def test_pipeline_summary_no_gpu_when_idle(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """GPU busy is omitted when GPUs are idle."""
    from voxint.api.resource_status import GpuActivity, ResourceStripView
    from voxint.api.routers.jobs import _pipeline_summary

    strip = ResourceStripView(
        telemetry_present=True,
        gpus=(GpuActivity(
            gpu_uuid="GPU-abc", short_uuid="abc",
            state="idle", utilization_percent=5, services=("transcription",),
        ),),
        warnings=(),
        unavailable_services=(),
        collected_age_seconds=1.0,
    )
    summary = _pipeline_summary({"running": 1}, 0, resource_strip=strip)
    assert "GPU busy" not in summary


def test_collapse_stages_with_degraded(
    session_factory: sessionmaker[Session],
) -> None:
    """#244 Delta 4: degraded services mark their display stages."""
    from voxint.api.jobs_query import StageActivity
    from voxint.api.routers.jobs import _collapse_stages

    raw = [StageActivity(stage="transcribe", queued=2, active=0)]
    result = _collapse_stages(raw, degraded_stages=frozenset({"transcribe"}))
    transcribe = next(s for s in result if s.key == "transcribe")
    assert transcribe.is_degraded is True
    assert transcribe.queued == 2

    not_degraded = [s for s in result if s.key != "transcribe"]
    assert all(not s.is_degraded for s in not_degraded)


def test_collapse_stages_with_wait_estimate(
    session_factory: sessionmaker[Session],
) -> None:
    """#244 Delta 6: queued stages get a wait estimate from averages."""
    from voxint.api.jobs_query import StageActivity
    from voxint.api.routers.jobs import _collapse_stages

    raw = [StageActivity(stage="transcribe", queued=3, active=1)]
    result = _collapse_stages(
        raw, stage_avg_seconds={"transcribe": 120.0}
    )
    transcribe = next(s for s in result if s.key == "transcribe")
    assert transcribe.wait_estimate_seconds == pytest.approx(360.0)

    enrich = next(s for s in result if s.key == "enrich")
    assert enrich.wait_estimate_seconds is None
