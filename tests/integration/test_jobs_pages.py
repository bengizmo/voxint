"""Jobs compatibility redirects (#382, S13).

After the canonical /runs surface replaced the Jobs pages, /jobs and
/jobs/{run_id} redirect with 303 so old bookmarks and links keep working.
The query-vocabulary mapping preserves intent: /jobs?filter=failed lands on
/runs?view=failed, not just /runs.

Tests that exercised the old page content (pipeline board, run table, detail
sections) are removed: the canonical /runs surface carries those now. Helpers
imported from jobs.py (_detect_degraded, _pipeline_summary) are still tested
via the /runs page and their unit tests here.
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
from voxint.db.models import MediaItem, PipelineRun, RunStatus

CREDS = ("reviewer", "s3cret")


def _client(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path,
    )
    client = TestClient(
        create_app(settings=settings, session_factory=session_factory),
        follow_redirects=False,
    )
    client.auth = CREDS
    seed_onboarded(session_factory)
    return client


def _make_run(
    session_factory: sessionmaker[Session],
    *,
    status: RunStatus,
) -> uuid.UUID:
    with session_factory() as session:
        media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
        session.add(media)
        session.flush()
        run = PipelineRun(
            media_item_id=media.id, status=status.value,
        )
        session.add(run)
        session.commit()
        return run.id


@pytest.fixture()
def client(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> TestClient:
    return _client(session_factory, tmp_path)


# ---- /jobs redirect tests ----


def test_jobs_redirects_to_runs(client: TestClient) -> None:
    """GET /jobs → 303 /runs."""
    resp = client.get("/jobs")
    assert resp.status_code == 303
    assert resp.headers["location"] == "/runs"


def test_jobs_filter_all_redirects_to_runs(client: TestClient) -> None:
    """/jobs?filter=all → /runs (the 'all' filter is the default)."""
    resp = client.get("/jobs?filter=all")
    assert resp.status_code == 303
    assert resp.headers["location"] == "/runs"


def test_jobs_filter_active_maps_to_view(client: TestClient) -> None:
    """/jobs?filter=active → /runs?view=active."""
    resp = client.get("/jobs?filter=active")
    assert resp.status_code == 303
    assert resp.headers["location"] == "/runs?view=active"


def test_jobs_filter_needs_review_maps_to_needs_attention(client: TestClient) -> None:
    """/jobs?filter=needs_review → /runs?view=needs_attention."""
    resp = client.get("/jobs?filter=needs_review")
    assert resp.status_code == 303
    assert resp.headers["location"] == "/runs?view=needs_attention"


def test_jobs_filter_failed_maps_to_view(client: TestClient) -> None:
    """/jobs?filter=failed → /runs?view=failed."""
    resp = client.get("/jobs?filter=failed")
    assert resp.status_code == 303
    assert resp.headers["location"] == "/runs?view=failed"


def test_jobs_unknown_filter_falls_through(client: TestClient) -> None:
    """An unknown filter value redirects to /runs with no query params."""
    resp = client.get("/jobs?filter=bogus")
    assert resp.status_code == 303
    assert resp.headers["location"] == "/runs"


def test_jobs_requires_auth(session_factory: sessionmaker[Session], tmp_path: Path) -> None:
    """Like every console page, /jobs is behind operator auth."""
    settings = Settings(
        voxint_user=CREDS[0], voxint_password=CREDS[1], media_root=tmp_path
    )
    client = TestClient(
        create_app(settings=settings, session_factory=session_factory),
        follow_redirects=False,
    )
    seed_onboarded(session_factory)
    assert client.get("/jobs").status_code == 401


# ---- /jobs/{run_id} redirect tests ----


def test_job_detail_redirects_to_run_detail(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """GET /jobs/{run_id} → 303 /runs/{run_id}."""
    run_id = _make_run(session_factory, status=RunStatus.COMPLETED)
    resp = client.get(f"/jobs/{run_id}")
    assert resp.status_code == 303
    assert resp.headers["location"] == f"/runs/{run_id}"


def test_job_detail_invalid_uuid_is_422(client: TestClient) -> None:
    """A non-UUID path segment returns 422 (FastAPI UUID validation)."""
    resp = client.get("/jobs/not-a-uuid")
    assert resp.status_code == 422


# ---- sidebar nav tests ----


def test_sidebar_runs_entry_always_points_at_runs(client: TestClient) -> None:
    """The sidebar Runs entry always points at /runs (no flag branch)."""
    home = client.get("/", follow_redirects=True)
    assert home.status_code == 200
    assert 'href="/runs"' in home.text
    assert 'aria-label="Runs"' in home.text


# ---- follow-through redirect tests ----


@pytest.mark.parametrize(
    "filter_param,expected_view",
    [
        ("active", "active"),
        ("needs_review", "needs_attention"),
        ("failed", "failed"),
    ],
)
def test_jobs_filter_redirects_render_200(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
    filter_param: str,
    expected_view: str,
) -> None:
    """Each /jobs?filter=X redirect lands on a valid /runs?view=Y page (200)."""
    settings = Settings(
        voxint_user=CREDS[0], voxint_password=CREDS[1], media_root=tmp_path
    )
    follow_client = TestClient(
        create_app(settings=settings, session_factory=session_factory),
        follow_redirects=True,
    )
    follow_client.auth = CREDS
    seed_onboarded(session_factory)
    resp = follow_client.get(f"/jobs?filter={filter_param}")
    assert resp.status_code == 200


# ---- helper unit tests (kept from the old file) ----


def test_pipeline_summary_includes_gpu_busy() -> None:
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
    summary = _pipeline_summary({"running": 1}, resource_strip=strip)
    assert "GPU busy" in summary


def test_pipeline_summary_no_gpu_when_idle() -> None:
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
    summary = _pipeline_summary({"running": 1}, resource_strip=strip)
    assert "GPU busy" not in summary


def test_legacy_transcript_crumb_uses_runs(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """The transcript page always uses 'runs' in the breadcrumb."""
    run_id = _make_run(session_factory, status=RunStatus.COMPLETED)
    resp = client.get(f"/runs/{run_id}/transcript", follow_redirects=True)
    assert resp.status_code == 200
    assert f'class="cb-breadcrumb">runs / <strong>{run_id.hex[:8]}</strong>' in resp.text
