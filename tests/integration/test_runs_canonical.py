"""The /runs canonical lifecycle surface (S12, #381).

Covers lifecycle tabs (view parameter), collapsed filter disclosure,
pipeline health summary, auxiliary jobs disclosure, title-first rows,
and row actions.
"""

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.config import Settings
from voxint.db.models import (
    EMBEDDING_DIM,
    DiarizationTurn,
    MediaItem,
    PipelineRun,
    RunStatus,
)

CREDS = ("reviewer", "s3cret")
SPACE = "titanet-large-v1"


def unit(dim: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    vector[dim % EMBEDDING_DIM] = 1.0
    return vector


@pytest.fixture()
def client(session_factory: sessionmaker[Session], tmp_path: Path) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path,
        runs_page_size=50,
    )
    test_client = TestClient(create_app(settings=settings, session_factory=session_factory))
    test_client.auth = CREDS
    seed_onboarded(session_factory)
    return test_client


def _make_run(
    session: Session,
    *,
    status: RunStatus = RunStatus.COMPLETED,
    labels: tuple[str, ...] = (),
) -> uuid.UUID:
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(
        media_item_id=media.id,
        status=status.value,
    )
    session.add(run)
    session.flush()
    for index, label in enumerate(labels):
        session.add(
            DiarizationTurn(
                pipeline_run_id=run.id,
                turn_index=index,
                start_seconds=float(index * 10),
                end_seconds=float(index * 10 + 8),
                label=label,
                embedding=unit(index),
                embedding_space=SPACE,
            )
        )
    session.commit()
    return run.id


class TestLifecycleTabs:
    """The view parameter selects lifecycle tabs."""

    def test_default_view_shows_all_runs(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as s:
            _make_run(s, status=RunStatus.COMPLETED)
            _make_run(s, status=RunStatus.FAILED)
            _make_run(s, status=RunStatus.RUNNING)
        resp = client.get("/runs")
        assert resp.status_code == 200
        assert resp.text.count("gt-row") >= 3

    def test_needs_attention_shows_awaiting_adjudication(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as s:
            run_id = _make_run(
                s, status=RunStatus.AWAITING_ADJUDICATION, labels=("A",)
            )
            _make_run(s, status=RunStatus.RUNNING)
            _make_run(s, status=RunStatus.FAILED)
        resp = client.get("/runs?view=needs_attention")
        assert resp.status_code == 200
        assert run_id.hex[:8] in resp.text

    def test_needs_attention_shows_completed_with_unresolved(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as s:
            run_id = _make_run(s, status=RunStatus.COMPLETED, labels=("A",))
        resp = client.get("/runs?view=needs_attention")
        assert resp.status_code == 200
        assert run_id.hex[:8] in resp.text

    def test_needs_attention_excludes_completed_no_labels(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as s:
            run_id = _make_run(s, status=RunStatus.COMPLETED)
        resp = client.get("/runs?view=needs_attention")
        assert resp.status_code == 200
        assert run_id.hex[:8] not in resp.text

    def test_active_view_shows_queued_running_paused(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as s:
            q = _make_run(s, status=RunStatus.QUEUED)
            r = _make_run(s, status=RunStatus.RUNNING)
            p = _make_run(s, status=RunStatus.PAUSED)
            _make_run(s, status=RunStatus.COMPLETED)
            _make_run(s, status=RunStatus.FAILED)
        resp = client.get("/runs?view=active")
        assert resp.status_code == 200
        body = resp.text
        assert q.hex[:8] in body
        assert r.hex[:8] in body
        assert p.hex[:8] in body

    def test_failed_view(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as s:
            f = _make_run(s, status=RunStatus.FAILED)
            _make_run(s, status=RunStatus.COMPLETED)
        resp = client.get("/runs?view=failed")
        assert resp.status_code == 200
        assert f.hex[:8] in resp.text

    def test_invalid_view_returns_422(self, client: TestClient) -> None:
        resp = client.get("/runs?view=bogus")
        assert resp.status_code == 422


class TestTabRendering:
    """The template renders the correct tab as active."""

    def test_all_tab_active_by_default(self, client: TestClient) -> None:
        resp = client.get("/runs")
        assert 'aria-current="page">All' in resp.text

    def test_needs_attention_tab_active(self, client: TestClient) -> None:
        resp = client.get("/runs?view=needs_attention")
        assert 'aria-current="page">Needs attention' in resp.text

    def test_active_tab_active(self, client: TestClient) -> None:
        resp = client.get("/runs?view=active")
        assert 'aria-current="page">Active' in resp.text

    def test_failed_tab_active(self, client: TestClient) -> None:
        resp = client.get("/runs?view=failed")
        assert 'aria-current="page">Failed' in resp.text


class TestFilterDisclosure:
    """The filter bar is a collapsible disclosure."""

    def test_filter_closed_by_default(self, client: TestClient) -> None:
        resp = client.get("/runs")
        assert resp.status_code == 200
        assert "runs-filters" in resp.text
        assert "open" not in resp.text.split("runs-filters")[1].split(">")[0]

    def test_filter_opens_when_status_active(self, client: TestClient) -> None:
        resp = client.get("/runs?status=completed")
        assert resp.status_code == 200
        chunk = resp.text.split('class="runs-filters"')[1].split(">")[0]
        assert "open" in chunk

    def test_filter_preserves_view(self, client: TestClient) -> None:
        resp = client.get("/runs?view=active&status=running")
        assert resp.status_code == 200
        assert 'name="view" value="active"' in resp.text

    def test_tab_links_preserve_active_filters(self, client: TestClient) -> None:
        resp = client.get("/runs?view=active&status=running")
        assert resp.status_code == 200
        body = resp.text
        assert "view=failed" in body
        assert "status=running" in body

    def test_more_filters_auto_opens_when_active(self, client: TestClient) -> None:
        resp = client.get("/runs?q=hello")
        assert resp.status_code == 200
        assert 'runs-more-filters" open' in resp.text or \
               "runs-more-filters' open" in resp.text


class TestTitleFirstRows:
    """Rows lead with the media title, run ID is secondary."""

    def test_media_title_is_primary_link(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as s:
            run_id = _make_run(s, status=RunStatus.COMPLETED)
        resp = client.get("/runs")
        assert resp.status_code == 200
        assert f'href="/runs/{run_id}" class="media-title"' in resp.text

    def test_run_id_is_secondary(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as s:
            run_id = _make_run(s, status=RunStatus.COMPLETED)
        resp = client.get("/runs")
        assert resp.status_code == 200
        assert "run-id-secondary" in resp.text
        assert f"<code>{run_id.hex[:8]}</code>" in resp.text


class TestRowActions:
    """State-dependent action links."""

    def test_completed_unresolved_shows_review(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as s:
            _make_run(s, status=RunStatus.COMPLETED, labels=("A",))
        resp = client.get("/runs")
        assert "Review →" in resp.text

    def test_failed_shows_retry(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as s:
            _make_run(s, status=RunStatus.FAILED)
        resp = client.get("/runs")
        assert "Retry →" in resp.text

    def test_running_shows_view(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as s:
            _make_run(s, status=RunStatus.RUNNING)
        resp = client.get("/runs")
        assert "View →" in resp.text


class TestPipelineSummary:
    """The page shows a one-line pipeline health summary."""

    def test_pipeline_summary_present(self, client: TestClient) -> None:
        resp = client.get("/runs")
        assert resp.status_code == 200
        assert "Pipeline status" in resp.text

    def test_pipeline_summary_links_to_settings(self, client: TestClient) -> None:
        resp = client.get("/runs")
        assert "/settings/status" in resp.text


class TestColumnHeaders:
    """The grid table has the expected column headers."""

    def test_media_column_first(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as s:
            _make_run(s, status=RunStatus.COMPLETED)
        resp = client.get("/runs")
        assert resp.status_code == 200
        body = resp.text
        media_pos = body.find(">MEDIA<")
        status_pos = body.find(">STATUS<")
        assert media_pos != -1
        assert status_pos != -1
        assert media_pos < status_pos
