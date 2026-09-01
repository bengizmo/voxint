"""Bulk retry of grouped identical failures on the /runs Failed tab (#381 PR2)."""

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_BULK_RETRY, mint_csrf_token
from voxint.config import Settings
from voxint.db.models import MediaItem, PipelineRun, RunStatus

CREDS = ("reviewer", "s3cret")

CONNECT_ERROR = "ConnectionError: connect refused"
FILE_ERROR = "FileNotFoundError: /tmp/missing.wav"


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


def _make_failed_run(
    session: Session,
    *,
    error: str = CONNECT_ERROR,
    current_stage: str = "transcribe",
    revision: int = 0,
    archived: bool = False,
    status: RunStatus = RunStatus.FAILED,
) -> uuid.UUID:
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(
        media_item_id=media.id,
        status=status.value,
        error=error if status == RunStatus.FAILED else None,
        current_stage=current_stage,
        revision=revision,
    )
    if archived:
        run.archived_at = datetime.now(tz=UTC)
    session.add(run)
    session.flush()
    session.commit()
    return run.id


def _csrf(client: TestClient) -> str:
    return mint_csrf_token(client.app.state.csrf_secret, CSRF_BULK_RETRY)


def _post_bulk_retry(
    client: TestClient,
    items: list[str],
    *,
    csrf_token: str | None = None,
):
    form: dict[str, str | list[str]] = {}
    if items:
        form["item"] = items
    if csrf_token is not None:
        form["csrf_token"] = csrf_token
    return client.post("/runs/bulk-retry", data=form)


class TestBulkRetryGrouping:
    """GET /runs?view=failed: grouped rendering."""

    def test_failed_tab_groups_identical_errors(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as s:
            for _ in range(3):
                _make_failed_run(s, error=CONNECT_ERROR)
            _make_failed_run(s, error=FILE_ERROR)
        resp = client.get("/runs", params={"view": "failed"})
        assert resp.status_code == 200
        assert "Retry all 3" in resp.text
        assert resp.text.count('name="item"') >= 3

    def test_failed_tab_singleton_not_grouped(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as s:
            _make_failed_run(s, error=FILE_ERROR)
        resp = client.get("/runs", params={"view": "failed"})
        assert resp.status_code == 200
        assert "Retry all" not in resp.text
        assert "Retry" in resp.text

    def test_grouped_form_embeds_revision(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as s:
            id_a = _make_failed_run(s, error=CONNECT_ERROR, revision=3)
            id_b = _make_failed_run(s, error=CONNECT_ERROR, revision=7)
        resp = client.get("/runs", params={"view": "failed"})
        assert resp.status_code == 200
        assert f"{id_a}:3" in resp.text
        assert f"{id_b}:7" in resp.text


class TestBulkRetryPost:
    """POST /runs/bulk-retry: mutation with CAS safety."""

    def test_bulk_retry_requeues_group(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as s:
            id_a = _make_failed_run(s, error=CONNECT_ERROR)
            id_b = _make_failed_run(s, error=CONNECT_ERROR)
        resp = _post_bulk_retry(
            client,
            [f"{id_a}:0", f"{id_b}:0"],
            csrf_token=_csrf(client),
        )
        assert resp.status_code == 200
        assert "Requeued 2 run" in resp.text
        with session_factory() as s:
            for rid in (id_a, id_b):
                run = s.get(PipelineRun, rid)
                assert run is not None
                assert run.status == RunStatus.QUEUED.value

    def test_bulk_retry_csrf_required(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as s:
            rid = _make_failed_run(s)
        resp = _post_bulk_retry(client, [f"{rid}:0"])
        assert resp.status_code == 403

    def test_bulk_retry_stale_revision_skips(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as s:
            id_ok = _make_failed_run(s, error=CONNECT_ERROR)
            id_stale = _make_failed_run(s, error=CONNECT_ERROR)
        resp = _post_bulk_retry(
            client,
            [f"{id_ok}:0", f"{id_stale}:99"],
            csrf_token=_csrf(client),
        )
        assert resp.status_code == 200
        assert "Requeued 1" in resp.text
        assert "Skipped 1" in resp.text
        with session_factory() as s:
            assert s.get(PipelineRun, id_ok).status == RunStatus.QUEUED.value  # type: ignore[union-attr]
            assert s.get(PipelineRun, id_stale).status == RunStatus.FAILED.value  # type: ignore[union-attr]

    def test_bulk_retry_archived_run_skips(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as s:
            rid = _make_failed_run(s, archived=True)
        resp = _post_bulk_retry(
            client,
            [f"{rid}:0"],
            csrf_token=_csrf(client),
        )
        assert resp.status_code == 200
        assert "Skipped" in resp.text

    def test_bulk_retry_already_requeued_skips(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as s:
            rid = _make_failed_run(s, status=RunStatus.QUEUED)
        resp = _post_bulk_retry(
            client,
            [f"{rid}:0"],
            csrf_token=_csrf(client),
        )
        assert resp.status_code == 200
        assert "Skipped" in resp.text

    def test_bulk_retry_deduplication(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as s:
            rid = _make_failed_run(s, error=CONNECT_ERROR)
        resp = _post_bulk_retry(
            client,
            [f"{rid}:0", f"{rid}:0"],
            csrf_token=_csrf(client),
        )
        assert resp.status_code == 200
        assert "Requeued 1 run" in resp.text
        assert resp.text.count("Requeued") <= 2  # heading + one pill

    def test_bulk_retry_over_cap_rejected(
        self, client: TestClient
    ) -> None:
        items = [f"{uuid.uuid4()}:0" for _ in range(101)]
        resp = _post_bulk_retry(client, items, csrf_token=_csrf(client))
        assert resp.status_code == 400

    def test_bulk_retry_empty_rejected(
        self, client: TestClient
    ) -> None:
        resp = _post_bulk_retry(client, [], csrf_token=_csrf(client))
        assert resp.status_code == 400

    def test_bulk_retry_malformed_pair_rejected(
        self, client: TestClient
    ) -> None:
        resp = _post_bulk_retry(client, ["not-a-uuid"], csrf_token=_csrf(client))
        assert resp.status_code == 400

    def test_bulk_retry_all_skipped(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as s:
            rid = _make_failed_run(s, archived=True)
        resp = _post_bulk_retry(
            client,
            [f"{rid}:0"],
            csrf_token=_csrf(client),
        )
        assert resp.status_code == 200
        assert "No runs were requeued" in resp.text

    def test_bulk_retry_partial_success(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as s:
            id_ok = _make_failed_run(s, error=CONNECT_ERROR)
            id_bad = _make_failed_run(s, error=CONNECT_ERROR, archived=True)
        resp = _post_bulk_retry(
            client,
            [f"{id_ok}:0", f"{id_bad}:0"],
            csrf_token=_csrf(client),
        )
        assert resp.status_code == 200
        assert "Requeued 1" in resp.text
        assert "Skipped 1" in resp.text
