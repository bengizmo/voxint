"""GET /media/{media_id}/editor -- the media detail page (issue #156).

Tests the HTTP surface: status codes, gating, run selection, claim-token
handling, and the editor island mount point (#157).
"""

import uuid
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.adjudication.slots import claim_run
from voxint.api.app import create_app
from voxint.config import Settings
from voxint.db.models import MediaItem, PipelineRun, RunStatus

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "editor-test-csrf-key"


def _app(
    session_factory: sessionmaker[Session], *, media_enabled: bool = True
) -> TestClient:
    with TemporaryDirectory() as tmpdir:
        settings = Settings(
            _env_file=None,  # type: ignore[call-arg]
            voxint_user=CREDS[0],
            voxint_password=CREDS[1],
            media_root=Path(tmpdir),
            console_media_enabled=media_enabled,
            csrf_secret=_CSRF_KEY,
        )
        seed_onboarded(session_factory)
        app = create_app(settings=settings, session_factory=session_factory)
        return TestClient(app)


def _seed_media_with_run(
    session_factory: sessionmaker[Session],
    *,
    status: str = RunStatus.COMPLETED.value,
) -> tuple[uuid.UUID, uuid.UUID]:
    with session_factory() as session:
        m = MediaItem(source_path=f"/audio/{uuid.uuid4()}.wav")
        session.add(m)
        session.flush()
        r = PipelineRun(
            media_item_id=m.id,
            status=status,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        session.add(r)
        session.commit()
        return m.id, r.id


def test_unknown_media_returns_404(
    session_factory: sessionmaker[Session],
) -> None:
    client = _app(session_factory)
    resp = client.get(
        f"/media/{uuid.uuid4()}/editor",
        auth=CREDS,
        follow_redirects=False,
    )
    assert resp.status_code == 404


def test_media_enabled_gate(
    session_factory: sessionmaker[Session],
) -> None:
    media_id, _ = _seed_media_with_run(session_factory)
    client = _app(session_factory, media_enabled=False)
    resp = client.get(
        f"/media/{media_id}/editor", auth=CREDS, follow_redirects=False
    )
    assert resp.status_code == 404


def test_detail_page_renders(
    session_factory: sessionmaker[Session],
) -> None:
    media_id, _run_id = _seed_media_with_run(session_factory)
    client = _app(session_factory)
    resp = client.get(
        f"/media/{media_id}/editor", auth=CREDS, follow_redirects=False
    )
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_no_store_on_detail(
    session_factory: sessionmaker[Session],
) -> None:
    media_id, _ = _seed_media_with_run(session_factory)
    client = _app(session_factory)
    resp = client.get(
        f"/media/{media_id}/editor", auth=CREDS, follow_redirects=False
    )
    assert resp.headers.get("cache-control") == "no-store"


def test_claim_token_read_only_when_absent(
    session_factory: sessionmaker[Session],
) -> None:
    media_id, _ = _seed_media_with_run(session_factory)
    client = _app(session_factory)
    resp = client.get(
        f"/media/{media_id}/editor", auth=CREDS, follow_redirects=False
    )
    assert resp.status_code == 200
    assert "Read-only" in resp.text or "Claim this run" in resp.text


def test_valid_claim_token_enables_editing(
    session_factory: sessionmaker[Session],
) -> None:
    media_id, run_id = _seed_media_with_run(session_factory)
    with session_factory() as session:
        token = claim_run(
            session, run_id, reviewer="reviewer", ttl_seconds=3600
        )
        session.commit()

    client = _app(session_factory)
    resp = client.get(
        f"/media/{media_id}/editor?token={token}",
        auth=CREDS,
        follow_redirects=False,
    )
    assert resp.status_code == 200
    assert "Editing enabled" in resp.text


def test_stale_claim_token_degrades_to_read_only(
    session_factory: sessionmaker[Session],
) -> None:
    media_id, _ = _seed_media_with_run(session_factory)
    stale_token = uuid.uuid4()
    client = _app(session_factory)
    resp = client.get(
        f"/media/{media_id}/editor?token={stale_token}",
        auth=CREDS,
        follow_redirects=False,
    )
    assert resp.status_code == 200
    assert "Read-only" in resp.text or "Claim this run" in resp.text


def test_media_no_runs_renders(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        m = MediaItem(source_path=f"/audio/{uuid.uuid4()}.wav")
        session.add(m)
        session.commit()
        media_id = m.id

    client = _app(session_factory)
    resp = client.get(
        f"/media/{media_id}/editor", auth=CREDS, follow_redirects=False
    )
    assert resp.status_code == 200
    assert "No runs" in resp.text


# ---- Island mount point (#157) ----


def test_completed_run_mounts_editor_island(
    session_factory: sessionmaker[Session],
) -> None:
    media_id, _ = _seed_media_with_run(session_factory)
    client = _app(session_factory)
    resp = client.get(
        f"/media/{media_id}/editor", auth=CREDS, follow_redirects=False
    )
    assert resp.status_code == 200
    assert 'data-island="media-editor"' in resp.text
    assert "data-props=" in resp.text


def test_non_completed_run_has_no_island(
    session_factory: sessionmaker[Session],
) -> None:
    media_id, _ = _seed_media_with_run(
        session_factory, status=RunStatus.QUEUED.value
    )
    client = _app(session_factory)
    resp = client.get(
        f"/media/{media_id}/editor", auth=CREDS, follow_redirects=False
    )
    assert resp.status_code == 200
    assert 'data-island="media-editor"' not in resp.text
