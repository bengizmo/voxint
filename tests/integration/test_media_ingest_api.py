"""POST /media/submit and /media/fetch — the P2b library ingest (issue #154).

Acceptance AC-1: upload and URL ingestion from /media produce IDENTICAL
media_items/pipeline_runs rows to the legacy /runs path, because both call the
same broker-free backends (submit_upload/submit_url). These tests submit the same
bytes/URL through each surface and diff the durable columns. They also pin the
/media redirect (back to the library, not /runs) and the flag-off 404. Needs the
real Postgres test DB, so skipped without VOXINT_TEST_DATABASE_URL.
"""

import hashlib
import io
import uuid
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import (
    CSRF_FETCH,
    CSRF_MEDIA_FETCH,
    CSRF_MEDIA_SUBMIT,
    CSRF_SUBMIT,
    mint_csrf_token,
)
from voxint.config import Settings
from voxint.db.models import MediaItem, PipelineRun, RunStatus

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "media-ingest-test-csrf-key"
_URL = "https://example.com/audio.mp3"


def _data(action: str, **fields: str) -> dict[str, str]:
    """Form fields with a valid CSRF token for ``action`` merged in."""
    return {"csrf_token": mint_csrf_token(_CSRF_KEY, action), **fields}


def _wav_bytes(seconds: float = 0.02) -> bytes:
    frames = int(16000 * seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * frames)
    return buf.getvalue()


def _make_client(
    session_factory: sessionmaker[Session],
    media_root: Path,
    *,
    media_enabled: bool = True,
) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=media_root,
        upload_max_bytes=10 * 1024 * 1024,
        ytdlp_enabled=True,
        console_media_enabled=media_enabled,
        csrf_secret=_CSRF_KEY,
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory)
    return client


@pytest.fixture()
def client(session_factory: sessionmaker[Session], tmp_path: Path) -> TestClient:
    return _make_client(session_factory, tmp_path)


@pytest.fixture()
def legacy_client(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> TestClient:
    return _make_client(session_factory, tmp_path, media_enabled=False)


@pytest.fixture()
def published(monkeypatch: pytest.MonkeyPatch) -> list[uuid.UUID]:
    calls: list[uuid.UUID] = []
    monkeypatch.setattr(
        "voxint.api.routers.deps._publish_run",
        lambda run_id, **_kwargs: calls.append(run_id),
    )
    return calls


def _media_and_run(
    session: Session, source_path: str
) -> tuple[MediaItem, PipelineRun]:
    media = session.execute(
        select(MediaItem).where(MediaItem.source_path == source_path)
    ).scalar_one()
    run = session.execute(
        select(PipelineRun)
        .where(PipelineRun.media_item_id == media.id)
        .order_by(PipelineRun.created_at.desc())
        .limit(1)
    ).scalar_one()
    return media, run


def test_media_upload_matches_legacy_upload(
    client: TestClient,
    legacy_client: TestClient,
    session_factory: sessionmaker[Session],
    published: list[uuid.UUID],
) -> None:
    body = _wav_bytes()
    legacy_sub, media_sub = uuid.uuid4().hex, uuid.uuid4().hex

    legacy = legacy_client.post(
        "/submit",
        files={"file": ("clip.wav", body, "audio/wav")},
        data=_data(CSRF_SUBMIT, submission_id=legacy_sub),
        follow_redirects=False,
    )
    assert legacy.status_code == 303
    assert legacy.headers["location"].startswith("/runs/")

    media = client.post(
        "/media/submit",
        files={"file": ("clip.wav", body, "audio/wav")},
        data=_data(CSRF_MEDIA_SUBMIT, submission_id=media_sub),
        follow_redirects=False,
    )
    assert media.status_code == 303
    # The library ingest returns to /media, not the legacy run page.
    assert media.headers["location"] == "/media?submitted=1"

    with session_factory() as session:
        m_legacy, r_legacy = _media_and_run(session, f"incoming/{legacy_sub}/clip.wav")
        m_media, r_media = _media_and_run(session, f"incoming/{media_sub}/clip.wav")
        # Identical durable content: the differing leg is only the uuid namespace.
        assert m_media.sha256 == m_legacy.sha256 == hashlib.sha256(body).hexdigest()
        assert m_media.size_bytes == m_legacy.size_bytes == len(body)
        assert m_media.media_folder_id is None and m_legacy.media_folder_id is None
        assert r_media.status == r_legacy.status == RunStatus.QUEUED.value
        assert r_media.domain_pack == r_legacy.domain_pack
    assert len(published) == 2


def test_media_fetch_matches_legacy_fetch(
    client: TestClient,
    legacy_client: TestClient,
    session_factory: sessionmaker[Session],
    published: list[uuid.UUID],
) -> None:
    legacy_sub, media_sub = uuid.uuid4().hex, uuid.uuid4().hex

    legacy = legacy_client.post(
        "/fetch",
        data=_data(CSRF_FETCH, url=_URL, submission_id=legacy_sub),
        follow_redirects=False,
    )
    assert legacy.status_code == 303

    media = client.post(
        "/media/fetch",
        data=_data(CSRF_MEDIA_FETCH, url=_URL, submission_id=media_sub),
        follow_redirects=False,
    )
    assert media.status_code == 303
    assert media.headers["location"] == "/media?submitted=1"

    with session_factory() as session:
        m_legacy, r_legacy = _media_and_run(session, f"incoming/{legacy_sub}/source")
        m_media, r_media = _media_and_run(session, f"incoming/{media_sub}/source")
        assert m_media.source_url == m_legacy.source_url == _URL
        assert m_media.media_folder_id is None and m_legacy.media_folder_id is None
        assert r_media.status == r_legacy.status == RunStatus.QUEUED.value
        assert r_media.domain_pack == r_legacy.domain_pack
    assert len(published) == 2


def test_media_forms_render_with_picker(client: TestClient) -> None:
    body = client.get("/media").text
    assert 'action="/media/submit"' in body
    assert 'action="/media/fetch"' in body
    # The settings-folder picker and its honest "not moved" copy render.
    assert 'name="media_folder_id"' in body
    assert "not moved" in body


def test_media_routes_404_when_flag_off(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _make_client(session_factory, tmp_path, media_enabled=False)
    assert client.get("/media").status_code == 404
    # The POST routes are registered but gated: 404, not a 405/403, with the flag off.
    resp = client.post(
        "/media/submit",
        files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
        data=_data(CSRF_MEDIA_SUBMIT, submission_id=uuid.uuid4().hex),
        follow_redirects=False,
    )
    assert resp.status_code == 404
