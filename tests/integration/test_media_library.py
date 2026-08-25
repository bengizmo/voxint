"""The media library page and its query (Console 2.0 P2a, #153) end to end.

The library lists every media file with its folder membership and latest-run
status. These tests pin the wiring the pure helpers cannot see: the area flag
gate (404 until ``console_media_enabled``, auth first), the aggregate's
latest-run-per-file and archived-run exclusion, the folder join, the sort
allowlist's honest degrade, and the card/table view toggle.
"""

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.media_query import media_library
from voxint.config import Settings
from voxint.db.models import (
    MediaFolder,
    MediaItem,
    MediaSourceMetadata,
    PipelineRun,
    RunStatus,
    SourceKind,
)

CREDS = ("reviewer", "s3cret")


def _make_client(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
    *,
    media_enabled: bool,
) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path,
        console_media_enabled=media_enabled,
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory)
    return client


@pytest.fixture()
def client(session_factory: sessionmaker[Session], tmp_path: Path) -> TestClient:
    """The page with the area flag ON (the common case these tests exercise)."""
    return _make_client(session_factory, tmp_path, media_enabled=True)


def _add_media(
    session: Session,
    *,
    source_path: str,
    title: str | None = None,
    folder_id: uuid.UUID | None = None,
    duration_seconds: float | None = None,
    size_bytes: int | None = None,
    created_at: datetime | None = None,
) -> MediaItem:
    media = MediaItem(
        source_path=source_path,
        media_folder_id=folder_id,
        duration_seconds=duration_seconds,
        size_bytes=size_bytes,
    )
    if created_at is not None:
        media.created_at = created_at
    session.add(media)
    session.flush()
    if title is not None:
        session.add(
            MediaSourceMetadata(
                media_item_id=media.id,
                source_kind=SourceKind.YTDLP.value,
                title=title,
                raw_schema_version=1,
                acquired_at=datetime.now(UTC),
            )
        )
    return media


def _add_run(
    session: Session,
    media: MediaItem,
    *,
    status: RunStatus,
    created_at: datetime | None = None,
    archived: bool = False,
) -> PipelineRun:
    run = PipelineRun(media_item_id=media.id, status=status.value)
    if created_at is not None:
        run.created_at = created_at
    if archived:
        run.archived_at = datetime.now(UTC)
    session.add(run)
    session.flush()
    return run


# ---- the area flag gate -----------------------------------------------------


def test_media_404s_when_flag_off(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _make_client(session_factory, tmp_path, media_enabled=False)
    assert client.get("/media").status_code == 404


def test_media_requires_auth_before_the_gate(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    # Auth runs ahead of the area gate, so an unauthenticated request gets a 401
    # challenge and never learns whether the hidden area exists — even off.
    client = _make_client(session_factory, tmp_path, media_enabled=False)
    client.auth = None
    assert client.get("/media").status_code == 401


def test_media_redirects_when_not_onboarded(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path,
        console_media_enabled=True,
    )
    client = TestClient(
        create_app(settings=settings, session_factory=session_factory)
    )
    client.auth = CREDS
    # No seed_onboarded: the onboarding gate (which runs before the area gate)
    # sends an ordinary navigation to /setup.
    resp = client.get("/media", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/setup"


# ---- the page ---------------------------------------------------------------


def test_empty_library_states_it_honestly(client: TestClient) -> None:
    resp = client.get("/media")
    assert resp.status_code == 200
    assert "No media yet" in resp.text


def test_library_lists_a_file_with_folder_and_status(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        folder = MediaFolder(path="interviews", domain_pack="interview")
        session.add(folder)
        session.flush()
        media = _add_media(
            session,
            source_path="interviews/mayor.wav",
            title="Mayor interview",
            folder_id=folder.id,
            duration_seconds=3661.0,
            size_bytes=1024 * 1024,
        )
        _add_run(session, media, status=RunStatus.COMPLETED)
        session.commit()

    resp = client.get("/media")
    assert resp.status_code == 200
    body = resp.text
    assert "Mayor interview" in body
    assert "interviews" in body
    assert "1:01:01" in body  # format_duration
    assert "1.0 MB" in body  # format_size
    assert "Completed" in body  # humanize_status of the latest run


# ---- the aggregate ----------------------------------------------------------


def test_latest_run_per_file_wins(
    session_factory: sessionmaker[Session],
) -> None:
    now = datetime.now(UTC)
    with session_factory() as session:
        media = _add_media(session, source_path="incoming/a.wav")
        _add_run(
            session, media, status=RunStatus.FAILED, created_at=now - timedelta(hours=2)
        )
        _add_run(
            session,
            media,
            status=RunStatus.COMPLETED,
            created_at=now - timedelta(minutes=5),
        )
        session.commit()

        rows = media_library(session)
    assert len(rows) == 1
    # The newer (completed) run, not the older failed one.
    assert rows[0].latest_run_status == RunStatus.COMPLETED.value


def test_archived_runs_are_excluded(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        media = _add_media(session, source_path="incoming/b.wav")
        _add_run(session, media, status=RunStatus.COMPLETED, archived=True)
        session.commit()

        rows = media_library(session)
    assert len(rows) == 1
    # A file whose only run is archived reads as "not processed yet".
    assert rows[0].latest_run_id is None
    assert rows[0].latest_run_status is None


def test_media_with_no_run_has_no_status(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        _add_media(session, source_path="incoming/c.wav")
        session.commit()
        rows = media_library(session)
    assert rows[0].latest_run_id is None


def test_limit_truncates(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        _add_media(session, source_path="incoming/d1.wav")
        _add_media(session, source_path="incoming/d2.wav")
        session.commit()
        rows = media_library(session, limit=1)
    assert len(rows) == 1


# ---- sort + view ------------------------------------------------------------


def test_sort_by_name_orders_alphabetically(
    session_factory: sessionmaker[Session],
) -> None:
    now = datetime.now(UTC)
    with session_factory() as session:
        # Zed added most recently (would lead the default "added" sort); Alpha
        # added first. Sorting by name must put Alpha before Zed regardless.
        _add_media(
            session,
            source_path="incoming/zed.wav",
            title="Zed",
            created_at=now,
        )
        _add_media(
            session,
            source_path="incoming/alpha.wav",
            title="Alpha",
            created_at=now - timedelta(hours=1),
        )
        session.commit()
        by_name = [r.source_title for r in media_library(session, sort="name")]
        by_added = [r.source_title for r in media_library(session, sort="added")]
    assert by_name == ["Alpha", "Zed"]
    assert by_added == ["Zed", "Alpha"]


def test_unknown_sort_degrades_to_default_not_422(client: TestClient) -> None:
    resp = client.get("/media?sort=bogus")
    assert resp.status_code == 200


def test_table_view_renders_a_table(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        _add_media(session, source_path="incoming/e.wav", title="Table me")
        session.commit()
    resp = client.get("/media?view=table")
    assert resp.status_code == 200
    assert "<table>" in resp.text


def test_unknown_view_degrades_to_cards(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        _add_media(session, source_path="incoming/f.wav", title="Card me")
        session.commit()
    resp = client.get("/media?view=bogus")
    assert resp.status_code == 200
    # The cards list, not the table region.
    assert 'class="lib-cards"' in resp.text
