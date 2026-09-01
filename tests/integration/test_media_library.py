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
    # R3 grid table does not show file size (mockup omits the column).
    assert "reviewed" in body  # completed run with no unresolved labels


# ---- search + status filters -----------------------------------------------


def test_search_by_title(client: TestClient, session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        _add_media(session, source_path="incoming/alpha.wav", title="Alpha interview")
        _add_media(session, source_path="incoming/bravo.wav", title="Bravo briefing")
        session.commit()

    resp = client.get("/media?q=Alpha")
    assert "Alpha interview" in resp.text
    assert "Bravo briefing" not in resp.text


def test_search_by_source_path(client: TestClient, session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        _add_media(session, source_path="incoming/council-session.wav", title="Council audio")
        _add_media(session, source_path="incoming/other.wav", title="Other audio")
        session.commit()

    resp = client.get("/media?q=council-session")
    assert "Council audio" in resp.text
    assert "Other audio" not in resp.text


def test_search_by_folder_name(client: TestClient, session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        folder = MediaFolder(path="field-recordings")
        session.add(folder)
        session.flush()
        _add_media(
            session,
            source_path="incoming/one.wav",
            title="Field recording one",
            folder_id=folder.id,
        )
        _add_media(
            session,
            source_path="incoming/two.wav",
            title="Field recording two",
            folder_id=folder.id,
        )
        _add_media(session, source_path="incoming/studio.wav", title="Studio recording")
        session.commit()

    resp = client.get("/media?q=field-recordings")
    assert "Field recording one" in resp.text
    assert "Field recording two" in resp.text
    assert "Studio recording" not in resp.text


def test_search_case_insensitive(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        _add_media(session, source_path="incoming/lower.wav", title="lowercase title")
        session.commit()

    resp = client.get("/media?q=LOWERCASE")
    assert "lowercase title" in resp.text


def test_search_empty_result(client: TestClient, session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        _add_media(session, source_path="incoming/present.wav", title="Present title")
        session.commit()

    resp = client.get("/media?q=does-not-exist")
    assert "No recordings match" in resp.text


def test_status_filter_needs_review(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        review = _add_media(session, source_path="incoming/review.wav", title="Review me")
        done = _add_media(session, source_path="incoming/done.wav", title="Already done")
        _add_run(session, review, status=RunStatus.AWAITING_ADJUDICATION)
        _add_run(session, done, status=RunStatus.COMPLETED)
        session.commit()

    resp = client.get("/media?status=needs_review")
    assert "Review me" in resp.text
    assert "Already done" not in resp.text


def test_status_filter_failed(client: TestClient, session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        failed = _add_media(session, source_path="incoming/failed.wav", title="Failed item")
        running = _add_media(session, source_path="incoming/running.wav", title="Running item")
        _add_run(session, failed, status=RunStatus.FAILED)
        _add_run(session, running, status=RunStatus.RUNNING)
        session.commit()

    resp = client.get("/media?status=failed")
    assert "Failed item" in resp.text
    assert "Running item" not in resp.text


def test_status_filter_reviewed(client: TestClient, session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        done = _add_media(session, source_path="incoming/done.wav", title="Reviewed item")
        queued = _add_media(session, source_path="incoming/queued.wav", title="Queued item")
        _add_run(session, done, status=RunStatus.COMPLETED)
        _add_run(session, queued, status=RunStatus.QUEUED)
        session.commit()

    resp = client.get("/media?status=reviewed")
    assert "Reviewed item" in resp.text
    assert "Queued item" not in resp.text


def test_search_and_status_compose(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        matching = _add_media(session, source_path="incoming/match.wav", title="Council match")
        wrong_status = _add_media(
            session, source_path="incoming/done.wav", title="Council completed"
        )
        wrong_search = _add_media(session, source_path="incoming/other.wav", title="Other failure")
        _add_run(session, matching, status=RunStatus.FAILED)
        _add_run(session, wrong_status, status=RunStatus.COMPLETED)
        _add_run(session, wrong_search, status=RunStatus.FAILED)
        session.commit()

    resp = client.get("/media?q=Council&status=failed")
    assert "Council match" in resp.text
    assert "Council completed" not in resp.text
    assert "Other failure" not in resp.text


def test_row_action_review_verb(client: TestClient, session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        media = _add_media(session, source_path="incoming/review.wav", title="Review action")
        _add_run(session, media, status=RunStatus.AWAITING_ADJUDICATION)
        session.commit()

    assert "Review →" in client.get("/media").text


def test_row_action_retry_verb(client: TestClient, session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        media = _add_media(session, source_path="incoming/retry.wav", title="Retry action")
        _add_run(session, media, status=RunStatus.FAILED)
        session.commit()

    assert "Retry →" in client.get("/media").text


def test_row_action_open_verb(client: TestClient, session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        media = _add_media(session, source_path="incoming/open.wav", title="Open action")
        _add_run(session, media, status=RunStatus.COMPLETED)
        session.commit()

    assert "Open →" in client.get("/media").text


def test_file_missing_plain_language(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        _add_media(session, source_path="incoming/missing.wav", title="Missing original")
        session.commit()

    assert "Original file not found" in client.get("/media").text


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
    assert "grid-table" in resp.text


def test_unknown_view_degrades_to_table(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        _add_media(session, source_path="incoming/f.wav", title="Card me")
        session.commit()
    resp = client.get("/media?view=bogus")
    assert resp.status_code == 200
    assert "grid-table" in resp.text


# ---- cards drill-down (PR #341) ---------------------------------------------


@pytest.fixture()
def _seeded_folders(
    session_factory: sessionmaker[Session],
) -> tuple[uuid.UUID, uuid.UUID]:
    """Two folders with media items; returns (interviews_id, lectures_id)."""
    with session_factory() as session:
        f_int = MediaFolder(path="interviews")
        f_lec = MediaFolder(path="lectures")
        session.add_all([f_int, f_lec])
        session.flush()
        for i in range(3):
            _add_media(
                session,
                source_path=f"interviews/rec_{i}.wav",
                folder_id=f_int.id,
                duration_seconds=60.0 * (i + 1),
            )
        _add_media(
            session,
            source_path="lectures/lec_1.mp3",
            folder_id=f_lec.id,
            duration_seconds=3600.0,
        )
        _add_media(session, source_path="misc/unfiled.flac")
        session.commit()
        return f_int.id, f_lec.id


def test_cards_top_level_shows_folder_cards(
    client: TestClient, _seeded_folders: tuple[uuid.UUID, uuid.UUID]
) -> None:
    f_int, f_lec = _seeded_folders
    resp = client.get("/media?view=cards")
    assert resp.status_code == 200
    body = resp.text
    assert "media-folder-card" in body
    assert f"open={f_int}" in body
    assert f"open={f_lec}" in body
    assert "Unfiled" in body
    # Top-level: no drill-down back link or folder-contents region
    assert "All folders" not in body
    assert 'aria-label="Folder contents"' not in body


def test_cards_drilldown_valid_folder(
    client: TestClient, _seeded_folders: tuple[uuid.UUID, uuid.UUID]
) -> None:
    f_int, _ = _seeded_folders
    resp = client.get(f"/media?view=cards&open={f_int}")
    assert resp.status_code == 200
    body = resp.text
    assert 'aria-label="Folder contents"' in body
    assert "interviews" in body
    assert "All folders" in body
    assert "rec_0.wav" in body
    assert "rec_1.wav" in body
    assert "rec_2.wav" in body
    assert "unfiled.flac" not in body


def test_cards_drilldown_invalid_uuid_falls_back(
    client: TestClient, _seeded_folders: tuple[uuid.UUID, uuid.UUID]
) -> None:
    bogus = uuid.uuid4()
    resp = client.get(f"/media?view=cards&open={bogus}")
    assert resp.status_code == 200
    body = resp.text
    assert "All folders" not in body
    assert "media-folder-card" in body


def test_cards_drilldown_non_uuid_falls_back(
    client: TestClient, _seeded_folders: tuple[uuid.UUID, uuid.UUID]
) -> None:
    resp = client.get("/media?view=cards&open=not-a-uuid")
    assert resp.status_code == 200
    body = resp.text
    assert "All folders" not in body
    assert "media-folder-card" in body


def test_cards_drilldown_sort_links_carry_open(
    client: TestClient, _seeded_folders: tuple[uuid.UUID, uuid.UUID]
) -> None:
    f_int, _ = _seeded_folders
    resp = client.get(f"/media?view=cards&sort=name&open={f_int}")
    assert resp.status_code == 200
    body = resp.text
    for sort_key in ("added", "name", "duration", "size"):
        assert f"sort={sort_key}&view=cards&open={f_int}" in body


def test_cards_drilldown_hidden_input_preserves_open(
    client: TestClient, _seeded_folders: tuple[uuid.UUID, uuid.UUID]
) -> None:
    f_int, _ = _seeded_folders
    resp = client.get(f"/media?view=cards&open={f_int}")
    assert resp.status_code == 200
    assert f'name="open" value="{f_int}"' in resp.text


def test_cards_open_ignored_in_table_view(
    client: TestClient, _seeded_folders: tuple[uuid.UUID, uuid.UUID]
) -> None:
    f_int, _ = _seeded_folders
    resp = client.get(f"/media?view=table&open={f_int}")
    assert resp.status_code == 200
    body = resp.text
    assert "All folders" not in body
    assert 'aria-label="Folder contents"' not in body
    assert "grid-table" in body


def test_cards_top_level_has_selection_affordances(
    client: TestClient, _seeded_folders: tuple[uuid.UUID, uuid.UUID]
) -> None:
    resp = client.get("/media?view=cards")
    assert resp.status_code == 200
    body = resp.text
    # The unfiled item card carries its own checkbox inside the bulk form,
    # and the section has a select-all box plus the (hidden) action bar.
    assert 'class="media-card-check"' in body
    assert 'name="media_id"' in body
    assert 'data-select-all-box aria-label="Select all"' in body
    assert "data-action-bar" in body


def test_cards_drilldown_has_selection_affordances(
    client: TestClient, _seeded_folders: tuple[uuid.UUID, uuid.UUID]
) -> None:
    f_int, _ = _seeded_folders
    resp = client.get(f"/media?view=cards&open={f_int}")
    assert resp.status_code == 200
    body = resp.text
    # Every in-folder row is selectable and the folder view has select-all.
    assert body.count('name="media_id"') == 3
    assert 'data-select-all-box aria-label="Select all"' in body
    assert "data-action-bar" in body
