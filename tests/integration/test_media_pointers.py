"""Flag-aware "Add media" pointers (issue #154, Console 2.0 P2b, Commit 6).

Once ``console_media_enabled`` is on, the operable /media library becomes the
Add-media destination: the sidebar Media link and the Home "Add media" quick
action point there instead of the legacy /runs placeholder. Flag off, every
pointer stays on /runs and no /media link leaks — the legacy path is unchanged.
These render both flag states and assert the pointers switch, and guard that the
flag-off /runs page shows no /media pointer (the dark-ship stays dark).

Skipped without VOXINT_TEST_DATABASE_URL (the pages issue real queries).
"""

from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.config import Settings

CREDS = ("reviewer", "s3cret")


def _client(
    session_factory: sessionmaker[Session], tmp_path: Path, *, media_enabled: bool
) -> TestClient:
    settings = Settings(
        _env_file=None,
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path / "media",
        console_media_enabled=media_enabled,
    )
    settings.media_root.mkdir(parents=True, exist_ok=True)
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory)
    return client


def test_home_add_media_points_at_media_when_enabled(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _client(session_factory, tmp_path, media_enabled=True)
    home = client.get("/")
    assert home.status_code == 200
    assert 'cb-btn-primary' in home.text
    assert 'href="/media"' in home.text
    assert "Add media" in home.text


def test_home_add_media_points_at_runs_when_disabled(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _client(session_factory, tmp_path, media_enabled=False)
    home = client.get("/")
    assert home.status_code == 200
    assert 'cb-btn-primary' in home.text
    assert 'href="/runs#add-media"' in home.text


def test_sidebar_media_link_follows_flag(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    # On Home (active_nav "home"), the Media link points at /media but carries no
    # aria-current — you are not on the Media area.
    on = _client(session_factory, tmp_path, media_enabled=True).get("/")
    assert 'href="/media">Media</a>' in on.text

    off = _client(session_factory, tmp_path, media_enabled=False).get("/")
    # Flag off: Media stays the /runs placeholder.
    assert 'href="/runs">Media</a>' in off.text


def test_sidebar_media_link_is_current_on_media_page(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """On /media (active_nav "media"), the flag-on Media link reads as current."""
    media = _client(session_factory, tmp_path, media_enabled=True).get("/media")
    assert media.status_code == 200
    assert 'href="/media" aria-current="page">Media</a>' in media.text


def test_flag_off_runs_page_has_no_media_pointer(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """The dark-ship stays dark: with the flag off, the legacy /runs page (its
    own Add-media section included) links to no /media route at all."""
    client = _client(session_factory, tmp_path, media_enabled=False)
    runs = client.get("/runs")
    assert runs.status_code == 200
    assert "/media" not in runs.text
    # The legacy Add-media section is still there and still owns the anchor.
    assert 'id="add-media"' in runs.text


def test_flag_on_runs_page_still_serves_legacy_upload(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """Flag on does not break the legacy /runs page — it stays reachable with its
    own upload section (P5 retires it later); only the pointers moved."""
    client = _client(session_factory, tmp_path, media_enabled=True)
    runs = client.get("/runs")
    assert runs.status_code == 200
    assert 'id="add-media"' in runs.text
    # The sidebar on /runs now points Media at the operable library.
    assert 'href="/media"' in runs.text
