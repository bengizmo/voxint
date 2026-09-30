"""The sidebar Media link and the "Add media" pointers (#154, #646, #682).

The Media library is the only place media is added, and it is not behind a flag
any more (#682), so every pointer leads to /media: the sidebar Media link, the
Home "+ Add media" and "Upload a recording" actions, and the Runs page's
"+ Add media". The sidebar link reads as the current page only on the Media
area. A read-only operator gets no add actions and no copy pointing at them.

Skipped without VOXINT_TEST_DATABASE_URL (the pages issue real queries).
"""

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.auth import SESSION_COOKIE
from voxint.api.csrf import CSRF_LOGIN, mint_csrf_token
from voxint.config import Settings
from voxint.db.models import UserRole
from voxint.users import create_user

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "media-pointers-test-csrf-key"
_ADMIN_PW = "admin-pass-123"  # pragma: allowlist secret
_VIEWER_PW = "viewer-pass-789"  # pragma: allowlist secret
_RAIL_MEDIA = re.compile(r'<a class="rail-link" href="([^"]*)"([^>]*) aria-label="Media">')


def _client(session_factory: sessionmaker[Session], tmp_path: Path) -> TestClient:
    settings = Settings(
        _env_file=None,
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path / "media",
    )
    settings.media_root.mkdir(parents=True, exist_ok=True)
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory)
    return client


def _viewer_client(session_factory: sessionmaker[Session], tmp_path: Path) -> TestClient:
    """A signed-in read-only operator (multi-user mode, viewer role)."""
    settings = Settings(
        _env_file=None,
        voxint_multi_user=True,
        voxint_user="ignored",
        voxint_password="ignored",  # pragma: allowlist secret
        csrf_secret=_CSRF_KEY,
        media_root=tmp_path / "media",
    )
    settings.media_root.mkdir(parents=True, exist_ok=True)
    with session_factory() as db:
        # The first account is always promoted to admin, so the viewer comes second.
        create_user(db, username="admin", password=_ADMIN_PW)
        create_user(db, username="viewer", password=_VIEWER_PW, role=UserRole.VIEWER)
        db.commit()
    seed_onboarded(session_factory)
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    resp = client.post(
        "/login",
        data={
            "username": "viewer",
            "password": _VIEWER_PW,
            "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_LOGIN),
            "next": "/",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303, f"login failed: {resp.status_code}"
    assert SESSION_COOKIE in client.cookies
    return client


@pytest.mark.parametrize(
    ("path", "current"), [("/", False), ("/runs", False), ("/media", True)]
)
def test_sidebar_media_link_points_at_the_library(
    session_factory: sessionmaker[Session], tmp_path: Path, path: str, current: bool
) -> None:
    page = _client(session_factory, tmp_path).get(path)
    assert page.status_code == 200
    links = _RAIL_MEDIA.findall(page.text)
    assert len(links) == 1, links
    href, attrs = links[0]
    assert href == "/media"
    # Current only on the Media area: the Runs page is not the Media area any more.
    assert ('aria-current="page"' in attrs) is current


def test_add_media_pointers_lead_to_the_library(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client = _client(session_factory, tmp_path)
    home = client.get("/").text
    assert '<a class="cb-btn cb-btn-primary" href="/media">+ Add media</a>' in home
    assert '<a class="start-btn" href="/media">Upload a recording</a>' in home
    runs = client.get("/runs").text
    assert '<a class="cb-btn cb-btn-primary" href="/media">+ Add media</a>' in runs
    for body in (home, runs):
        assert "#add-media" not in body
        assert 'action="/submit"' not in body
        assert 'action="/fetch"' not in body


def test_empty_library_copy_points_a_writer_at_the_add_button(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    body = _client(session_factory, tmp_path).get("/media").text
    assert (
        "No media yet. Add a recording with the button above, or register a folder "
        "on the server." in body
    )


def test_read_only_operator_gets_no_add_media_pointers(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """The add button is gated on write access, so a viewer is not told to use it."""
    client = _viewer_client(session_factory, tmp_path)
    media = client.get("/media")
    assert media.status_code == 200
    assert '<p class="oc-muted">No media yet.</p>' in media.text
    assert "button above" not in media.text
    assert "+ Add media" not in media.text
    for path in ("/", "/runs"):
        page = client.get(path)
        assert page.status_code == 200
        assert "+ Add media" not in page.text
        assert "Upload a recording" not in page.text
