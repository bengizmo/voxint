"""The sidebar Media link and the "Add media" pointers (#154, #646, #682).

The Media library is the only place media is added, and it is not behind a flag
any more (#682), so every pointer leads to /media: the sidebar Media link, the
Home "+ Add media" and "Upload a recording" actions, and the Runs page's
"+ Add media". The sidebar link reads as the current page only on the Media
area.

Skipped without VOXINT_TEST_DATABASE_URL (the pages issue real queries).
"""

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.config import Settings

CREDS = ("reviewer", "s3cret")
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
