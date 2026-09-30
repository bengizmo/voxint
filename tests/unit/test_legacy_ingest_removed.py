"""The legacy browser ingest routes are gone (#682).

``POST /submit`` and ``POST /fetch`` were the upload and URL-fetch forms on the
old ``/runs`` page. The Media library (``POST /media/submit``,
``POST /media/fetch``) replaced them, and while its flag existed they only
redirected to ``/media`` without processing, so a stale form lost its input
silently. Both now 404 like any unknown route: a stale form fails visibly.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from voxint.api.app import create_app
from voxint.api.csrf import CSRF_MEDIA_FETCH, CSRF_MEDIA_SUBMIT, mint_csrf_token
from voxint.api.routers.deps import _get_session, require_onboarded
from voxint.config import Settings

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "legacy-ingest-removed-test"


def _client(media_root: Path) -> tuple[TestClient, MagicMock]:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        csrf_secret=_CSRF_KEY,
        media_root=media_root,
        _env_file=None,
    )
    app = create_app(settings=settings)
    mock_session = MagicMock(spec=Session)
    app.dependency_overrides[_get_session] = lambda: mock_session
    app.dependency_overrides[require_onboarded] = lambda: None
    client = TestClient(app)
    client.auth = CREDS
    return client, mock_session


def test_legacy_upload_route_is_gone(tmp_path: Path) -> None:
    client, session = _client(tmp_path)
    resp = client.post(
        "/submit",
        files={"file": ("episode.wav", b"RIFF0000WAVE", "audio/wav")},
        data={
            "submission_id": uuid.uuid4().hex,
            "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_MEDIA_SUBMIT),
        },
        follow_redirects=False,
    )
    assert resp.status_code == 404
    assert "location" not in resp.headers
    # Nothing was processed: no bytes landed and no DB work happened.
    assert list(tmp_path.rglob("*")) == []
    session.add.assert_not_called()
    session.commit.assert_not_called()


def test_legacy_fetch_route_is_gone(tmp_path: Path) -> None:
    client, session = _client(tmp_path)
    resp = client.post(
        "/fetch",
        data={
            "url": "https://example.com/audio.wav",
            "submission_id": uuid.uuid4().hex,
            "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_MEDIA_FETCH),
        },
        follow_redirects=False,
    )
    assert resp.status_code == 404
    assert "location" not in resp.headers
    session.add.assert_not_called()
    session.commit.assert_not_called()


def _routed_paths(routes: Iterable[object]) -> set[str]:
    """Every route path, descending the nested console sub-routers."""
    paths: set[str] = set()
    for route in routes:
        if isinstance(route, APIRoute):
            paths.add(route.path)
        else:
            sub = getattr(route, "original_router", None)
            if sub is not None:
                paths |= _routed_paths(sub.routes)
    return paths


@pytest.mark.parametrize("path", ["/submit", "/fetch"])
def test_legacy_ingest_paths_are_not_routed(tmp_path: Path, path: str) -> None:
    """No method at all is registered on the old paths (not just POST), while
    their replacements are."""
    client, _ = _client(tmp_path)
    routed = _routed_paths(client.app.routes)  # type: ignore[attr-defined]
    assert f"/media{path}" in routed
    assert path not in routed
    assert client.get(path, follow_redirects=False).status_code == 404
