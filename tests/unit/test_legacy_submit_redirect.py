"""Legacy submit/fetch redirects when the Media library is enabled.

When ``console_media_enabled`` is on, POST /submit and POST /fetch redirect to
/media (303) instead of processing the upload. The /runs listing replaces the
upload forms with a banner linking to /media. Batch 0F (Forgejo #9).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from voxint.api.app import create_app
from voxint.api.csrf import CSRF_FETCH, CSRF_SUBMIT, mint_csrf_token
from voxint.api.routers.deps import _get_session, require_onboarded
from voxint.config import Settings

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "legacy-submit-redirect-test"


def _client(
    *, media_enabled: bool, media_root: Path | None = None
) -> TestClient:
    kwargs: dict[str, object] = dict(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        console_media_enabled=media_enabled,
        csrf_secret=_CSRF_KEY,
    )
    if media_root is not None:
        kwargs["media_root"] = media_root
    settings = Settings(**kwargs)  # type: ignore[arg-type]
    app = create_app(settings=settings)
    mock_session = MagicMock(spec=Session)
    app.dependency_overrides[_get_session] = lambda: mock_session
    app.dependency_overrides[require_onboarded] = lambda: None
    client = TestClient(app)
    client.auth = CREDS
    return client


class TestPostSubmitRedirect:
    """POST /submit -> 303 /media when media is enabled."""

    def test_redirects_to_media(self) -> None:
        client = _client(media_enabled=True)
        resp = client.post(
            "/submit",
            data={
                "submission_id": uuid.uuid4().hex,
                "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SUBMIT),
            },
            files={"file": ("test.wav", b"\x00" * 100, "audio/wav")},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert resp.headers["location"] == "/media"

    def test_no_redirect_when_media_disabled(self, tmp_path: Path) -> None:
        """Without the flag, /submit falls through to normal processing."""
        client = _client(media_enabled=False, media_root=tmp_path)
        resp = client.post(
            "/submit",
            data={
                "submission_id": uuid.uuid4().hex,
                "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SUBMIT),
            },
            files={"file": ("test.wav", b"\x00" * 100, "audio/wav")},
            follow_redirects=False,
        )
        assert resp.headers.get("location", "") != "/media"


class TestPostFetchRedirect:
    """POST /fetch -> 303 /media when media is enabled."""

    def test_redirects_to_media(self) -> None:
        client = _client(media_enabled=True)
        resp = client.post(
            "/fetch",
            data={
                "submission_id": uuid.uuid4().hex,
                "url": "https://example.com/audio.wav",
                "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_FETCH),
            },
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert resp.headers["location"] == "/media"

    def test_no_redirect_when_media_disabled(self) -> None:
        client = _client(media_enabled=False)
        resp = client.post(
            "/fetch",
            data={
                "submission_id": uuid.uuid4().hex,
                "url": "https://example.com/audio.wav",
                "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_FETCH),
            },
            follow_redirects=False,
        )
        assert resp.headers.get("location", "") != "/media"


class TestRunsTemplateBanner:
    """GET /runs shows the media redirect banner when media is enabled."""

    def test_banner_shown_when_media_enabled(self) -> None:
        client = _client(media_enabled=True)
        resp = client.get("/runs")
        assert resp.status_code == 200
        # V3: when media library is enabled, the command bar links directly
        # to /media instead of showing an inline banner.
        assert 'href="/media"' in resp.text
        assert 'action="/submit"' not in resp.text

    def test_forms_shown_when_media_disabled(self) -> None:
        client = _client(media_enabled=False)
        resp = client.get("/runs")
        assert resp.status_code == 200
        assert 'action="/submit"' in resp.text
        assert "New submissions live in the" not in resp.text
