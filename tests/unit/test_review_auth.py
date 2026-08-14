"""Auth boundary: everything but /healthz challenges; identity is credentials-only."""

import pytest
from fastapi.testclient import TestClient

from voxint.api.app import create_app
from voxint.config import Settings

CREDS = ("reviewer", "s3cret")


@pytest.fixture()
def client() -> TestClient:
    settings = Settings(voxint_user=CREDS[0], voxint_password=CREDS[1])
    return TestClient(create_app(settings=settings))


def test_healthz_is_open(client: TestClient) -> None:
    assert client.get("/healthz").status_code == 200


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/"),
        ("GET", "/runs"),
        ("GET", "/runs/00000000-0000-0000-0000-000000000000"),
        ("GET", "/runs/00000000-0000-0000-0000-000000000000/transcript"),
        ("POST", "/submit"),
        ("POST", "/fetch"),
        ("POST", "/runs/00000000-0000-0000-0000-000000000000/requeue"),
        ("GET", "/review"),
        ("GET", "/review/00000000-0000-0000-0000-000000000000"),
        ("POST", "/review/00000000-0000-0000-0000-000000000000/claim"),
        ("POST", "/review/00000000-0000-0000-0000-000000000000/labels/S0/decision"),
        ("POST", "/review/00000000-0000-0000-0000-000000000000/labels/S0/enroll"),
        ("GET", "/review/00000000-0000-0000-0000-000000000000/export.txt"),
        ("GET", "/review/00000000-0000-0000-0000-000000000000/export.srt"),
        ("GET", "/review/00000000-0000-0000-0000-000000000000/export.vtt"),
        ("GET", "/review/00000000-0000-0000-0000-000000000000/export.json"),
        ("GET", "/review/00000000-0000-0000-0000-000000000000/export.rttm"),
        ("GET", "/metrics"),
        ("GET", "/media/00000000-0000-0000-0000-000000000000"),
        ("POST", "/review/00000000-0000-0000-0000-000000000000/release"),
        ("GET", "/static/htmx.min.js"),
    ],
)
def test_routes_challenge_unauthenticated(
    client: TestClient, method: str, path: str
) -> None:
    resp = client.request(method, path)
    assert resp.status_code == 401
    assert "WWW-Authenticate" in resp.headers


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_generated_doc_surfaces_disabled(client: TestClient, path: str) -> None:
    # Docs/OpenAPI would be unauthenticated routes — they must not exist.
    assert client.get(path).status_code == 404


def test_htmx_asset_served_authenticated(client: TestClient) -> None:
    resp = client.get("/static/htmx.min.js", auth=CREDS)
    assert resp.status_code == 200
    assert "htmx" in resp.text[:200]


def test_wrong_password_rejected(client: TestClient) -> None:
    resp = client.get("/review", auth=(CREDS[0], "wrong"))
    assert resp.status_code == 401


def test_wrong_username_rejected(client: TestClient) -> None:
    resp = client.get("/review", auth=("intruder", CREDS[1]))
    assert resp.status_code == 401


@pytest.mark.parametrize("password", ["change-me", ""])
def test_default_or_empty_credentials_refused_off_loopback(password: str) -> None:
    with pytest.raises(ValueError, match="voxint_password"):
        Settings(api_host="0.0.0.0", voxint_password=password)


def test_default_credentials_fine_on_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fully hermetic: exercise the *code* default, not the ambient environment.
    # Settings source precedence is init kwargs > process env > dotenv, so both
    # need neutralizing: _env_file=None skips any on-disk .env, and delenv drops
    # an exported VOXINT_PASSWORD (either would otherwise flip this assertion).
    monkeypatch.delenv("VOXINT_PASSWORD", raising=False)
    assert Settings(api_host="127.0.0.1", _env_file=None).voxint_password == "change-me"
