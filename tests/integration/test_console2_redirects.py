"""Live verification of the Console 2.0 redirect map (issue #150).

The declarative table lives in ``tests/contracts/test_console2_characterization``
(``REDIRECT_MAP``); the structural guards there run without a database. This
integration test drives a real onboarded, authenticated client so each declared
redirect is proven live. Future phases that append a legacy redirect get its
end-to-end assertion for free by adding a row to the table.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.contracts.test_console2_characterization import REDIRECT_MAP, RedirectRule
from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.config import Settings

CREDS = ("operator", "pw")


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path,
        csrf_secret="test-csrf-secret-value-0123456789",
    )


@pytest.fixture()
def client(
    session_factory: sessionmaker[Session], settings: Settings
) -> TestClient:
    app = create_app(settings=settings, session_factory=session_factory)
    client = TestClient(app)
    client.auth = CREDS
    seed_onboarded(session_factory)
    return client


@pytest.mark.parametrize("rule", REDIRECT_MAP, ids=lambda r: r.source)
def test_redirect_map_is_live(client: TestClient, rule: RedirectRule) -> None:
    response = client.get(rule.source, follow_redirects=False)
    assert response.status_code == rule.status, (
        f"{rule.source} should {rule.status}-redirect, got {response.status_code}"
    )
    location = response.headers["location"]
    # A redirect may carry a query string (e.g. a claim token); match the path.
    assert location.split("?")[0] == rule.target, (
        f"{rule.source} should redirect to {rule.target}, got {location}"
    )
