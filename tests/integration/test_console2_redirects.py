"""Live verification of the Console 2.0 redirect map (issue #150).

The declarative table lives in ``tests/contracts/test_console2_characterization``
(``REDIRECT_MAP``); the structural guards there run without a database. This
integration test drives a real onboarded, authenticated client so each declared
redirect is proven live. Future phases that append a legacy redirect get its
end-to-end assertion for free by adding a row to the table.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.contracts.test_console2_characterization import REDIRECT_MAP, RedirectRule
from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.config import Settings
from voxint.db.models import MediaItem, PipelineRun, RunStatus

CREDS = ("operator", "pw")


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path,
        csrf_secret="test-csrf-secret-value-0123456789",
    )


def _seed_run(session_factory: sessionmaker[Session]) -> tuple[uuid.UUID, uuid.UUID]:
    with session_factory() as session:
        media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
        session.add(media)
        session.flush()
        run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
        session.add(run)
        session.commit()
        return run.id, media.id


def _client(
    session_factory: sessionmaker[Session], settings: Settings, *, auth: bool
) -> TestClient:
    app = create_app(settings=settings, session_factory=session_factory)
    client = TestClient(app)
    if auth:
        client.auth = CREDS
    seed_onboarded(session_factory)
    return client


@pytest.mark.parametrize("rule", REDIRECT_MAP, ids=lambda r: r.source)
def test_redirect_map_is_live(
    session_factory: sessionmaker[Session], settings: Settings, rule: RedirectRule
) -> None:
    client = _client(session_factory, settings, auth=rule.auth)
    run_id: uuid.UUID | None = None
    media_id: uuid.UUID | None = None
    source = rule.source
    target = rule.target
    if "{run_id}" in source:
        run_id, media_id = _seed_run(session_factory)
        source = source.replace("{run_id}", str(run_id))
        target = target.replace("{media_id}", str(media_id))
    response = client.get(source, follow_redirects=False)
    assert response.status_code == rule.status, (
        f"{source} should {rule.status}-redirect, got {response.status_code}"
    )
    location = response.headers["location"]
    assert location.split("?")[0] == target, (
        f"{source} should redirect to {target}, got {location}"
    )
