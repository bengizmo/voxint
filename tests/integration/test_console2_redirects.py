"""Live verification of the Console 2.0 redirect map (issue #150).

The declarative table lives in ``tests/contracts/test_console2_characterization``
(``REDIRECT_MAP``); the structural guards there run without a database. This
integration test drives a real onboarded, authenticated client so each declared
redirect is proven live. Future phases that append a legacy redirect get its
end-to-end assertion for free by adding a row to the table.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from tests.contracts.test_console2_characterization import REDIRECT_MAP, RedirectRule
from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_CLAIM, mint_csrf_token
from voxint.config import Settings
from voxint.db.models import MediaItem, PipelineRun, RunStatus

CREDS = ("operator", "pw")


@pytest.fixture()
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    # Shipped defaults only: no .env file, and no CONSOLE_* flag leaking in from
    # the process environment (which would mask a default regressing, #646).
    for name in list(os.environ):
        if name.upper().startswith("CONSOLE_"):
            monkeypatch.delenv(name)
    return Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path,
        csrf_secret="test-csrf-secret-value-0123456789",
        _env_file=None,
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


def test_review_entry_points_reach_the_editor_on_a_default_install_646(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    """#646: with the shipped defaults, every review entry point must land on a
    live page. The redirect map above checks only each Location header, which is
    how a 302-then-404 editor shipped unnoticed; this follows the hops."""
    assert settings.console_media_enabled is True  # the shipped default, not an override
    client = _client(session_factory, settings, auth=True)
    run_id, media_id = _seed_run(session_factory)
    editor = f"/media/{media_id}/editor"

    for source in (
        f"/review/{run_id}",
        f"/review/{run_id}/transcript",
        f"/runs/{run_id}/transcript",
    ):
        hop = client.get(source, follow_redirects=False)
        assert hop.status_code == 302, source
        assert hop.headers["location"].split("?")[0] == editor, source
        landed = client.get(source)
        assert landed.status_code == 200, f"{source} -> {landed.url} = {landed.status_code}"
        assert landed.url.path == editor

    queue = client.get("/review", follow_redirects=False)
    assert queue.status_code == 303
    assert queue.headers["location"] == "/media"
    assert client.get("/review").status_code == 200

    # The run page's "Open in editor" button.
    run_page = client.get(f"/runs/{run_id}")
    assert run_page.status_code == 200
    button = f"/media/{media_id}/editor?run={run_id}"
    assert f'href="{button}"' in run_page.text
    assert client.get(button).status_code == 200

    # Claiming a run for review (the tutorial's claim form posts here).
    token = mint_csrf_token(settings.csrf_secret, CSRF_CLAIM)
    claim = client.post(
        f"/review/{run_id}/claim", data={"csrf_token": token}, follow_redirects=False
    )
    assert claim.status_code == 303
    assert claim.headers["location"].startswith(f"{editor}?run={run_id}&token=")
    assert client.get(claim.headers["location"]).status_code == 200


def test_legacy_submit_and_fetch_send_a_default_install_to_media_646(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    """On a default install media is added on /media: the legacy Runs-page
    handlers redirect there without processing anything (no run is created)."""
    client = _client(session_factory, settings, auth=True)

    def run_count() -> int:
        with session_factory() as session:
            return session.execute(select(func.count()).select_from(PipelineRun)).scalar_one()

    before = run_count()
    submit = client.post(
        "/submit",
        files={"file": ("clip.wav", b"RIFF", "audio/wav")},
        data={"submission_id": uuid.uuid4().hex},
        follow_redirects=False,
    )
    fetch = client.post(
        "/fetch",
        data={"url": "https://example.com/clip", "submission_id": uuid.uuid4().hex},
        follow_redirects=False,
    )
    for response in (submit, fetch):
        assert response.status_code == 303
        assert response.headers["location"] == "/media"
    assert run_count() == before
