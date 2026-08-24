"""In-UI tutorial seeding (issue #75), end to end against real Postgres.

Covers the two web surfaces that replace the CLI-only ``voxint tutorial seed`` so a
non-technical operator never touches a command line:

* ``POST /settings/tutorial/seed`` — seed from the Settings page and drop straight
  into the run; idempotent; CSRF-guarded; GET is not a route; a classified
  storage/asset failure rolls back and re-renders bounded, non-secret guidance.
* ``POST /setup/finish`` with ``start_tutorial=1`` — seed-on-finish during the
  wizard's pre-onboarding Finish step; a seed failure aborts the whole request
  (onboarding NOT completed).

The advisory lock inside ``seed_tutorial_run`` that serialises concurrent seeds
(so no duplicate tutorial run can be built) is exercised directly.
"""

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_SETTINGS, CSRF_SETUP, mint_csrf_token
from voxint.app_settings import get_app_settings, is_onboarded
from voxint.config import Settings
from voxint.db.models import MediaItem, PipelineRun
from voxint.db.session import session_scope
from voxint.tutorial import seed as seed_module
from voxint.tutorial.seed import (
    _SEED_ADVISORY_LOCK_KEY,
    TutorialSeedError,
    seed_tutorial_run,
)

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "tutorial-seed-route-test-csrf-key"


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path,
        csrf_secret=_CSRF_KEY,
    )


def _client(
    session_factory: sessionmaker[Session], settings: Settings, *, onboarded: bool
) -> TestClient:
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    if onboarded:
        seed_onboarded(session_factory)
    return client


def _settings_form(**fields: str) -> dict[str, str]:
    return {"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETTINGS), **fields}


def _setup_form(**fields: str) -> dict[str, str]:
    return {"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETUP), **fields}


def _tutorial_run_id(session_factory: sessionmaker[Session]) -> object:
    with session_factory() as session:
        row = get_app_settings(session)
        return row.tutorial_run_id if row is not None else None


def _count(session_factory: sessionmaker[Session], model: type) -> int:
    with session_factory() as session:
        return session.execute(select(func.count()).select_from(model)).scalar_one()


# --------------------------------------------------------- settings seed route


def test_seed_route_seeds_and_enters_run(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    client = _client(session_factory, settings, onboarded=True)
    resp = client.post(
        "/settings/tutorial/seed", data=_settings_form(), follow_redirects=False
    )
    assert resp.status_code == 303
    run_id = _tutorial_run_id(session_factory)
    assert run_id is not None
    assert resp.headers["location"] == f"/runs/{run_id}?tutorial=run"
    # The live "Start" control now renders (the CLI hint is gone).
    page = client.get("/settings")
    assert "Start the guided tutorial" in page.text
    assert "voxint tutorial seed" not in page.text


def test_seed_route_is_idempotent(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    client = _client(session_factory, settings, onboarded=True)
    first = client.post(
        "/settings/tutorial/seed", data=_settings_form(), follow_redirects=False
    )
    run_id = _tutorial_run_id(session_factory)
    second = client.post(
        "/settings/tutorial/seed", data=_settings_form(), follow_redirects=False
    )
    assert first.status_code == second.status_code == 303
    # Same run, and exactly one tutorial run built — no duplicate on repost.
    assert _tutorial_run_id(session_factory) == run_id
    assert second.headers["location"] == f"/runs/{run_id}?tutorial=run"
    assert _count(session_factory, PipelineRun) == 1


def test_seed_route_requires_csrf(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    client = _client(session_factory, settings, onboarded=True)
    resp = client.post("/settings/tutorial/seed", data={}, follow_redirects=False)
    assert resp.status_code == 403
    # Nothing seeded on a rejected request.
    assert _tutorial_run_id(session_factory) is None
    assert _count(session_factory, PipelineRun) == 0


def test_seed_route_get_is_not_a_route(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    client = _client(session_factory, settings, onboarded=True)
    resp = client.get("/settings/tutorial/seed", follow_redirects=False)
    assert resp.status_code in (404, 405)


# ------------------------------------------------ failure guidance + rollback


def _seed_raises(exc: Exception):
    """A fake seeder that FLUSHES a partial row, then fails — so a correct handler
    must roll the flushed row back despite the request committing on a 200."""

    def _fake(session: Session, *, media_root: Path, settings: Settings) -> object:
        session.add(
            MediaItem(source_path="incoming/partial-seed.wav", media_type="audio/wav")
        )
        session.flush()
        raise exc

    return _fake


def test_seed_storage_failure_rolls_back_and_shows_bounded_message(
    session_factory: sessionmaker[Session],
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(session_factory, settings, onboarded=True)
    monkeypatch.setattr(
        "voxint.api.routers.settings.seed_tutorial_run",
        _seed_raises(PermissionError("/secret/media/root denied")),
    )
    resp = client.post(
        "/settings/tutorial/seed", data=_settings_form(), follow_redirects=False
    )
    assert resp.status_code == 200
    assert "media folder is not writable" in resp.text
    # No path/exception detail or traceback leaks to the operator.
    assert "/secret/media/root" not in resp.text
    assert "PermissionError" not in resp.text
    assert "Traceback" not in resp.text
    # Rollback ran: the flushed partial row is gone, nothing seeded.
    assert _count(session_factory, MediaItem) == 0
    assert _tutorial_run_id(session_factory) is None


def test_seed_asset_failure_shows_asset_message(
    session_factory: sessionmaker[Session],
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(session_factory, settings, onboarded=True)
    monkeypatch.setattr(
        "voxint.api.routers.settings.seed_tutorial_run",
        _seed_raises(FileNotFoundError("sample-3speaker.wav")),
    )
    resp = client.post(
        "/settings/tutorial/seed", data=_settings_form(), follow_redirects=False
    )
    assert resp.status_code == 200
    assert "bundled sample data is missing or unreadable" in resp.text
    assert _count(session_factory, MediaItem) == 0


def test_unclassified_seed_error_propagates_not_masked(
    session_factory: sessionmaker[Session],
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A builder-invariant failure (TutorialSeedError) is a programmer defect, NOT a
    # storage/asset problem: it must surface loudly (500, session_scope rolls back),
    # never dressed up as "reinstall your bundled data" tutorial copy.
    app = create_app(settings=settings, session_factory=session_factory)
    client = TestClient(app, raise_server_exceptions=False)
    client.auth = CREDS
    seed_onboarded(session_factory)
    monkeypatch.setattr(
        "voxint.api.routers.settings.seed_tutorial_run",
        _seed_raises(TutorialSeedError("grounded label drifted")),
    )
    resp = client.post(
        "/settings/tutorial/seed", data=_settings_form(), follow_redirects=False
    )
    assert resp.status_code == 500
    assert "bundled sample data" not in resp.text
    # Nothing committed — the flushed partial row rolled back with the request.
    assert _count(session_factory, MediaItem) == 0
    assert _tutorial_run_id(session_factory) is None


# ------------------------------------------------ setup wizard seed-on-finish


def test_setup_finish_requires_csrf(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    client = _client(session_factory, settings, onboarded=False)
    resp = client.post(
        "/setup/finish", data={"start_tutorial": "1"}, follow_redirects=False
    )
    assert resp.status_code == 403
    # CSRF is checked before any seed or onboarding write.
    assert _tutorial_run_id(session_factory) is None
    with session_factory() as session:
        assert not is_onboarded(session)


def test_setup_finish_seeds_and_launches_when_requested(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    # Fresh install, tutorial NOT seeded: the "Finish setup & start tutorial" button
    # (start_tutorial=1) seeds it and drops the operator into the run.
    client = _client(session_factory, settings, onboarded=False)
    resp = client.post(
        "/setup/finish", data=_setup_form(start_tutorial="1"), follow_redirects=False
    )
    assert resp.status_code == 303
    run_id = _tutorial_run_id(session_factory)
    assert run_id is not None
    assert resp.headers["location"] == f"/runs/{run_id}?tutorial=run"
    with session_factory() as session:
        assert is_onboarded(session)


def test_setup_finish_seed_failure_aborts_onboarding(
    session_factory: sessionmaker[Session],
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client(session_factory, settings, onboarded=False)
    monkeypatch.setattr(
        "voxint.api.routers.settings.seed_tutorial_run",
        _seed_raises(PermissionError("denied")),
    )
    resp = client.post(
        "/setup/finish", data=_setup_form(start_tutorial="1"), follow_redirects=False
    )
    # Re-rendered Finish step with bounded guidance; onboarding NOT completed and
    # nothing seeded (the whole request rolled back).
    assert resp.status_code == 200
    assert "media folder is not writable" in resp.text
    assert _tutorial_run_id(session_factory) is None
    # Rollback ran: the fake's flushed partial row is gone. Without the route's
    # session.rollback() the commit-on-200 would leak it (and, in a real late-stage
    # failure, a whole COMPLETED run into the review queue).
    assert _count(session_factory, MediaItem) == 0
    assert _count(session_factory, PipelineRun) == 0
    with session_factory() as session:
        assert not is_onboarded(session)


# ------------------------------------------------------- concurrency guard (D3)


def test_seed_holds_transaction_advisory_lock(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    # seed_tutorial_run takes a transaction advisory lock before its idempotency
    # read, so a concurrent seeder on another connection cannot proceed in parallel
    # and build a duplicate run. Prove the lock is held while the seeding txn is
    # still open: a second connection's try-lock must fail.
    with session_scope(session_factory) as seeding:
        seed_tutorial_run(seeding, media_root=settings.media_root, settings=settings)
        with session_factory() as other:
            got = other.execute(
                select(func.pg_try_advisory_xact_lock(_SEED_ADVISORY_LOCK_KEY))
            ).scalar()
        assert got is False


def _advisory_waiters(session_factory: sessionmaker[Session]) -> int:
    # Sessions blocked waiting on any advisory lock. In an otherwise-idle test DB
    # this is exactly the concurrent seeder blocked on _SEED_ADVISORY_LOCK_KEY.
    with session_factory() as watcher:
        return watcher.execute(
            text(
                "SELECT count(*) FROM pg_locks "
                "WHERE locktype = 'advisory' AND NOT granted"
            )
        ).scalar_one()


def test_concurrent_seeds_build_exactly_one_run(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    # The real duplicate-prevention proof (and a placement guard): while seeder A
    # holds the lock mid-build in an open transaction, a second seeder B must block
    # on the lock, and once A commits, B adopts A's run instead of building a second.
    # This FAILS if the advisory lock is ever moved below the idempotency read (B
    # would pass its read before blocking and then build a duplicate run).
    result: dict[str, object] = {}
    error: dict[str, BaseException] = {}

    def _seed_b() -> None:
        try:
            with session_scope(session_factory) as sb:
                result["b"] = seed_tutorial_run(
                    sb, media_root=settings.media_root, settings=settings
                )
        except BaseException as exc:  # surface any thread failure to the assertion
            error["b"] = exc

    with session_scope(session_factory) as sa:
        a_id = seed_tutorial_run(sa, media_root=settings.media_root, settings=settings)
        thread = threading.Thread(target=_seed_b)
        thread.start()
        # Wait until B is genuinely blocked on the advisory lock before releasing A.
        for _ in range(500):
            if _advisory_waiters(session_factory) >= 1:
                break
            time.sleep(0.01)
        else:
            thread.join(timeout=5)
            pytest.fail("second seeder never blocked on the advisory lock")
        # Exiting the `with` commits A and releases the lock; B is then admitted.

    thread.join(timeout=10)
    assert not error, f"concurrent seeder raised: {error.get('b')!r}"
    assert result["b"] == a_id
    assert _count(session_factory, PipelineRun) == 1


def test_seed_lock_constant_is_stable() -> None:
    # A drifting key would silently stop serialising against already-deployed
    # seeders; pin it.
    assert seed_module._SEED_ADVISORY_LOCK_KEY == 0x766F78696E747574
