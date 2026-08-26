"""Integration fixtures: a real Postgres migrated by the alembic chain.

Set VOXINT_TEST_DATABASE_URL to run these (CI provides a pgvector service);
they are skipped when it is absent. The database is wiped per test.
"""

import os
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from voxint.app_settings import complete_onboarding
from voxint.db.models import Base

TEST_DB_URL = os.environ.get("VOXINT_TEST_DATABASE_URL")


def seed_onboarded(
    session_factory: sessionmaker[Session], *, llm_enabled: bool = False
) -> None:
    """Mark the app onboarded so the first-run gate lets protected routes through.

    The onboarding gate (issue #3) 303s every non-exempt route to ``/setup`` until
    ``onboarding_complete`` is set, so an API test that means to exercise a handler
    must start onboarded. Called explicitly by the API client fixtures/builders —
    deliberately NOT a global autouse fixture, because the ``app_settings``
    repository and migration tests assert on the absent-row ("not onboarded") state.

    ``llm_enabled`` seeds the row's LLM enablement. The enrichment gates resolve
    enablement row-over-env (issue #10), so a test that means to exercise an LLM
    path must onboard with ``llm_enabled=True`` — an onboarded row with the default
    ``False`` now correctly closes those gates even when env ``LLM_ENABLED`` is set.
    """
    with session_factory() as session:
        row = complete_onboarding(session, llm_enabled_default=llm_enabled)
        row.llm_enabled = llm_enabled
        session.commit()

REPO_ROOT = Path(__file__).resolve().parents[2]


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # pytestmark in a conftest does NOT propagate to test modules, and this hook
    # receives the whole session's items — mark only this directory's tests.
    if TEST_DB_URL is None:
        here = Path(__file__).parent
        skip = pytest.mark.skip(reason="VOXINT_TEST_DATABASE_URL not set")
        for item in items:
            if here in Path(str(item.fspath)).parents:
                item.add_marker(skip)


_SAFE_DB_NAME = re.compile(r"\A[A-Za-z0-9_]+\Z")


def _assert_disposable_db(url: str) -> None:
    """Refuse to run destructive setup against anything but a throwaway DB.

    The ``engine`` fixture drops schemas and (under xdist) drops whole databases;
    a typo in ``VOXINT_TEST_DATABASE_URL`` pointed at the live ``voxint`` database
    would silently destroy operator data. Fail closed: the base database name must
    carry a disposable marker (``test`` or ``e2e``). Mirrors the E2E lane's guard
    (``tests/e2e/conftest.py``).
    """
    db_name = (make_url(url).database or "").lower()
    if not db_name or ("test" not in db_name and "e2e" not in db_name):
        pytest.fail(
            "VOXINT_TEST_DATABASE_URL must name a DISPOSABLE database whose name "
            f"contains 'test' or 'e2e' (its schema/database is dropped and "
            f"rebuilt); got {db_name!r}. Refusing to run destructive setup."
        )


def _worker_database_url(base_url: str, worker_id: str, testrun_uid: str) -> str:
    """Per-xdist-worker database URL.

    Without xdist the ``worker_id`` fixture is ``"master"`` and we keep the
    single shared database exactly as before (byte-identical serial behaviour).
    Under ``-n`` each worker gets its own database, named with BOTH the
    per-run ``testrun_uid`` and the ``worker_id`` (e.g.
    ``voxint_test_1a2b3c4d_gw0``). Folding in ``testrun_uid`` is what makes
    *concurrent pytest invocations* safe too — deterministic ``gw0`` names alone
    would let two simultaneous runs force-drop each other's databases, only
    half-fixing the one-invocation-at-a-time deadlock this change targets.

    The name is kept within PostgreSQL's 63-byte identifier limit by using the
    first 8 hex chars of ``testrun_uid``.
    """
    if worker_id == "master":
        return base_url
    url = make_url(base_url)
    name = f"{url.database}_{testrun_uid[:8]}_{worker_id}"
    if not _SAFE_DB_NAME.match(name):
        pytest.fail(f"refusing to create unsafe worker database name {name!r}")
    return url.set(database=name).render_as_string(hide_password=False)


def _admin_url() -> str:
    assert TEST_DB_URL is not None
    return make_url(TEST_DB_URL).set(database="postgres").render_as_string(
        hide_password=False
    )


def _drop_worker_db(name: str) -> None:
    """Drop a per-worker database via an AUTOCOMMIT maintenance connection."""
    admin = create_engine(_admin_url(), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
    finally:
        admin.dispose()


@pytest.fixture(scope="session")
def engine(worker_id: str, testrun_uid: str) -> Iterator[Engine]:
    assert TEST_DB_URL is not None
    _assert_disposable_db(TEST_DB_URL)
    db_url = _worker_database_url(TEST_DB_URL, worker_id, testrun_uid)
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    os.environ["DATABASE_URL"] = db_url

    if worker_id == "master":
        # Single shared database: wipe the schema in place, as before. No
        # per-run database is created, so nothing is dropped at teardown.
        eng = create_engine(db_url)
        with eng.connect() as conn:
            conn.execute(text("DROP SCHEMA public CASCADE"))
            conn.execute(text("CREATE SCHEMA public"))
            conn.commit()
        command.upgrade(cfg, "head")
        try:
            yield eng
        finally:
            eng.dispose()
        return

    # Per-worker database: (re)create it from scratch via the maintenance DB.
    # CREATE/DROP DATABASE cannot run inside a transaction, so the admin
    # connection is AUTOCOMMIT. WITH (FORCE) evicts any stale connection left by
    # a crashed run (Postgres 13+); the database is normally dropped at teardown.
    target = make_url(db_url).database
    assert target is not None
    admin = create_engine(_admin_url(), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{target}" WITH (FORCE)'))
            conn.execute(text(f'CREATE DATABASE "{target}"'))
    finally:
        admin.dispose()

    eng = create_engine(db_url)
    try:
        command.upgrade(cfg, "head")
        yield eng
    finally:
        eng.dispose()
        _drop_worker_db(target)


@pytest.fixture(scope="session", autouse=True)
def _bind_worker_database(engine: Engine) -> None:
    """Pin every integration test to this worker's disposable database.

    Some tests never request ``engine``/``session_factory`` directly — e.g. the
    CLI tests that call ``main([...])`` and let the app build its own engine from
    ``os.environ["DATABASE_URL"]``. They passed under the serial suite only by
    accident: the session-scoped ``engine`` fixture had already run for some
    earlier test and left ``DATABASE_URL`` set as a side effect. Run such a test
    in isolation (or land it alone on an xdist worker) and ``DATABASE_URL`` is
    unset, so ``get_settings()`` falls back to the default *live* DSN — a real
    data hazard and a flaky failure. Depending on ``engine`` here makes that
    setup explicit and universal for every worker.
    """


@pytest.fixture()
def session_factory(engine: Engine) -> Iterator[sessionmaker[Session]]:
    yield sessionmaker(engine, expire_on_commit=False)
    with engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(text(f'TRUNCATE TABLE "{table.name}" CASCADE'))
        conn.commit()
