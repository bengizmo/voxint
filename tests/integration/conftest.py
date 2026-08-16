"""Integration fixtures: a real Postgres migrated by the alembic chain.

Set VOXINT_TEST_DATABASE_URL to run these (CI provides a pgvector service);
they are skipped when it is absent. The database is wiped per test.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text
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


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    assert TEST_DB_URL is not None
    eng = create_engine(TEST_DB_URL)
    with eng.connect() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        conn.commit()
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    os.environ["DATABASE_URL"] = TEST_DB_URL
    command.upgrade(cfg, "head")
    yield eng
    eng.dispose()


@pytest.fixture()
def session_factory(engine: Engine) -> Iterator[sessionmaker[Session]]:
    yield sessionmaker(engine, expire_on_commit=False)
    with engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(text(f'TRUNCATE TABLE "{table.name}" CASCADE'))
        conn.commit()
