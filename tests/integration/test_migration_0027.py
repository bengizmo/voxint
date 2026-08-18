"""Migration 0027 (app_settings.llm_bundled_enabled), up/down.

Real alembic up/down against the shared test database (head restored in
teardown): the one nullable column appears/disappears, a row that predates the
migration reads NULL (the NULL-parity invariant — env still governs the bundle
toggle, no behavior change on upgrade), and a stored override round-trips.
Issue #67.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect, text

REPO_ROOT = Path(__file__).resolve().parents[2]

_NEW_COLUMN = "llm_bundled_enabled"


@pytest.fixture()
def alembic_cfg(engine: Engine) -> Iterator[Config]:
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    try:
        yield cfg
    finally:
        with engine.connect() as conn:
            conn.execute(text("DROP SCHEMA public CASCADE"))
            conn.execute(text("CREATE SCHEMA public"))
            conn.commit()
        command.upgrade(cfg, "head")


def _app_settings_columns(engine: Engine) -> set[str]:
    return {c["name"] for c in inspect(engine).get_columns("app_settings")}


def test_migration_0027_column_null_parity_and_roundtrip(
    alembic_cfg: Config, engine: Engine
) -> None:
    # Before: at 0026 the column does not exist.
    command.downgrade(alembic_cfg, "0026")
    assert _NEW_COLUMN not in _app_settings_columns(engine)

    # A row that predates the migration (only the server-defaulted columns set).
    with engine.connect() as conn:
        conn.execute(text("INSERT INTO app_settings (id) VALUES (1)"))
        conn.commit()

    command.upgrade(alembic_cfg, "0027")

    # The nullable column now exists...
    assert _NEW_COLUMN in _app_settings_columns(engine)

    with engine.connect() as conn:
        # ...and the pre-existing row reads NULL (NULL-parity: env LLM_BUNDLED_ENABLED
        # still governs, the upgrade changed no behavior).
        value = conn.execute(
            text(f"SELECT {_NEW_COLUMN} FROM app_settings WHERE id = 1")
        ).scalar_one()
        assert value is None

        # A stored override round-trips (tri-state: non-NULL wins).
        conn.execute(
            text(f"UPDATE app_settings SET {_NEW_COLUMN} = true WHERE id = 1")
        )
        conn.commit()
        assert (
            conn.execute(
                text(f"SELECT {_NEW_COLUMN} FROM app_settings WHERE id = 1")
            ).scalar_one()
            is True
        )

    # Downgrade drops the column again.
    command.downgrade(alembic_cfg, "0026")
    assert _NEW_COLUMN not in _app_settings_columns(engine)
