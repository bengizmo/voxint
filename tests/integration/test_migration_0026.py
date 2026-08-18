"""Migration 0026 (app_settings watch-folder columns), up/down.

Real alembic up/down against the shared test database (head restored in
teardown): the two nullable columns appear/disappear, a row that predates the
migration reads NULL for both (the NULL-parity invariant — env still governs the
gate, no sweep summary yet), and a JSON summary + tri-state flag round-trip.
Issue #60.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect, text

REPO_ROOT = Path(__file__).resolve().parents[2]

_NEW_COLUMNS = ("watch_folder_enabled", "watch_folder_last_sweep")


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


def test_migration_0026_columns_null_parity_and_roundtrip(
    alembic_cfg: Config, engine: Engine
) -> None:
    # Before: at 0025 neither new column exists.
    command.downgrade(alembic_cfg, "0025")
    before = _app_settings_columns(engine)
    for name in _NEW_COLUMNS:
        assert name not in before, name

    # A row that predates the migration.
    with engine.connect() as conn:
        conn.execute(text("INSERT INTO app_settings (id) VALUES (1)"))
        conn.commit()

    command.upgrade(alembic_cfg, "0026")

    after = _app_settings_columns(engine)
    for name in _NEW_COLUMNS:
        assert name in after, name

    with engine.connect() as conn:
        # NULL-parity: the pre-existing row reads NULL for both (env still governs
        # the gate; the sweep has never run).
        _select = (
            "SELECT watch_folder_enabled, watch_folder_last_sweep FROM app_settings WHERE id = 1"
        )
        vals = conn.execute(text(_select)).one()
        assert vals == (None, None)

        # A stored override + last-sweep summary round-trips.
        summary = {
            "picked_up": 3,
            "already_known": 12,
            "settling": 2,
            "completed_at": "2026-08-18T10:42:00+00:00",
        }
        conn.execute(
            text(
                "UPDATE app_settings SET watch_folder_enabled = true,"
                " watch_folder_last_sweep = CAST(:s AS jsonb) WHERE id = 1"
            ),
            {"s": json.dumps(summary)},
        )
        conn.commit()
        row = conn.execute(text(_select)).one()
        assert row[0] is True
        assert row[1] == summary

    # Downgrade drops the columns again.
    command.downgrade(alembic_cfg, "0025")
    reverted = _app_settings_columns(engine)
    for name in _NEW_COLUMNS:
        assert name not in reverted, name
