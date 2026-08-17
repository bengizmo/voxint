"""Migration 0024 (app_settings feature-flag + external-sources columns), up/down.

Real alembic up/down against the shared test database (head restored in
teardown): the ten nullable columns appear/disappear, a row that predates the
migration reads NULL for every new column (the NULL-parity invariant — env still
governs, no behavior change on upgrade), and a value round-trips. Issue #74.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect, text

REPO_ROOT = Path(__file__).resolve().parents[2]

_NEW_COLUMNS = (
    "enrichment_names_enabled",
    "enrichment_names_llm_enabled",
    "enrichment_run_assets_enabled",
    "enrichment_run_assets_autogenerate",
    "voxint_web_research",
    "enrichment_web_research_enabled",
    "ytdlp_enabled",
    "source_authority_domains",
    "web_search_base_url",
    "web_search_api_key",
)


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


def test_migration_0024_columns_null_parity_and_roundtrip(
    alembic_cfg: Config, engine: Engine
) -> None:
    # Before: at 0023 none of the new columns exist.
    command.downgrade(alembic_cfg, "0023")
    before = _app_settings_columns(engine)
    for name in _NEW_COLUMNS:
        assert name not in before, name

    # A row that predates the migration (only the server-defaulted columns set).
    with engine.connect() as conn:
        conn.execute(text("INSERT INTO app_settings (id) VALUES (1)"))
        conn.commit()

    command.upgrade(alembic_cfg, "0024")

    # The ten nullable columns now exist...
    after = _app_settings_columns(engine)
    for name in _NEW_COLUMNS:
        assert name in after, name

    # ...and the pre-existing row reads NULL for every one of them (NULL-parity:
    # env still governs, the upgrade changed no behavior).
    with engine.connect() as conn:
        cols = ", ".join(_NEW_COLUMNS)
        vals = conn.execute(
            text(f"SELECT {cols} FROM app_settings WHERE id = 1")
        ).one()
        assert all(v is None for v in vals)

        # A stored override round-trips (tri-state: non-NULL wins).
        conn.execute(
            text(
                "UPDATE app_settings SET ytdlp_enabled = true,"
                " voxint_web_research = false,"
                " web_search_base_url = 'http://searx.lan:8888',"
                " web_search_api_key = 'k-secret'"
                " WHERE id = 1"
            )
        )
        conn.commit()
        row = conn.execute(
            text(
                "SELECT ytdlp_enabled, voxint_web_research, web_search_base_url,"
                " web_search_api_key FROM app_settings WHERE id = 1"
            )
        ).one()
        assert row == (True, False, "http://searx.lan:8888", "k-secret")

    # Downgrade drops the columns again.
    command.downgrade(alembic_cfg, "0023")
    reverted = _app_settings_columns(engine)
    for name in _NEW_COLUMNS:
        assert name not in reverted, name
