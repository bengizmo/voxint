"""Migration 0006 (app_settings singleton), up/down + model↔migration parity.

Mirrors test_migration_0005: runs the real alembic up/down against the shared
test database and restores head in the fixture teardown. Also asserts the ORM
model matches the migrated schema — no autogenerate parity test exists in-tree
and the whole suite builds its schema from the alembic chain, so a drift here
would surface far from its cause.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import IntegrityError

from voxint.db.models import AppSettings

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def alembic_cfg(engine: Engine) -> Iterator[Config]:
    # Depending on `engine` guarantees DATABASE_URL is exported and the schema
    # starts at head. The finally restores a pristine head regardless of outcome
    # so a mid-suite migration test never leaves the DB downgraded for the tests
    # that follow.
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


def _tables(engine: Engine) -> set[str]:
    return set(inspect(engine).get_table_names())


def test_migration_0006_roundtrip(alembic_cfg: Config, engine: Engine) -> None:
    # --- at 0005: app_settings does not exist yet ---
    command.downgrade(alembic_cfg, "0005")
    assert "app_settings" not in _tables(engine)

    # --- upgrade to 0006: the table appears ---
    command.upgrade(alembic_cfg, "0006")
    assert "app_settings" in _tables(engine)

    # a bare singleton insert relies on the server defaults
    with engine.connect() as conn:
        conn.execute(text("INSERT INTO app_settings (id) VALUES (1)"))
        conn.commit()
        row = conn.execute(
            text(
                "SELECT onboarding_complete, media_folders, vocabulary, llm_enabled"
                " FROM app_settings WHERE id = 1"
            )
        ).one()
    assert row.onboarding_complete is False
    assert row.media_folders == []
    assert row.vocabulary == []
    assert row.llm_enabled is False

    # the id = 1 CHECK pins the table to a single row
    with engine.connect() as conn, pytest.raises(IntegrityError):
        conn.execute(text("INSERT INTO app_settings (id) VALUES (2)"))

    # a second id = 1 violates the primary key
    with engine.connect() as conn, pytest.raises(IntegrityError):
        conn.execute(text("INSERT INTO app_settings (id) VALUES (1)"))

    # --- downgrade is clean (purely additive migration) ---
    command.downgrade(alembic_cfg, "0005")
    assert "app_settings" not in _tables(engine)


def test_app_settings_model_matches_migration(engine: Engine) -> None:
    """The ORM model and the migrated table agree on columns + nullability.

    The suite builds its schema from the alembic chain, not create_all, and there
    is no autogenerate parity check — so assert it here for the new table.
    """
    insp = inspect(engine)
    migrated = {c["name"]: c for c in insp.get_columns("app_settings")}
    model = {c.name: c for c in AppSettings.__table__.columns}
    assert set(migrated) == set(model)
    for name, col in model.items():
        assert migrated[name]["nullable"] == col.nullable, f"{name} nullability drift"
    check_names = {c["name"] for c in insp.get_check_constraints("app_settings")}
    assert "app_settings_single_row_check" in check_names
    fks = insp.get_foreign_keys("app_settings")
    assert any(fk["referred_table"] == "pipeline_runs" for fk in fks)
