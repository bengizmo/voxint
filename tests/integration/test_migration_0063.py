"""Migration 0063 (watch_pickup_marker), issue #478.

Up/down against the shared test database: the two nullable columns appear on
media_items, the partial index exists, downgrade removes them, and the ORM
model matches the migrated DDL.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[2]


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


def _col_names(engine: Engine) -> set[str]:
    inspector = inspect(engine)
    return {c["name"] for c in inspector.get_columns("media_items")}


def _index_names(engine: Engine) -> set[str]:
    inspector = inspect(engine)
    return {idx["name"] for idx in inspector.get_indexes("media_items")}


def test_upgrade_adds_columns_and_index(
    alembic_cfg: Config, engine: Engine
) -> None:
    command.downgrade(alembic_cfg, "0062")
    cols_before = _col_names(engine)
    assert "picked_up_by_sweep_at" not in cols_before
    assert "picked_up_from_folder" not in cols_before

    command.upgrade(alembic_cfg, "0063")

    cols_after = _col_names(engine)
    assert "picked_up_by_sweep_at" in cols_after
    assert "picked_up_from_folder" in cols_after
    assert "ix_media_items_pickup" in _index_names(engine)


def test_downgrade_removes_columns_and_index(
    alembic_cfg: Config, engine: Engine
) -> None:
    command.upgrade(alembic_cfg, "0063")
    assert "picked_up_by_sweep_at" in _col_names(engine)

    command.downgrade(alembic_cfg, "0062")

    cols = _col_names(engine)
    assert "picked_up_by_sweep_at" not in cols
    assert "picked_up_from_folder" not in cols
    assert "ix_media_items_pickup" not in _index_names(engine)


def test_columns_are_nullable(
    alembic_cfg: Config, engine: Engine, session_factory: sessionmaker[Session]
) -> None:
    command.upgrade(alembic_cfg, "0063")
    with session_factory() as session:
        session.execute(
            text(
                "INSERT INTO media_items (id, source_path)"
                " VALUES (gen_random_uuid(), 'test/nullable.wav')"
            )
        )
        row = session.execute(
            text(
                "SELECT picked_up_by_sweep_at, picked_up_from_folder"
                " FROM media_items WHERE source_path = 'test/nullable.wav'"
            )
        ).one()
        assert row[0] is None
        assert row[1] is None


def test_model_columns_match_migration(engine: Engine) -> None:
    from voxint.db.models import MediaItem

    inspector = inspect(engine)
    db_cols = {c["name"] for c in inspector.get_columns("media_items")}
    model_cols = {c.name for c in MediaItem.__table__.columns}
    assert "picked_up_by_sweep_at" in db_cols & model_cols
    assert "picked_up_from_folder" in db_cols & model_cols
