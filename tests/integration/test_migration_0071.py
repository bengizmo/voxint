"""Filler lists inherit by SQL NULL and survive a migration round trip."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect, text
from sqlalchemy.orm import Session

from voxint.db.models import AppSettings


def test_filler_lists_migration(engine: Engine) -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "alembic"))
    names = {"fillers_add", "fillers_keep"}
    try:
        command.downgrade(cfg, "0070")
        assert names.isdisjoint(c["name"] for c in inspect(engine).get_columns("app_settings"))
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM app_settings"))
            conn.execute(text("INSERT INTO app_settings (id) VALUES (1)"))
        command.upgrade(cfg, "0071")
        columns = {c["name"]: c for c in inspect(engine).get_columns("app_settings")}
        assert all(columns[name]["nullable"] for name in names)
        with engine.connect() as conn:
            assert conn.execute(
                text("SELECT fillers_add IS NULL, fillers_keep IS NULL FROM app_settings")
            ).one() == (True, True)
        with Session(engine) as session:
            row = session.get(AppSettings, 1)
            assert row is not None
            row.fillers_add = {"en": ["you know"]}
            row.fillers_keep = {"en": ["um"]}
            session.commit()
            session.expire_all()
            assert row.fillers_add == {"en": ["you know"]}
            assert row.fillers_keep == {"en": ["um"]}
            row.fillers_add = None
            row.fillers_keep = None
            session.commit()
            assert session.execute(
                text("SELECT fillers_add IS NULL, fillers_keep IS NULL FROM app_settings")
            ).one() == (True, True)
        command.downgrade(cfg, "0070")
        assert names.isdisjoint(c["name"] for c in inspect(engine).get_columns("app_settings"))
    finally:
        command.upgrade(cfg, "head")
