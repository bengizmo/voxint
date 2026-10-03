"""Recording date migration upgrades and downgrades cleanly."""

from pathlib import Path

import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect


def test_recorded_date_migration(engine: Engine) -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "alembic"))
    try:
        command.downgrade(cfg, "0068")
        assert "recorded_on" not in {c["name"] for c in inspect(engine).get_columns("media_items")}
        command.upgrade(cfg, "0069")
        column = next(c for c in inspect(engine).get_columns("media_items")
                      if c["name"] == "recorded_on")
        assert column["nullable"]
        assert isinstance(column["type"], sa.Date)
    finally:
        command.upgrade(cfg, "head")
