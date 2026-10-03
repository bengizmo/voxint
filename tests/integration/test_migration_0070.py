"""Readiness starts closed on upgrade and removes cleanly on downgrade."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect, text


def test_llm_readiness_migration(engine: Engine) -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "alembic"))
    try:
        command.downgrade(cfg, "0069")
        assert "llm_ready" not in {c["name"] for c in inspect(engine).get_columns("gpu_phase")}
        command.upgrade(cfg, "0070")
        columns = {c["name"]: c for c in inspect(engine).get_columns("gpu_phase")}
        assert not columns["llm_ready"]["nullable"]
        assert columns["llm_checked_at"]["nullable"]
        assert columns["llm_checked_at"]["type"].timezone
        with engine.begin() as conn:
            row = conn.execute(text("SELECT llm_ready, llm_checked_at FROM gpu_phase")).one()
            assert row == (False, None)
            conn.execute(text("DELETE FROM gpu_phase"))
            conn.execute(text("INSERT INTO gpu_phase (id) VALUES (1)"))
            assert conn.execute(text("SELECT llm_ready, llm_checked_at FROM gpu_phase")).one() == (
                False,
                None,
            )
        command.downgrade(cfg, "0069")
        names = {c["name"] for c in inspect(engine).get_columns("gpu_phase")}
        assert "llm_ready" not in names and "llm_checked_at" not in names
    finally:
        command.upgrade(cfg, "head")
