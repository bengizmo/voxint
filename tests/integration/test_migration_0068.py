"""Migration 0068 creates and seeds the singleton, and downgrades cleanly."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect, text


def test_gpu_phase_migration(engine: Engine) -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "alembic"))
    try:
        command.downgrade(cfg, "0067")
        assert "gpu_phase" not in inspect(engine).get_table_names()
        command.upgrade(cfg, "0068")
        with engine.connect() as conn:
            row = conn.execute(text("SELECT * FROM gpu_phase")).mappings().one()
            assert row["id"] == 1 and row["phase"] == "llm"
            assert row["failures"] == 0
            assert row["phase_since"] == row["updated_at"]
            assert row["phase_since"] is not None
            for field in (
                "lease_id",
                "lease_expires_at",
                "last_error",
                "retry_after",
                "operator_request",
            ):
                assert row[field] is None
    finally:
        command.upgrade(cfg, "head")
