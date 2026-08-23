"""Migration 0034 (pipeline_runs.initial_prompt), up/down (issue #123).

Real alembic up/down against the shared test database (head restored in
teardown): the nullable ``initial_prompt`` text column appears on
``pipeline_runs``, a value round-trips, downgrade to 0033 drops it, upgrade
restores it, and the ORM model stays in lockstep with the migrated DDL.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.types import TypeEngine

from voxint.db.models import PipelineRun

REPO_ROOT = Path(__file__).resolve().parents[2]


def _pg_type(type_: TypeEngine[object]) -> str:
    compiled = type_.compile(dialect=postgresql.dialect())
    return "DOUBLE PRECISION" if compiled == "FLOAT" else compiled


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


def _column(engine: Engine) -> dict[str, object] | None:
    for col in inspect(engine).get_columns("pipeline_runs"):
        if col["name"] == "initial_prompt":
            return col
    return None


def test_column_present_and_nullable_text(engine: Engine, alembic_cfg: Config) -> None:
    col = _column(engine)
    assert col is not None
    assert col["nullable"] is True
    assert isinstance(col["type"], postgresql.TEXT | postgresql.VARCHAR)


def test_value_round_trips(engine: Engine, alembic_cfg: Config) -> None:
    media_id, run_id = uuid.uuid4(), uuid.uuid4()
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO media_items (id, source_path) VALUES (:id, :p)"),
            {"id": media_id, "p": f"incoming/{media_id}.wav"},
        )
        conn.execute(
            text(
                "INSERT INTO pipeline_runs (id, media_item_id, status, revision,"
                " initial_prompt) VALUES (:id, :m, 'completed', 0, :prompt)"
            ),
            {"id": run_id, "m": media_id, "prompt": "Zoning Board, NUCA"},
        )
        conn.commit()
        stored = conn.execute(
            text("SELECT initial_prompt FROM pipeline_runs WHERE id = :r"), {"r": run_id}
        ).scalar_one()
    assert stored == "Zoning Board, NUCA"


def test_downgrade_drops_column(engine: Engine, alembic_cfg: Config) -> None:
    assert _column(engine) is not None
    command.downgrade(alembic_cfg, "0033")
    assert _column(engine) is None
    command.upgrade(alembic_cfg, "head")
    assert _column(engine) is not None


def test_model_matches_migrated_schema(engine: Engine, alembic_cfg: Config) -> None:
    reflected = {
        col["name"]: _pg_type(col["type"])
        for col in inspect(engine).get_columns("pipeline_runs")
    }
    model = {col.name: _pg_type(col.type) for col in PipelineRun.__table__.columns}
    assert reflected == model
