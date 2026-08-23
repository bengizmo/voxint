"""Migration 0037 (pipeline_runs detected language), up/down (issue #124).

Real alembic up/down against the shared test database (head restored in
teardown): the nullable ``detected_language`` text column and
``detected_language_probability`` float column appear on ``pipeline_runs``,
values round-trip, the CHECK constraint refuses an out-of-range probability,
downgrade to 0036 drops both, upgrade restores them, and the ORM model stays
in lockstep with the migrated DDL.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.types import TypeEngine

from voxint.db.models import PipelineRun

REPO_ROOT = Path(__file__).resolve().parents[2]

_COLUMNS = ("detected_language", "detected_language_probability")


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


def _columns(engine: Engine) -> dict[str, dict[str, object]]:
    return {
        col["name"]: col
        for col in inspect(engine).get_columns("pipeline_runs")
        if col["name"] in _COLUMNS
    }


def _insert_run(
    conn: object, *, language: str | None, probability: float | None
) -> uuid.UUID:
    media_id, run_id = uuid.uuid4(), uuid.uuid4()
    conn.execute(  # type: ignore[attr-defined]
        text("INSERT INTO media_items (id, source_path) VALUES (:id, :p)"),
        {"id": media_id, "p": f"incoming/{media_id}.wav"},
    )
    conn.execute(  # type: ignore[attr-defined]
        text(
            "INSERT INTO pipeline_runs (id, media_item_id, status, revision,"
            " detected_language, detected_language_probability)"
            " VALUES (:id, :m, 'completed', 0, :lang, :prob)"
        ),
        {"id": run_id, "m": media_id, "lang": language, "prob": probability},
    )
    return run_id


def test_columns_present_nullable_and_typed(
    engine: Engine, alembic_cfg: Config
) -> None:
    cols = _columns(engine)
    assert set(cols) == set(_COLUMNS)
    assert cols["detected_language"]["nullable"] is True
    assert isinstance(
        cols["detected_language"]["type"], postgresql.TEXT | postgresql.VARCHAR
    )
    assert cols["detected_language_probability"]["nullable"] is True
    assert isinstance(
        cols["detected_language_probability"]["type"],
        postgresql.FLOAT | postgresql.DOUBLE_PRECISION,
    )


def test_values_round_trip(engine: Engine, alembic_cfg: Config) -> None:
    with engine.connect() as conn:
        run_id = _insert_run(conn, language="es", probability=0.92)
        null_id = _insert_run(conn, language=None, probability=None)
        conn.commit()
        stored = conn.execute(
            text(
                "SELECT detected_language, detected_language_probability"
                " FROM pipeline_runs WHERE id = :r"
            ),
            {"r": run_id},
        ).one()
        stored_null = conn.execute(
            text(
                "SELECT detected_language, detected_language_probability"
                " FROM pipeline_runs WHERE id = :r"
            ),
            {"r": null_id},
        ).one()
    assert stored == ("es", 0.92)
    assert stored_null == (None, None)


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_check_rejects_out_of_range_probability(
    engine: Engine, alembic_cfg: Config, bad: float
) -> None:
    with engine.connect() as conn, pytest.raises(IntegrityError) as exc_info:
        _insert_run(conn, language="en", probability=bad)
    assert "pipeline_runs_detected_language_probability_check" in str(exc_info.value)


def test_check_permits_boundaries_and_null(
    engine: Engine, alembic_cfg: Config
) -> None:
    with engine.connect() as conn:
        _insert_run(conn, language="en", probability=0.0)
        _insert_run(conn, language="en", probability=1.0)
        # NULL probability beside a recorded language is legal: the forced/
        # fallback branches record a language with no detection score.
        _insert_run(conn, language="en", probability=None)
        conn.commit()


def test_downgrade_drops_columns(engine: Engine, alembic_cfg: Config) -> None:
    assert set(_columns(engine)) == set(_COLUMNS)
    command.downgrade(alembic_cfg, "0036")
    assert _columns(engine) == {}
    command.upgrade(alembic_cfg, "head")
    assert set(_columns(engine)) == set(_COLUMNS)


def test_model_matches_migrated_schema(engine: Engine, alembic_cfg: Config) -> None:
    reflected = {
        col["name"]: _pg_type(col["type"])
        for col in inspect(engine).get_columns("pipeline_runs")
    }
    model = {col.name: _pg_type(col.type) for col in PipelineRun.__table__.columns}
    assert reflected == model
