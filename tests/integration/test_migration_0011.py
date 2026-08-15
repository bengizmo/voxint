"""Migration 0011 (research_jobs), up/down + model parity.

Mirrors test_migration_0010's shape at the scale one mutable table needs:
real alembic up/down against the shared test database (head restored in
teardown), every named CHECK exercised with a rejecting row, and ORM/DDL
column parity for the new table.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.types import TypeEngine

from voxint.db.models import ResearchJob

REPO_ROOT = Path(__file__).resolve().parents[2]

SPEAKER_ID = "00000000-0000-0000-0000-00000000010a"
JOB_ID = "00000000-0000-0000-0000-000000000110"


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


def _seed_speaker(engine: Engine) -> None:
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO speakers (id, display_name) VALUES"
                f" ('{SPEAKER_ID}', 'Research Jobs Parity Speaker')"
            )
        )
        conn.commit()


def _insert_job(engine: Engine, **overrides: str) -> None:
    values = {
        "id": f"'{JOB_ID}'",
        "speaker_id": f"'{SPEAKER_ID}'",
        "status": "'queued'",
        "budget": "'{\"max_searches\": 3}'::jsonb",
        **overrides,
    }
    columns = ", ".join(values)
    row = ", ".join(values.values())
    with engine.connect() as conn:
        conn.execute(text(f"INSERT INTO research_jobs ({columns}) VALUES ({row})"))
        conn.commit()


def test_migration_0011_roundtrip_and_checks(alembic_cfg: Config, engine: Engine) -> None:
    command.downgrade(alembic_cfg, "0010")
    assert "research_jobs" not in inspect(engine).get_table_names()

    command.upgrade(alembic_cfg, "0011")
    assert "research_jobs" in inspect(engine).get_table_names()

    _seed_speaker(engine)
    _insert_job(engine)
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT cancel_requested, searches_used, reads_used, rounds_used"
                f" FROM research_jobs WHERE id = '{JOB_ID}'"
            )
        ).one()
        assert tuple(row) == (False, 0, 0, 0)  # server defaults

    # Named CHECKs each reject a bad row.
    with pytest.raises(IntegrityError, match="research_jobs_status_check"):
        _insert_job(engine, id="'00000000-0000-0000-0000-000000000111'", status="'paused'")
    with pytest.raises(IntegrityError, match="research_jobs_counters_check"):
        _insert_job(engine, id="'00000000-0000-0000-0000-000000000112'", searches_used="-1")
    with pytest.raises((IntegrityError, DBAPIError)):
        _insert_job(engine, id="'00000000-0000-0000-0000-000000000113'", budget="'[]'::jsonb")
    with pytest.raises(IntegrityError, match="research_jobs_finished_requires_started_check"):
        _insert_job(engine, id="'00000000-0000-0000-0000-000000000114'", finished_at="now()")

    # Downgrade removes the table again; head restored by the fixture.
    command.downgrade(alembic_cfg, "0010")
    assert "research_jobs" not in inspect(engine).get_table_names()


def test_model_matches_migrated_schema(engine: Engine) -> None:
    reflected = {
        col["name"]: _pg_type(col["type"]) for col in inspect(engine).get_columns("research_jobs")
    }
    model = {col.name: _pg_type(col.type) for col in ResearchJob.__table__.columns}
    assert reflected == model
