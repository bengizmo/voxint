"""Migration 0012 (run_enrichment_assets + run_asset_jobs), up/down + parity.

Mirrors test_migration_0011's shape: real alembic up/down against the shared
test database (head restored in teardown), every named CHECK exercised with a
rejecting row, the immutability trigger exercised directly, and ORM/DDL
column parity for both new tables.
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

from voxint.db.models import RunAssetJob, RunEnrichmentAsset

REPO_ROOT = Path(__file__).resolve().parents[2]

MEDIA_ID = "00000000-0000-0000-0000-00000000020a"
RUN_ID = "00000000-0000-0000-0000-00000000020b"
ASSET_ID = "00000000-0000-0000-0000-000000000210"
JOB_ID = "00000000-0000-0000-0000-000000000230"

HASH = "0" * 64


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


def _seed_run(engine: Engine) -> None:
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO media_items (id, source_path) VALUES"
                f" ('{MEDIA_ID}', 'incoming/assets/source')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO pipeline_runs (id, media_item_id, status, revision)"
                f" VALUES ('{RUN_ID}', '{MEDIA_ID}', 'completed', 0)"
            )
        )
        conn.commit()


def _insert_asset(engine: Engine, **overrides: str) -> None:
    values = {
        "id": f"'{ASSET_ID}'",
        "pipeline_run_id": f"'{RUN_ID}'",
        "asset_kind": "'summary'",
        "generation": "1",
        "payload": '\'{"summary": "x"}\'::jsonb',
        "payload_schema_version": "1",
        "producer": "'run_assets.llm'",
        "producer_version": "'1'",
        "model": "'m'",
        "source_content_hash": f"'{HASH}'",
        "idempotency_key": f"'key-{ASSET_ID}'",
        "started_at": "now()",
        "completed_at": "now()",
        **overrides,
    }
    columns = ", ".join(values)
    row = ", ".join(values.values())
    with engine.connect() as conn:
        conn.execute(text(f"INSERT INTO run_enrichment_assets ({columns}) VALUES ({row})"))
        conn.commit()


def _insert_job(engine: Engine, **overrides: str) -> None:
    values = {
        "id": f"'{JOB_ID}'",
        "pipeline_run_id": f"'{RUN_ID}'",
        "asset_kind": "'summary'",
        "status": "'queued'",
        "config": '\'{"model": "m"}\'::jsonb',
        **overrides,
    }
    columns = ", ".join(values)
    row = ", ".join(values.values())
    with engine.connect() as conn:
        conn.execute(text(f"INSERT INTO run_asset_jobs ({columns}) VALUES ({row})"))
        conn.commit()


def test_migration_0012_roundtrip_and_checks(alembic_cfg: Config, engine: Engine) -> None:
    command.downgrade(alembic_cfg, "0011")
    tables = inspect(engine).get_table_names()
    assert "run_enrichment_assets" not in tables
    assert "run_asset_jobs" not in tables

    command.upgrade(alembic_cfg, "0012")
    tables = inspect(engine).get_table_names()
    assert "run_enrichment_assets" in tables
    assert "run_asset_jobs" in tables

    _seed_run(engine)
    _insert_asset(engine)

    # -- asset CHECKs, each rejecting a bad row --------------------------------
    bad = "00000000-0000-0000-0000-0000000002"
    with pytest.raises(IntegrityError, match="run_enrichment_assets_kind_check"):
        _insert_asset(
            engine,
            id=f"'{bad}11'",
            asset_kind="'sentiment'",
            idempotency_key="'k11'",
            generation="2",
        )
    with pytest.raises(IntegrityError, match="run_enrichment_assets_generation_check"):
        _insert_asset(engine, id=f"'{bad}12'", generation="0", idempotency_key="'k12'")
    with pytest.raises((IntegrityError, DBAPIError)):
        _insert_asset(
            engine,
            id=f"'{bad}13'",
            generation="2",
            payload="'[]'::jsonb",
            idempotency_key="'k13'",
        )
    with pytest.raises(IntegrityError, match="run_enrichment_assets_source_hash_check"):
        _insert_asset(
            engine,
            id=f"'{bad}14'",
            generation="2",
            source_content_hash="'not-hex'",
            idempotency_key="'k14'",
        )
    with pytest.raises(IntegrityError, match="run_enrichment_assets_config_pair_check"):
        _insert_asset(
            engine,
            id=f"'{bad}15'",
            generation="2",
            config_schema_version="1",
            idempotency_key="'k15'",
        )
    with pytest.raises(IntegrityError, match="run_enrichment_assets_completed_after_started_check"):
        _insert_asset(
            engine,
            id=f"'{bad}16'",
            generation="2",
            completed_at="now() - interval '1 hour'",
            idempotency_key="'k16'",
        )
    with pytest.raises(IntegrityError, match="run_enrichment_assets_generation_key"):
        _insert_asset(engine, id=f"'{bad}17'", idempotency_key="'k17'")
    with pytest.raises(IntegrityError, match="run_enrichment_assets_idempotency_key"):
        _insert_asset(engine, id=f"'{bad}18'", generation="2")

    # -- immutability trigger --------------------------------------------------
    with pytest.raises(DBAPIError, match="born unsuperseded"):
        _insert_asset(
            engine,
            id=f"'{bad}19'",
            generation="2",
            idempotency_key="'k19'",
            superseded_by_asset_id=f"'{ASSET_ID}'",
        )
    with engine.connect() as conn, pytest.raises(DBAPIError, match="content is immutable"):
        conn.execute(
            text(
                'UPDATE run_enrichment_assets SET payload = \'{"summary": "y"}\'::jsonb'
                f" WHERE id = '{ASSET_ID}'"
            )
        )
    with engine.connect() as conn, pytest.raises(DBAPIError, match="DELETE blocked"):
        conn.execute(text(f"DELETE FROM run_enrichment_assets WHERE id = '{ASSET_ID}'"))
    # The one permitted mutation: stamping supersession once.
    _insert_asset(engine, id=f"'{bad}20'", generation="2", idempotency_key="'k20'")
    with engine.connect() as conn:
        conn.execute(
            text(
                f"UPDATE run_enrichment_assets SET superseded_by_asset_id = '{bad}20'"
                f" WHERE id = '{ASSET_ID}'"
            )
        )
        conn.commit()
    with engine.connect() as conn, pytest.raises(DBAPIError, match="write-once"):
        conn.execute(
            text(
                f"UPDATE run_enrichment_assets SET superseded_by_asset_id = '{ASSET_ID}'"
                f" WHERE id = '{ASSET_ID}'"
            )
        )

    # -- job CHECKs + one-active index ------------------------------------------
    _insert_job(engine)
    with engine.connect() as conn:
        row = conn.execute(
            text(f"SELECT cancel_requested FROM run_asset_jobs WHERE id = '{JOB_ID}'")
        ).one()
        assert tuple(row) == (False,)
    badj = "00000000-0000-0000-0000-0000000002"
    with pytest.raises(IntegrityError, match="run_asset_jobs_kind_check"):
        _insert_job(engine, id=f"'{badj}31'", asset_kind="'sentiment'", status="'failed'")
    with pytest.raises(IntegrityError, match="run_asset_jobs_status_check"):
        _insert_job(engine, id=f"'{badj}32'", status="'paused'")
    with pytest.raises((IntegrityError, DBAPIError)):
        _insert_job(engine, id=f"'{badj}33'", status="'failed'", config="'[]'::jsonb")
    with pytest.raises(IntegrityError, match="run_asset_jobs_finished_requires_started_check"):
        _insert_job(engine, id=f"'{badj}34'", status="'failed'", finished_at="now()")
    # One active job per (run, kind); a different kind or a terminal row is fine.
    with pytest.raises(IntegrityError, match="run_asset_jobs_one_active_per_run_kind"):
        _insert_job(engine, id=f"'{badj}35'")
    _insert_job(engine, id=f"'{badj}36'", asset_kind="'topics'")
    _insert_job(engine, id=f"'{badj}37'", status="'failed'")

    # Downgrade removes both tables again; head restored by the fixture.
    command.downgrade(alembic_cfg, "0011")
    tables = inspect(engine).get_table_names()
    assert "run_enrichment_assets" not in tables
    assert "run_asset_jobs" not in tables


def test_asset_model_matches_migrated_schema(engine: Engine) -> None:
    reflected = {
        col["name"]: _pg_type(col["type"])
        for col in inspect(engine).get_columns("run_enrichment_assets")
    }
    model = {col.name: _pg_type(col.type) for col in RunEnrichmentAsset.__table__.columns}
    assert reflected == model


def test_job_model_matches_migrated_schema(engine: Engine) -> None:
    reflected = {
        col["name"]: _pg_type(col["type"]) for col in inspect(engine).get_columns("run_asset_jobs")
    }
    model = {col.name: _pg_type(col.type) for col in RunAssetJob.__table__.columns}
    assert reflected == model
