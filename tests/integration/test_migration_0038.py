"""Migration 0038 (transcript translation), up/down (issue #133).

Real alembic up/down against the shared test database (head restored in
teardown): the two nullable ``app_settings`` columns and the
``run_translations`` / ``translation_jobs`` tables appear, key CHECK
constraints and the one-active partial unique index hold, downgrade to 0037
drops everything, and upgrade restores it.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import IntegrityError

REPO_ROOT = Path(__file__).resolve().parents[2]

_APP_SETTINGS_COLUMNS = ("translation_target_language", "translation_autogenerate")
_TABLES = ("run_translations", "translation_jobs")

_HASH = "0" * 64


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


def _seed_run(conn: object) -> uuid.UUID:
    media_id, run_id = uuid.uuid4(), uuid.uuid4()
    conn.execute(  # type: ignore[attr-defined]
        text("INSERT INTO media_items (id, source_path) VALUES (:id, :p)"),
        {"id": media_id, "p": f"incoming/{media_id}.wav"},
    )
    conn.execute(  # type: ignore[attr-defined]
        text(
            "INSERT INTO pipeline_runs (id, media_item_id, status, revision)"
            " VALUES (:id, :m, 'completed', 0)"
        ),
        {"id": run_id, "m": media_id},
    )
    return run_id


def _insert_job(conn: object, run_id: uuid.UUID, *, status: str = "queued") -> uuid.UUID:
    job_id = uuid.uuid4()
    conn.execute(  # type: ignore[attr-defined]
        text(
            "INSERT INTO translation_jobs (id, pipeline_run_id, target_language,"
            " status, cancel_requested, config, source_content_hash)"
            " VALUES (:id, :r, 'es', :s, false, '{}'::jsonb, :h)"
        ),
        {"id": job_id, "r": run_id, "s": status, "h": _HASH},
    )
    return job_id


def test_schema_present_and_constraints_hold(engine: Engine, alembic_cfg: Config) -> None:
    inspector = inspect(engine)
    for table in _TABLES:
        assert inspector.has_table(table)
    app_cols = {c["name"]: c for c in inspector.get_columns("app_settings")}
    for name in _APP_SETTINGS_COLUMNS:
        assert name in app_cols and app_cols[name]["nullable"]

    with engine.connect() as conn:
        run_id = _seed_run(conn)
        _insert_job(conn, run_id)
        conn.commit()
    # One active job per (run, language): a second queued 'es' job is refused.
    with engine.connect() as conn, pytest.raises(IntegrityError):
        _insert_job(conn, run_id)
    # A terminal job frees the slot.
    with engine.connect() as conn:
        conn.execute(
            text("UPDATE translation_jobs SET status = 'failed', started_at = now(),"
                 " finished_at = now() WHERE pipeline_run_id = :r"),
            {"r": run_id},
        )
        _insert_job(conn, run_id)
        conn.commit()
    # CHECKs: bad status and malformed hash refused.
    with engine.connect() as conn, pytest.raises(IntegrityError):
        conn.execute(
            text(
                "INSERT INTO translation_jobs (id, pipeline_run_id,"
                " target_language, status, cancel_requested, config,"
                " source_content_hash) VALUES (:id, :r, 'fr', 'sideways',"
                " false, '{}'::jsonb, :h)"
            ),
            {"id": uuid.uuid4(), "r": run_id, "h": _HASH},
        )
    with engine.connect() as conn, pytest.raises(IntegrityError):
        conn.execute(
            text(
                "INSERT INTO run_translations (id, pipeline_run_id,"
                " target_language, generation, lines, payload_schema_version,"
                " producer, producer_version, model, source_content_hash,"
                " started_at, completed_at) VALUES (:id, :r, 'es', 1,"
                " '[]'::jsonb, 1, 'p', '1', 'm', 'not-a-hash', now(), now())"
            ),
            {"id": uuid.uuid4(), "r": run_id},
        )
    # lines must be a JSON array, never an object.
    with engine.connect() as conn, pytest.raises(IntegrityError):
        conn.execute(
            text(
                "INSERT INTO run_translations (id, pipeline_run_id,"
                " target_language, generation, lines, payload_schema_version,"
                " producer, producer_version, model, source_content_hash,"
                " started_at, completed_at) VALUES (:id, :r, 'es', 1,"
                " '{}'::jsonb, 1, 'p', '1', 'm', :h, now(), now())"
            ),
            {"id": uuid.uuid4(), "r": run_id, "h": _HASH},
        )


def test_downgrade_drops_and_upgrade_restores(engine: Engine, alembic_cfg: Config) -> None:
    command.downgrade(alembic_cfg, "0037")
    inspector = inspect(engine)
    for table in _TABLES:
        assert not inspector.has_table(table)
    app_cols = {c["name"] for c in inspector.get_columns("app_settings")}
    assert not app_cols & set(_APP_SETTINGS_COLUMNS)

    command.upgrade(alembic_cfg, "0038")
    inspector = inspect(engine)
    for table in _TABLES:
        assert inspector.has_table(table)
    app_cols = {c["name"] for c in inspector.get_columns("app_settings")}
    assert app_cols >= set(_APP_SETTINGS_COLUMNS)
