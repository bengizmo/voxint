"""Migration 0067 (synthdetect_jobs.created_at NOT NULL), issue #692.

Real alembic up/down against the shared test database: rows seeded at 0066
with a NULL ``created_at`` are backfilled (``started_at`` when set, else
``now()``), rows that already had one keep it, the column becomes NOT NULL,
and the downgrade makes it nullable again while keeping backfilled values.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import IntegrityError

REPO_ROOT = Path(__file__).resolve().parents[2]

T0 = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


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


def _created_at_nullable(engine: Engine) -> bool:
    cols = {c["name"]: c for c in inspect(engine).get_columns("synthdetect_jobs")}
    return bool(cols["created_at"]["nullable"])


def _seed_job(
    engine: Engine,
    *,
    status: str,
    created_at: datetime | None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
) -> uuid.UUID:
    """Insert media item -> run -> job with raw SQL (the ORM is at head)."""
    mid, rid, jid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO media_items (id, source_path) VALUES (:id, :sp)"),
            {"id": mid, "sp": f"incoming/{mid}/source"},
        )
        conn.execute(
            text(
                "INSERT INTO pipeline_runs (id, media_item_id, status, revision)"
                " VALUES (:id, :mid, 'completed', 0)"
            ),
            {"id": rid, "mid": mid},
        )
        conn.execute(
            text(
                "INSERT INTO synthdetect_jobs (id, pipeline_run_id, inference_space,"
                " calibration_policy_id, status, created_at, started_at, finished_at)"
                " VALUES (:id, :rid, 'space', 'policy', :status, :created_at,"
                " :started_at, :finished_at)"
            ),
            {
                "id": jid,
                "rid": rid,
                "status": status,
                "created_at": created_at,
                "started_at": started_at,
                "finished_at": finished_at,
            },
        )
    return jid


def _created_at(engine: Engine, job_id: uuid.UUID) -> datetime | None:
    with engine.connect() as conn:
        value: datetime | None = conn.execute(
            text("SELECT created_at FROM synthdetect_jobs WHERE id = :id"), {"id": job_id}
        ).scalar_one()
    return value


def _db_clock(engine: Engine) -> datetime:
    with engine.connect() as conn:
        value: datetime = conn.execute(text("SELECT clock_timestamp()")).scalar_one()
    return value


def test_upgrade_backfills_null_created_at_and_sets_not_null(
    alembic_cfg: Config, engine: Engine, caplog: pytest.LogCaptureFixture
) -> None:
    command.downgrade(alembic_cfg, "0066")
    assert _created_at_nullable(engine)

    queued = _seed_job(engine, status="queued", created_at=None)
    started = T0 + timedelta(minutes=5)
    finished_job = _seed_job(
        engine,
        status="succeeded",
        created_at=None,
        started_at=started,
        finished_at=started + timedelta(minutes=1),
    )
    intact = _seed_job(engine, status="succeeded", created_at=T0)

    before = _db_clock(engine)
    with caplog.at_level(logging.WARNING, logger="alembic.runtime.migration"):
        command.upgrade(alembic_cfg, "0067")
    after = _db_clock(engine)

    assert not _created_at_nullable(engine)
    # A job that never started falls back to now() (the migration's clock).
    queued_created = _created_at(engine, queued)
    assert queued_created is not None
    assert before <= queued_created <= after
    # A started job takes started_at, so started_at >= created_at stays true.
    assert _created_at(engine, finished_job) == started
    # A row that already had a creation time is untouched.
    assert _created_at(engine, intact) == T0
    assert any(
        "backfilled created_at on 2 synthdetect_jobs row(s)" in r.getMessage()
        for r in caplog.records
    )


def test_upgrade_without_null_rows_logs_nothing(
    alembic_cfg: Config, engine: Engine, caplog: pytest.LogCaptureFixture
) -> None:
    command.downgrade(alembic_cfg, "0066")
    _seed_job(engine, status="succeeded", created_at=T0)

    with caplog.at_level(logging.WARNING, logger="alembic.runtime.migration"):
        command.upgrade(alembic_cfg, "0067")

    assert not _created_at_nullable(engine)
    assert not any("backfilled created_at" in r.getMessage() for r in caplog.records)


def test_not_null_rejects_explicit_null_insert(alembic_cfg: Config, engine: Engine) -> None:
    command.upgrade(alembic_cfg, "0067")
    with pytest.raises(IntegrityError, match="created_at"):
        _seed_job(engine, status="queued", created_at=None)


def test_server_default_still_fills_created_at(alembic_cfg: Config, engine: Engine) -> None:
    command.upgrade(alembic_cfg, "0067")
    mid, rid, jid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO media_items (id, source_path) VALUES (:id, :sp)"),
            {"id": mid, "sp": f"incoming/{mid}/source"},
        )
        conn.execute(
            text(
                "INSERT INTO pipeline_runs (id, media_item_id, status, revision)"
                " VALUES (:id, :mid, 'completed', 0)"
            ),
            {"id": rid, "mid": mid},
        )
        conn.execute(
            text(
                "INSERT INTO synthdetect_jobs (id, pipeline_run_id, inference_space,"
                " calibration_policy_id) VALUES (:id, :rid, 'space', 'policy')"
            ),
            {"id": jid, "rid": rid},
        )
    assert _created_at(engine, jid) is not None


def test_offline_sql_renders_backfill_then_not_null(
    alembic_cfg: Config, capsys: pytest.CaptureFixture[str]
) -> None:
    """``alembic upgrade --sql`` renders a script: no bind, so no row count."""
    command.upgrade(alembic_cfg, "0066:0067", sql=True)
    sql = capsys.readouterr().out
    backfill = sql.index(
        "UPDATE synthdetect_jobs SET created_at = COALESCE(started_at, now())"
        " WHERE created_at IS NULL"
    )
    assert backfill < sql.index("ALTER TABLE synthdetect_jobs ALTER COLUMN created_at SET NOT NULL")


def test_downgrade_drops_not_null_and_keeps_backfilled_values(
    alembic_cfg: Config, engine: Engine
) -> None:
    command.downgrade(alembic_cfg, "0066")
    job = _seed_job(engine, status="queued", created_at=None)
    command.upgrade(alembic_cfg, "0067")
    backfilled = _created_at(engine, job)
    assert backfilled is not None

    command.downgrade(alembic_cfg, "0066")

    assert _created_at_nullable(engine)
    assert _created_at(engine, job) == backfilled
    # Nullable again: an explicit NULL insert succeeds at 0066.
    _seed_job(engine, status="queued", created_at=None)
