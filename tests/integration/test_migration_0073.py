"""Migration 0073 (LLM clean-up variant), up/down and the DB invariants (#758).

Real alembic against the shared test database (head restored in teardown):
the tables, CHECKs and one-active index hold, the immutability trigger allows
only a one-time stamp of a newer same-run supersessor, downgrade to 0072
drops everything and upgrade restores it.
"""

import json
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, Engine, inspect, text
from sqlalchemy.exc import DBAPIError, IntegrityError

REPO_ROOT = Path(__file__).resolve().parents[2]
_TABLES = ("run_cleanups", "cleanup_jobs")
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


def _seed_run(conn: Connection) -> uuid.UUID:
    media_id, run_id = uuid.uuid4(), uuid.uuid4()
    conn.execute(
        text("INSERT INTO media_items (id, source_path) VALUES (:id, :p)"),
        {"id": media_id, "p": f"incoming/{media_id}.wav"},
    )
    conn.execute(
        text(
            "INSERT INTO pipeline_runs (id, media_item_id, status, revision)"
            " VALUES (:id, :m, 'completed', 0)"
        ),
        {"id": run_id, "m": media_id},
    )
    return run_id


def _insert_cleanup(
    conn: Connection, run_id: uuid.UUID, generation: int, **overrides: object,
) -> uuid.UUID:
    values: dict[str, object] = {
        "id": uuid.uuid4(), "r": run_id, "g": generation, "lines": "[]",
        "counts": "{}", "config": "{}", "h": _HASH, "k": str(uuid.uuid4()), "d": _HASH,
        "s": None,
    } | overrides
    conn.execute(
        text(
            "INSERT INTO run_cleanups (id, pipeline_run_id, generation, lines, counts,"
            " config, payload_schema_version, producer, producer_version, model,"
            " source_content_hash, idempotency_key, replay_digest,"
            " superseded_by_cleanup_id, started_at, completed_at)"
            " VALUES (:id, :r, :g, CAST(:lines AS jsonb), CAST(:counts AS jsonb),"
            " CAST(:config AS jsonb), 1, 'p', '1', 'm', :h, :k, :d, :s, now(), now())"
        ),
        values,
    )
    return values["id"]  # type: ignore[return-value]


def _insert_job(
    conn: Connection,
    run_id: uuid.UUID,
    *,
    status: str = "queued",
    cleanup_id: uuid.UUID | None = None,
) -> None:
    conn.execute(
        text(
            "INSERT INTO cleanup_jobs (id, pipeline_run_id, status, cancel_requested,"
            " config, source_content_hash, cleanup_id, started_at, finished_at)"
            " VALUES (:id, :r, :s, false, '{}'::jsonb, :h, :c,"
            " CASE WHEN :s IN ('queued') THEN NULL ELSE now() END,"
            " CASE WHEN :s IN ('queued', 'running') THEN NULL ELSE now() END)"
        ),
        {"id": uuid.uuid4(), "r": run_id, "s": status, "h": _HASH, "c": cleanup_id},
    )


def test_job_constraints(engine: Engine, alembic_cfg: Config) -> None:
    with engine.connect() as conn:
        run_id = _seed_run(conn)
        _insert_job(conn, run_id)
        conn.commit()
    # One active job per run; a terminal job frees the slot.
    for status in ("queued", "running"):
        with engine.connect() as conn, pytest.raises(IntegrityError):
            _insert_job(conn, run_id, status=status)
    with engine.connect() as conn:
        _insert_job(conn, run_id, status="failed")
        conn.execute(
            text("UPDATE cleanup_jobs SET status = 'cancelled', started_at = now(),"
                 " finished_at = now() WHERE status = 'queued'")
        )
        _insert_job(conn, run_id, status="running")
        cleanup_id = _insert_cleanup(conn, run_id, 1)
        _insert_job(conn, run_id, status="succeeded", cleanup_id=cleanup_id)
        conn.commit()
    for bad in (
        {"status": "sideways"},
        {"status": "failed", "cleanup_id": cleanup_id},
        {"status": "succeeded"},
    ):
        with engine.connect() as conn, pytest.raises(IntegrityError):
            _insert_job(conn, run_id, **bad)


@pytest.mark.parametrize(
    "overrides",
    [
        {"g": 0},
        {"lines": "{}"},
        {"counts": "[]"},
        {"config": "[]"},
        {"h": "not-a-hash"},
        {"d": "A" * 64},
        {"k": " "},
    ],
)
def test_cleanup_checks(engine: Engine, alembic_cfg: Config, overrides: dict[str, object]) -> None:
    with engine.connect() as conn:
        run_id = _seed_run(conn)
        conn.commit()
    with engine.connect() as conn, pytest.raises(IntegrityError):
        _insert_cleanup(conn, run_id, 1, **overrides)


def test_cleanup_uniques(engine: Engine, alembic_cfg: Config) -> None:
    with engine.connect() as conn:
        run_id = _seed_run(conn)
        _insert_cleanup(conn, run_id, 1, k="key")
        conn.commit()
    with engine.connect() as conn, pytest.raises(IntegrityError):
        _insert_cleanup(conn, run_id, 1)
    with engine.connect() as conn, pytest.raises(IntegrityError):
        _insert_cleanup(conn, run_id, 2, k="key")


def test_immutability_trigger(engine: Engine, alembic_cfg: Config) -> None:
    with engine.connect() as conn:
        run_id, other_run = _seed_run(conn), _seed_run(conn)
        first = _insert_cleanup(conn, run_id, 1, lines=json.dumps([{"i": 0}]))
        second = _insert_cleanup(conn, run_id, 2)
        third = _insert_cleanup(conn, run_id, 3)
        foreign = _insert_cleanup(conn, other_run, 5)
        conn.commit()

    def refused(sql: str, params: dict[str, object], message: str) -> None:
        with engine.connect() as conn, pytest.raises(DBAPIError, match=message):
            conn.execute(text(sql), params)

    with engine.connect() as conn, pytest.raises(DBAPIError, match="born unsuperseded"):
        _insert_cleanup(conn, run_id, 9, s=first)
    for column, value in (
        ("lines", "'[]'::jsonb"),
        ("counts", "'{\"x\": 1}'::jsonb"),
        ("config", "'{\"x\": 1}'::jsonb"),
        ("model", "'other'"), ("generation", "7"), ("replay_digest", f"'{'1' * 64}'"),
        ("completed_at", "now() + interval '1 day'"),
    ):
        refused(
            f"UPDATE run_cleanups SET {column} = {value} WHERE id = :id",
            {"id": first}, "content is immutable",
        )
    refused("DELETE FROM run_cleanups WHERE id = :id", {"id": first}, "DELETE blocked")
    stamp = "UPDATE run_cleanups SET superseded_by_cleanup_id = :to WHERE id = :id"
    refused(stamp, {"to": first, "id": second}, "unsuperseded generation of the same run")
    refused(stamp, {"to": foreign, "id": first}, "unsuperseded generation of the same run")
    with engine.connect() as conn:
        conn.execute(text(stamp), {"to": third, "id": second})
        conn.commit()
    # Newer and same run, but no longer the head: refused.
    refused(stamp, {"to": second, "id": first}, "newer, unsuperseded generation")
    with engine.connect() as conn:
        conn.execute(text(stamp), {"to": third, "id": first})
        conn.commit()
    with engine.connect() as conn:
        fourth = _insert_cleanup(conn, run_id, 4)
        conn.commit()
    refused(stamp, {"to": fourth, "id": first}, "write-once")
    with engine.connect() as conn:
        # Re-stamping the same value is a no-op, not a change.
        conn.execute(text(stamp), {"to": third, "id": first})
        assert conn.execute(
            text("SELECT superseded_by_cleanup_id FROM run_cleanups WHERE id = :id"), {"id": first}
        ).scalar_one() == third


def test_downgrade_drops_and_upgrade_restores(engine: Engine, alembic_cfg: Config) -> None:
    command.downgrade(alembic_cfg, "0072")
    inspector = inspect(engine)
    for table in _TABLES:
        assert not inspector.has_table(table)
    with engine.connect() as conn:
        assert conn.execute(
            text("SELECT count(*) FROM pg_proc WHERE proname = 'run_cleanups_content_immutable'")
        ).scalar_one() == 0
    command.upgrade(alembic_cfg, "0073")
    inspector = inspect(engine)
    for table in _TABLES:
        assert inspector.has_table(table)
    indexes = {index["name"] for index in inspector.get_indexes("cleanup_jobs")}
    assert "cleanup_jobs_one_active_per_run" in indexes
