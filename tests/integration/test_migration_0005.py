"""Migration 0005 (ACQUIRE stage + media_items.source_url), up and down.

Runs the real alembic up/down against the shared test database, then restores it
to head in the fixture teardown so the rest of the session sees a pristine
schema. These tests mutate the shared schema, so they must never run alongside a
second concurrent pytest against the same database (see the session HAZARD note).
"""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def alembic_cfg(engine: Engine) -> Iterator[Config]:
    # Depending on `engine` guarantees DATABASE_URL is exported and the schema
    # starts at head. The finally restores a pristine head regardless of outcome
    # so a mid-suite migration test never leaves the DB downgraded for the tests
    # that follow.
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


def _media_columns(engine: Engine) -> set[str]:
    with engine.connect() as conn:
        return set(
            conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns"
                    " WHERE table_name = 'media_items'"
                )
            ).scalars()
        )


def _seed_0004_run(engine: Engine) -> uuid.UUID:
    """Insert a constraint-valid 0004-shape run at a pre-acquire stage."""
    media_id = uuid.uuid4()
    run_id = uuid.uuid4()
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO media_items (id, source_path) VALUES (:id, :p)"),
            {"id": media_id, "p": f"incoming/{media_id}.wav"},
        )
        conn.execute(
            text(
                "INSERT INTO pipeline_runs (id, media_item_id, status, current_stage,"
                " revision) VALUES (:id, :mid, 'running', 'prepare', 1)"
            ),
            {"id": run_id, "mid": media_id},
        )
        conn.execute(
            text(
                "INSERT INTO stage_runs (id, pipeline_run_id, stage, status, attempt)"
                " VALUES (:id, :rid, 'prepare', 'completed', 1)"
            ),
            {"id": uuid.uuid4(), "rid": run_id},
        )
        conn.commit()
    return run_id


def _insert_acquire_stage_run(engine: Engine, run_id: uuid.UUID, attempt: int) -> None:
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO stage_runs (id, pipeline_run_id, stage, status, attempt)"
                " VALUES (:id, :rid, 'acquire', 'completed', :a)"
            ),
            {"id": uuid.uuid4(), "rid": run_id, "a": attempt},
        )
        conn.commit()


def test_migration_0005_roundtrip(alembic_cfg: Config, engine: Engine) -> None:
    # --- at 0004: no source_url column, 'acquire' is not a valid stage ---
    command.downgrade(alembic_cfg, "0004")
    run_id = _seed_0004_run(engine)
    assert "source_url" not in _media_columns(engine)
    with pytest.raises(IntegrityError):
        _insert_acquire_stage_run(engine, run_id, attempt=2)

    # --- upgrade to 0005: source_url appears, 'acquire' becomes valid ---
    command.upgrade(alembic_cfg, "0005")
    assert "source_url" in _media_columns(engine)
    with engine.connect() as conn:
        conn.execute(
            text("UPDATE media_items SET source_url = :u WHERE id = ("
                 "SELECT media_item_id FROM pipeline_runs WHERE id = :rid)"),
            {"u": "https://example.com/v", "rid": run_id},
        )
        conn.commit()
    _insert_acquire_stage_run(engine, run_id, attempt=2)  # accepted now

    # --- downgrade REFUSES while acquire *ledger history* exists ---
    with pytest.raises(RuntimeError, match="refusing to downgrade 0005"):
        command.downgrade(alembic_cfg, "0004")
    # the refusal is pre-DDL, so the schema is untouched (still 0005)
    assert "source_url" in _media_columns(engine)

    # --- and REFUSES on a *current* acquire run even with no acquire stage_run
    # (the OR's second arm: a run sitting at current_stage='acquire') ---
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM stage_runs WHERE stage = 'acquire'"))
        conn.execute(
            text("UPDATE pipeline_runs SET current_stage = 'acquire' WHERE id = :rid"),
            {"rid": run_id},
        )
        conn.commit()
    with pytest.raises(RuntimeError, match="refusing to downgrade 0005"):
        command.downgrade(alembic_cfg, "0004")
    assert "source_url" in _media_columns(engine)

    # --- once no acquire rows remain in either place, downgrade is clean ---
    with engine.connect() as conn:
        conn.execute(text("UPDATE pipeline_runs SET current_stage = 'prepare'"))
        conn.commit()
    command.downgrade(alembic_cfg, "0004")
    assert "source_url" not in _media_columns(engine)
    with pytest.raises(IntegrityError):
        _insert_acquire_stage_run(engine, run_id, attempt=3)
