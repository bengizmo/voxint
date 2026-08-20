"""Migration 0030 (pipeline_runs.sidecar frozen YAML sidecar snapshot), up/down.

Real alembic up/down against the shared test database (head restored in
teardown): the column appears/disappears, a run row that predates the migration
reads NULL (no data pass — NULL is the honest "no sidecar existed at ingest"),
a JSON-object snapshot round-trips, and the object-shape check constraint
rejects a non-object value. Issue #104.
"""

import json
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import IntegrityError

REPO_ROOT = Path(__file__).resolve().parents[2]


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


def _run_columns(engine: Engine) -> set[str]:
    return {c["name"] for c in inspect(engine).get_columns("pipeline_runs")}


def _seed_pre_migration_run(engine: Engine) -> str:
    """Insert a media_item → pipeline_run row via RAW SQL.

    Raw SQL (not the ORM) because the ORM model already carries the new column,
    which does not exist at revision 0029 — an ORM insert would name it.
    Returns the run id (hex).
    """
    media_id = uuid.uuid4().hex
    run_id = uuid.uuid4().hex
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO media_items (id, source_path) VALUES (:id, :p)"),
            {"id": media_id, "p": f"incoming/{media_id}.wav"},
        )
        conn.execute(
            text(
                "INSERT INTO pipeline_runs (id, media_item_id, status, revision)"
                " VALUES (:id, :m, 'queued', 0)"
            ),
            {"id": run_id, "m": media_id},
        )
        conn.commit()
    return run_id


def test_migration_0030_column_null_backfill_and_roundtrip(
    alembic_cfg: Config, engine: Engine
) -> None:
    # Before: at 0029 the column does not exist.
    command.downgrade(alembic_cfg, "0029")
    assert "sidecar" not in _run_columns(engine)

    run_id = _seed_pre_migration_run(engine)

    command.upgrade(alembic_cfg, "0030")
    assert "sidecar" in _run_columns(engine)

    _select = "SELECT sidecar FROM pipeline_runs WHERE id = :id"
    with engine.connect() as conn:
        # The pre-existing run reads NULL — no sidecar existed at its ingest.
        assert conn.execute(text(_select), {"id": run_id}).scalar_one() is None

        # A whole-mapping snapshot (applied + reference-only keys) round-trips.
        snapshot = {
            "title": "Interview with Jane Doe",
            "speakers": ["Jane Doe"],
            "notes": "spring conference",
            "content_item_id": 12345,
            "published": "2026-01-15",
        }
        conn.execute(
            text(
                "UPDATE pipeline_runs SET sidecar = CAST(:s AS jsonb) WHERE id = :id"
            ),
            {"s": json.dumps(snapshot), "id": run_id},
        )
        conn.commit()
        assert conn.execute(text(_select), {"id": run_id}).scalar_one() == snapshot

    # The check constraint pins the value to a JSON object (or NULL).
    with engine.connect() as conn, pytest.raises(IntegrityError):
        conn.execute(
            text(
                "UPDATE pipeline_runs SET sidecar = CAST(:s AS jsonb) WHERE id = :id"
            ),
            {"s": json.dumps(["not", "an", "object"]), "id": run_id},
        )

    # Downgrade drops the column (and its constraint) again.
    command.downgrade(alembic_cfg, "0029")
    assert "sidecar" not in _run_columns(engine)
