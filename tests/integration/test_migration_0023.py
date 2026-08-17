"""Migration 0023 (segment_split_boundaries), up/down + constraints (issue #59).

Real alembic up/down against the shared test database (head restored in
teardown): the new table appears with its UNIQUE(parent, word_index) and
``word_index >= 1`` CHECK, a boundary round-trips, a duplicate cut and an
interior-violating cut are both rejected, the FK cascades on parent delete,
downgrade drops the table, and the ORM model stays in lockstep with the DDL.
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

from voxint.db.models import SegmentSplitBoundary

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


def _seed_segment(engine: Engine) -> tuple[uuid.UUID, uuid.UUID]:
    media_id, run_id, seg_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    with engine.connect() as conn:
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
        conn.execute(
            text(
                "INSERT INTO transcript_segments"
                " (id, pipeline_run_id, segment_index, start_seconds, end_seconds, raw_text)"
                " VALUES (:id, :r, 0, 0.0, 1.0, 'hello world')"
            ),
            {"id": seg_id, "r": run_id},
        )
        conn.commit()
    return run_id, seg_id


def _insert_boundary(
    engine: Engine, run_id: uuid.UUID, seg_id: uuid.UUID, word_index: int
) -> None:
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO segment_split_boundaries"
                " (id, pipeline_run_id, parent_segment_id, word_index, operator)"
                " VALUES (:id, :r, :p, :w, 'ben')"
            ),
            {"id": uuid.uuid4(), "r": run_id, "p": seg_id, "w": word_index},
        )
        conn.commit()


def test_upgrade_roundtrips_and_enforces_constraints(
    engine: Engine, alembic_cfg: Config
) -> None:
    run_id, seg_id = _seed_segment(engine)
    cols = {c["name"] for c in inspect(engine).get_columns("segment_split_boundaries")}
    assert cols == {
        "id",
        "pipeline_run_id",
        "parent_segment_id",
        "word_index",
        "operator",
        "created_at",
    }

    _insert_boundary(engine, run_id, seg_id, 1)  # interior cut round-trips
    with engine.connect() as conn:
        stored = conn.execute(
            text(
                "SELECT word_index, operator FROM segment_split_boundaries"
                " WHERE parent_segment_id = :p"
            ),
            {"p": seg_id},
        ).one()
    assert stored == (1, "ben")

    # UNIQUE(parent, word_index): the same cut cannot be recorded twice.
    with pytest.raises(IntegrityError):
        _insert_boundary(engine, run_id, seg_id, 1)

    # word_index >= 1 CHECK: a cut "before word 0" is not a split.
    with pytest.raises(IntegrityError):
        _insert_boundary(engine, run_id, seg_id, 0)


def test_fk_cascades_on_parent_delete(engine: Engine, alembic_cfg: Config) -> None:
    run_id, seg_id = _seed_segment(engine)
    _insert_boundary(engine, run_id, seg_id, 1)
    with engine.connect() as conn:
        conn.execute(
            text("DELETE FROM transcript_segments WHERE id = :p"), {"p": seg_id}
        )
        conn.commit()
        remaining = conn.execute(
            text("SELECT count(*) FROM segment_split_boundaries WHERE parent_segment_id = :p"),
            {"p": seg_id},
        ).scalar_one()
    assert remaining == 0  # cascade cleaned the boundary with its parent


def test_downgrade_drops_table(engine: Engine, alembic_cfg: Config) -> None:
    run_id, seg_id = _seed_segment(engine)
    _insert_boundary(engine, run_id, seg_id, 1)
    command.downgrade(alembic_cfg, "0022")
    assert "segment_split_boundaries" not in inspect(engine).get_table_names()
    command.upgrade(alembic_cfg, "head")
    assert "segment_split_boundaries" in inspect(engine).get_table_names()


def test_model_matches_migrated_schema(engine: Engine, alembic_cfg: Config) -> None:
    reflected = {
        col["name"]: _pg_type(col["type"])
        for col in inspect(engine).get_columns("segment_split_boundaries")
    }
    model = {
        col.name: _pg_type(col.type)
        for col in SegmentSplitBoundary.__table__.columns
    }
    assert reflected == model
