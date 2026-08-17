"""Migration 0019 (segment_review_states), up/down + constraints (issues #53/#58).

Real alembic up/down against the shared test database (head restored in
teardown): the table + its indexes appear/disappear, the paired-shape and
length CHECKs are enforced, the FK cascades on segment delete, a verified/
corrected row round-trips, and ORM/DDL parity holds.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import IntegrityError

from voxint.db.search import CORRECTED_FTS_INDEX_NAME

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


def _seed_segment(engine: Engine) -> uuid.UUID:
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
                " VALUES (:id, :r, 0, 0.0, 1.0, 'hello')"
            ),
            {"id": seg_id, "r": run_id},
        )
        conn.commit()
    return seg_id


def _run_of(engine: Engine, seg_id: uuid.UUID) -> uuid.UUID:
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT pipeline_run_id FROM transcript_segments WHERE id = :id"),
            {"id": seg_id},
        ).scalar_one()


def test_migration_0019_table_and_constraints(alembic_cfg: Config, engine: Engine) -> None:
    command.downgrade(alembic_cfg, "0018")
    assert "segment_review_states" not in inspect(engine).get_table_names()

    seg_id = _seed_segment(engine)
    run_id = _run_of(engine, seg_id)

    command.upgrade(alembic_cfg, "0019")
    insp = inspect(engine)
    assert "segment_review_states" in insp.get_table_names()
    index_names = {ix["name"] for ix in insp.get_indexes("segment_review_states")}
    assert "ix_segment_review_states_run" in index_names
    assert CORRECTED_FTS_INDEX_NAME in index_names

    # A verified + corrected row round-trips.
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO segment_review_states (transcript_segment_id,"
                " pipeline_run_id, verified_at, corrected_text, corrected_at)"
                " VALUES (:s, :r, now(), 'fixed text', now())"
            ),
            {"s": seg_id, "r": run_id},
        )
        conn.commit()
        assert (
            conn.execute(
                text(
                    "SELECT corrected_text FROM segment_review_states"
                    " WHERE transcript_segment_id = :s"
                ),
                {"s": seg_id},
            ).scalar_one()
            == "fixed text"
        )

    # Paired-shape CHECK: corrected_text without corrected_at is rejected.
    with pytest.raises(IntegrityError), engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO segment_review_states"
                " (transcript_segment_id, pipeline_run_id, corrected_text)"
                " VALUES (:s, :r, 'x')"
            ),
            {"s": uuid.uuid4(), "r": run_id},
        )
        conn.commit()

    # FK ON DELETE CASCADE: deleting the segment removes its review row.
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM transcript_segments WHERE id = :id"), {"id": seg_id})
        conn.commit()
        assert (
            conn.execute(text("SELECT count(*) FROM segment_review_states")).scalar_one() == 0
        )

    command.downgrade(alembic_cfg, "0018")
    assert "segment_review_states" not in inspect(engine).get_table_names()
