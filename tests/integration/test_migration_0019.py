"""Migration 0019 (transcript_segments.confidence), up/down + parity (issue #53).

Mirrors the 0016 migration test: real alembic up/down against the shared test
database (head restored in teardown), the new column asserted present/absent,
existing populated segments **preserved** across the upgrade (new column NULL),
the [0, 1] CHECK enforced, a value round-trip, and ORM/DDL parity.
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

from voxint.db.models import TranscriptSegment

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


def _seed_segment(engine: Engine) -> uuid.UUID:
    """A media_item + pipeline_run + one transcript segment written under 0018
    (pre-confidence). Returns the segment id. Mirrors test_migration_0008."""
    media_id, run_id, seg_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO media_items (id, source_path) VALUES (:id, :source_path)"),
            {"id": media_id, "source_path": f"incoming/{media_id}.wav"},
        )
        conn.execute(
            text(
                "INSERT INTO pipeline_runs (id, media_item_id, status, revision)"
                " VALUES (:id, :media_item_id, 'completed', 0)"
            ),
            {"id": run_id, "media_item_id": media_id},
        )
        conn.execute(
            text(
                "INSERT INTO transcript_segments"
                " (id, pipeline_run_id, segment_index, start_seconds, end_seconds, raw_text)"
                " VALUES (:id, :run_id, 0, 0.0, 1.0, 'hello')"
            ),
            {"id": seg_id, "run_id": run_id},
        )
        conn.commit()
    return seg_id


def test_migration_0019_roundtrip_preserves_segments(
    alembic_cfg: Config, engine: Engine
) -> None:
    command.downgrade(alembic_cfg, "0018")
    cols = {c["name"] for c in inspect(engine).get_columns("transcript_segments")}
    assert "confidence" not in cols

    seg_id = _seed_segment(engine)

    command.upgrade(alembic_cfg, "0019")
    reflected = {c["name"]: c for c in inspect(engine).get_columns("transcript_segments")}
    assert "confidence" in reflected
    assert reflected["confidence"]["nullable"] is True
    assert _pg_type(reflected["confidence"]["type"]) == "DOUBLE PRECISION"

    with engine.connect() as conn:
        # The pre-existing segment survives with confidence NULL (never fabricated).
        assert (
            conn.execute(
                text("SELECT confidence FROM transcript_segments WHERE id = :id"),
                {"id": seg_id},
            ).scalar_one()
            is None
        )
        # A valid value round-trips.
        conn.execute(
            text("UPDATE transcript_segments SET confidence = 0.42 WHERE id = :id"),
            {"id": seg_id},
        )
        conn.commit()
        assert (
            conn.execute(
                text("SELECT confidence FROM transcript_segments WHERE id = :id"),
                {"id": seg_id},
            ).scalar_one()
            == 0.42
        )

    # The [0, 1] CHECK rejects an out-of-range value.
    with pytest.raises(IntegrityError), engine.connect() as conn:
        conn.execute(
            text("UPDATE transcript_segments SET confidence = 1.5 WHERE id = :id"),
            {"id": seg_id},
        )
        conn.commit()

    command.downgrade(alembic_cfg, "0018")
    cols = {c["name"] for c in inspect(engine).get_columns("transcript_segments")}
    assert "confidence" not in cols


def test_transcript_segment_model_matches_migrated_schema(engine: Engine) -> None:
    reflected = {
        col["name"]: _pg_type(col["type"])
        for col in inspect(engine).get_columns("transcript_segments")
    }
    model = {col.name: _pg_type(col.type) for col in TranscriptSegment.__table__.columns}
    assert reflected == model
