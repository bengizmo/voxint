"""Migration 0025 (adjudication_decisions word-range scope), up/down + CHECKs
(issue #59 slice 3).

Real alembic up/down against the shared test database (head restored in
teardown): the two nullable range columns appear, a whole-segment row keeps them
NULL, a paired range round-trips, the pair-CHECK rejects a half-set range, the
bounds-CHECK rejects an empty/negative/segment-less range, downgrade refuses
while a ranged ruling exists then drops the columns once cleared, and the ORM
model stays in lockstep with the DDL.
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

from voxint.db.models import AdjudicationDecision

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


def _seed_speaker(engine: Engine) -> uuid.UUID:
    speaker_id = uuid.uuid4()
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO speakers (id, display_name) VALUES (:id, :n)"),
            {"id": speaker_id, "n": f"Speaker {speaker_id}"},
        )
        conn.commit()
    return speaker_id


def _insert_decision(
    engine: Engine,
    run_id: uuid.UUID,
    seg_id: uuid.UUID | None,
    speaker_id: uuid.UUID,
    *,
    key: str,
    start: int | None,
    end: int | None,
) -> None:
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO adjudication_decisions"
                " (id, pipeline_run_id, diarization_label, decision, speaker_id,"
                "  operator, idempotency_key, transcript_segment_id,"
                "  start_word_index, end_word_index)"
                " VALUES (:id, :r, 'S0', 'assign', :sp, 'ben', :k, :seg, :s, :e)"
            ),
            {
                "id": uuid.uuid4(),
                "r": run_id,
                "sp": speaker_id,
                "k": key,
                "seg": seg_id,
                "s": start,
                "e": end,
            },
        )
        conn.commit()


def test_upgrade_adds_nullable_range_columns(
    engine: Engine, alembic_cfg: Config
) -> None:
    cols = {
        c["name"]: c["nullable"]
        for c in inspect(engine).get_columns("adjudication_decisions")
    }
    assert cols.get("start_word_index") is True
    assert cols.get("end_word_index") is True


def test_whole_segment_and_ranged_rows_roundtrip(
    engine: Engine, alembic_cfg: Config
) -> None:
    run_id, seg_id = _seed_segment(engine)
    speaker_id = _seed_speaker(engine)
    # Whole-segment scope keeps the range NULL (the 0018 grain, unchanged).
    _insert_decision(engine, run_id, seg_id, speaker_id, key="k-seg", start=None, end=None)
    # A paired half-open range round-trips.
    _insert_decision(engine, run_id, seg_id, speaker_id, key="k-range", start=0, end=2)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT idempotency_key, start_word_index, end_word_index"
                " FROM adjudication_decisions WHERE pipeline_run_id = :r"
                " ORDER BY idempotency_key"
            ),
            {"r": run_id},
        ).all()
    assert rows == [("k-range", 0, 2), ("k-seg", None, None)]


def test_pair_check_rejects_half_set_range(
    engine: Engine, alembic_cfg: Config
) -> None:
    run_id, seg_id = _seed_segment(engine)
    speaker_id = _seed_speaker(engine)
    with pytest.raises(IntegrityError):  # start without end
        _insert_decision(engine, run_id, seg_id, speaker_id, key="k", start=1, end=None)


def test_bounds_check_rejects_empty_and_negative_range(
    engine: Engine, alembic_cfg: Config
) -> None:
    run_id, seg_id = _seed_segment(engine)
    speaker_id = _seed_speaker(engine)
    with pytest.raises(IntegrityError):  # end == start is empty (half-open)
        _insert_decision(engine, run_id, seg_id, speaker_id, key="k1", start=2, end=2)
    with pytest.raises(IntegrityError):  # end < start
        _insert_decision(engine, run_id, seg_id, speaker_id, key="k2", start=3, end=1)
    with pytest.raises(IntegrityError):  # start < 0
        _insert_decision(engine, run_id, seg_id, speaker_id, key="k3", start=-1, end=1)


def test_bounds_check_rejects_range_without_segment(
    engine: Engine, alembic_cfg: Config
) -> None:
    run_id, _ = _seed_segment(engine)
    speaker_id = _seed_speaker(engine)
    with pytest.raises(IntegrityError):  # a range with no segment to scope
        _insert_decision(engine, run_id, None, speaker_id, key="k", start=0, end=1)


def test_downgrade_refuses_while_ranged_rows_exist_then_drops(
    engine: Engine, alembic_cfg: Config
) -> None:
    run_id, seg_id = _seed_segment(engine)
    speaker_id = _seed_speaker(engine)
    _insert_decision(engine, run_id, seg_id, speaker_id, key="k-range", start=0, end=2)
    # The guard refuses to silently promote a sub-segment ruling to whole-segment.
    with pytest.raises(RuntimeError, match="word-range reassignment"):
        command.downgrade(alembic_cfg, "0024")
    # Clear the ranged ruling deliberately, then the downgrade drops the columns.
    # The ledger is append-only (a trigger blocks DELETE); removing a permanent
    # ruling is exactly the deliberate act the guard's message calls for, so the
    # test bypasses the trigger the way an operator running maintenance SQL would.
    with engine.connect() as conn:
        conn.execute(text("SET session_replication_role = replica"))
        conn.execute(text("DELETE FROM adjudication_decisions WHERE pipeline_run_id = :r"),
                     {"r": run_id})
        conn.execute(text("SET session_replication_role = DEFAULT"))
        conn.commit()
    command.downgrade(alembic_cfg, "0024")
    cols = {c["name"] for c in inspect(engine).get_columns("adjudication_decisions")}
    assert "start_word_index" not in cols and "end_word_index" not in cols
    command.upgrade(alembic_cfg, "head")
    cols = {c["name"] for c in inspect(engine).get_columns("adjudication_decisions")}
    assert "start_word_index" in cols and "end_word_index" in cols


def test_model_matches_migrated_schema(engine: Engine, alembic_cfg: Config) -> None:
    reflected = {
        col["name"]: _pg_type(col["type"])
        for col in inspect(engine).get_columns("adjudication_decisions")
    }
    model = {
        col.name: _pg_type(col.type)
        for col in AdjudicationDecision.__table__.columns
    }
    assert reflected == model
