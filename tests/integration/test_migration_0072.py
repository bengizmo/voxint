"""Word-mark schema, populated migration round trip and append-only cascades."""

import uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import CheckConstraint, Engine, delete, inspect, select, text, update
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.test_adjudication_ledger import seed_splittable_segment
from tests.integration.test_word_marks_writer import mark, undo
from voxint.db.models import SegmentWordMark, TranscriptSegment, User


def test_populated_migration_roundtrip(
    engine: Engine, session_factory: sessionmaker[Session]
) -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "alembic"))
    try:
        command.downgrade(cfg, "0071")
        assert "segment_word_marks" not in inspect(engine).get_table_names()
        with session_factory() as session:
            run_id, segment_id, _ = seed_splittable_segment(session)
            session.commit()
        command.upgrade(cfg, "0072")
        with session_factory() as session:
            assert session.get(TranscriptSegment, segment_id) is not None
            original, _ = mark(session, run_id, segment_id)
            reversal, _ = undo(session, original)
            assert reversal.seq > original.seq
            session.commit()
        command.downgrade(cfg, "0071")
        assert "segment_word_marks" not in inspect(engine).get_table_names()
        with engine.connect() as conn:
            assert (
                conn.scalar(
                    text(
                        "SELECT count(*) FROM pg_proc "
                        "WHERE proname = 'segment_word_marks_append_only'"
                    )
                )
                == 0
            )
        command.upgrade(cfg, "0072")
        with session_factory() as session:
            assert session.get(TranscriptSegment, segment_id) is not None
            assert list(session.scalars(select(SegmentWordMark))) == []
            mark(session, run_id, segment_id)
            session.commit()
    finally:
        command.upgrade(cfg, "head")


def test_model_schema_parity(engine: Engine) -> None:
    table = SegmentWordMark.__table__
    inspector = inspect(engine)
    columns = {c["name"]: c for c in inspector.get_columns(table.name)}
    assert set(columns) == set(table.columns.keys())
    for column in table.columns:
        assert columns[column.name]["nullable"] == column.nullable
        assert columns[column.name]["type"].compile(dialect=postgresql.dialect()) == (
            column.type.compile(dialect=postgresql.dialect())
        )
    assert columns["seq"]["identity"]["always"] is True
    assert columns["created_at"]["default"] == "now()"
    assert {c["name"] for c in inspector.get_check_constraints(table.name)} == {
        c.name for c in table.constraints if isinstance(c, CheckConstraint)
    }
    assert {c["name"] for c in inspector.get_unique_constraints(table.name)} == {
        "segment_word_marks_seq_key",
        "segment_word_marks_idempotency_key",
    }
    indexes = {i["name"]: i for i in inspector.get_indexes(table.name)}
    assert indexes["ix_segment_word_marks_run_segment_seq"]["column_names"] == [
        "pipeline_run_id",
        "segment_id",
        "seq",
    ]
    assert indexes["ix_segment_word_marks_voids"]["unique"]
    assert "voids_mark_id IS NOT NULL" in str(
        indexes["ix_segment_word_marks_voids"]["dialect_options"]["postgresql_where"]
    )
    fks = {tuple(f["constrained_columns"]): f for f in inspector.get_foreign_keys(table.name)}
    assert fks[("segment_id",)]["options"]["ondelete"] == "CASCADE"
    assert fks[("voids_mark_id",)]["options"]["ondelete"] == "CASCADE"
    with engine.connect() as conn:
        definition = conn.scalar(
            text(
                "SELECT pg_get_triggerdef(oid) FROM pg_trigger "
                "WHERE tgname = 'segment_word_marks_append_only_trigger'"
            )
        )
        assert "BEFORE DELETE OR UPDATE" in definition
        assert "FOR EACH ROW" in definition


@pytest.mark.parametrize(
    ("constraint", "changes"),
    [
        ("word_range_bounds", {"start_word_index": -1}),
        ("word_range_bounds", {"end_word_index": 0}),
        ("action", {"action": "invalid"}),
        ("undo", {"action": "undo"}),
        ("undo", {"voids_mark_id": "original"}),
        ("no_self_void", {"action": "undo", "voids_mark_id": "self"}),
        ("user_not_system", {"operator": "system:test", "user_id": "user"}),
        ("system_not_user", {"operator": "system:test", "user_id": "user"}),
    ],
    ids=[
        "negative-start",
        "empty-range",
        "bad-action",
        "undo-without-target",
        "target-without-undo",
        "self-void",
        "user-not-system",
        "system-not-user",
    ],
)
def test_checks_reject_bad_rows(
    session_factory: sessionmaker[Session], constraint: str, changes: dict[str, object]
) -> None:
    with session_factory() as session:
        run_id, segment_id, _ = seed_splittable_segment(session)
        original, _ = mark(session, run_id, segment_id)
        user = User(username="ben", password_hash="hash", role="admin")
        session.add(user)
        session.flush()
        row_id = uuid.uuid4()
        values: dict[str, object] = {
            "id": row_id,
            "pipeline_run_id": run_id,
            "segment_id": segment_id,
            "start_word_index": 0,
            "end_word_index": 1,
            "action": "keep",
            "operator": "ben",
            "idempotency_key": "bad",
        }
        replacements = {"original": original.id, "self": row_id, "user": user.id}
        values.update(
            {
                key: replacements.get(value, value) if isinstance(value, str) else value
                for key, value in changes.items()
            }
        )
        name = f"segment_word_marks_{constraint}_check"
        with pytest.raises(IntegrityError, match=name), session.begin_nested():
            # The inherited system/user pair is logically redundant; isolate each
            # constraint to prove that both enforce the ledger's exact grammar.
            if constraint in ("user_not_system", "system_not_user"):
                other = "system_not_user" if constraint == "user_not_system" else "user_not_system"
                session.execute(
                    text(
                        "ALTER TABLE segment_word_marks DROP CONSTRAINT "
                        f"segment_word_marks_{other}_check"
                    )
                )
            session.execute(SegmentWordMark.__table__.insert().values(**values))


def test_identity_and_one_undo_per_mark(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, segment_id, _ = seed_splittable_segment(session)
        original, _ = mark(session, run_id, segment_id)
        newer, _ = mark(session, run_id, segment_id, key="newer", start=1, end=2)
        reversal, _ = undo(session, original)
        assert 0 < original.seq < newer.seq < reversal.seq
        with (
            pytest.raises(IntegrityError, match="ix_segment_word_marks_voids"),
            session.begin_nested(),
        ):
            session.add(
                SegmentWordMark(
                    pipeline_run_id=run_id,
                    segment_id=segment_id,
                    start_word_index=0,
                    end_word_index=1,
                    action="undo",
                    voids_mark_id=original.id,
                    operator="ben",
                    idempotency_key="second-undo",
                )
            )
        with pytest.raises(DBAPIError, match="GENERATED ALWAYS"), session.begin_nested():
            session.execute(
                SegmentWordMark.__table__.insert().values(
                    id=uuid.uuid4(),
                    seq=original.seq + 100,
                    pipeline_run_id=run_id,
                    segment_id=segment_id,
                    start_word_index=0,
                    end_word_index=1,
                    action="keep",
                    operator="ben",
                    idempotency_key="explicit-seq",
                )
            )
        session.commit()


def test_updates_and_direct_deletes_refused(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, segment_id, _ = seed_splittable_segment(session)
        original, _ = mark(session, run_id, segment_id)
        reversal, _ = undo(session, original)
        session.commit()
        for row in (original, reversal):
            with pytest.raises(DBAPIError, match="UPDATE blocked"), session.begin_nested():
                session.execute(
                    update(SegmentWordMark)
                    .where(SegmentWordMark.id == row.id)
                    .values(operator=row.operator)
                )
            with pytest.raises(DBAPIError, match="DELETE blocked"), session.begin_nested():
                session.execute(delete(SegmentWordMark).where(SegmentWordMark.id == row.id))


def test_segment_delete_cascades_mark_and_undo(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, segment_id, _ = seed_splittable_segment(session)
        original, _ = mark(session, run_id, segment_id)
        reversal, _ = undo(session, original)
        session.commit()
        ids = [original.id, reversal.id]
        session.execute(delete(TranscriptSegment).where(TranscriptSegment.id == segment_id))
        session.commit()
        assert session.get(TranscriptSegment, segment_id) is None
        assert (
            list(session.scalars(select(SegmentWordMark.id).where(SegmentWordMark.id.in_(ids))))
            == []
        )
