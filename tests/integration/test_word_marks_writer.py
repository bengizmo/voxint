"""Attributed, idempotent word-mark writes and compensating undo in Postgres."""

import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Barrier, Lock
from typing import Any

import pytest
from sqlalchemy import Connection, Engine, event, func, select
from sqlalchemy.engine import ExceptionContext, ExecutionContext
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.test_adjudication_ledger import seed_splittable_segment
from voxint.adjudication.ledger import ConflictingReplayError, WordRangeError
from voxint.adjudication.undo import UndoDriftError, UndoError, UndoExpiredError
from voxint.adjudication.word_marks import effective_marks, record_word_mark, undo_word_mark
from voxint.db.models import SegmentWordMark, User, WordMarkAction


def mark(
    session: Session,
    run_id: uuid.UUID,
    segment_id: uuid.UUID,
    *,
    action: WordMarkAction = WordMarkAction.OMIT,
    key: str = "mark",
    start: int = 0,
    end: int = 1,
    operator: str = "ben",
    user_id: uuid.UUID | None = None,
) -> tuple[SegmentWordMark, bool]:
    return record_word_mark(
        session,
        run_id=run_id,
        segment_id=segment_id,
        start=start,
        end=end,
        action=action,
        operator=operator,
        user_id=user_id,
        idempotency_key=key,
    )


def undo(
    session: Session, original: SegmentWordMark, *, operator: str = "ben"
) -> tuple[SegmentWordMark, bool]:
    return undo_word_mark(
        session,
        run_id=original.pipeline_run_id,
        mark_id=original.id,
        operator=operator,
        user_id=None,
        grace_seconds=300,
    )


def test_adopt_or_conflict(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, segment_id, _ = seed_splittable_segment(session)
        first, replay = mark(session, run_id, segment_id)
        assert not replay
        adopted, replay = mark(session, run_id, segment_id)
        assert replay and adopted.id == first.id
        with pytest.raises(ConflictingReplayError):
            mark(session, run_id, segment_id, action=WordMarkAction.KEEP)
        with pytest.raises(ConflictingReplayError):
            mark(session, run_id, segment_id, start=1, end=2)
        with pytest.raises(ConflictingReplayError):
            mark(session, run_id, segment_id, operator="other")
        assert session.scalar(select(func.count()).select_from(SegmentWordMark)) == 1
        session.commit()


def test_attribution_is_part_of_replay(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, segment_id, _ = seed_splittable_segment(session)
        user = User(username="ben", password_hash="hash", role="admin")
        session.add(user)
        session.flush()
        first, _ = mark(session, run_id, segment_id, user_id=user.id)
        session.commit()
        assert first.user_id == user.id and first.operator == "ben"
        assert mark(session, run_id, segment_id, user_id=user.id)[1]
        with pytest.raises(ConflictingReplayError):
            mark(session, run_id, segment_id)


def test_undo_namespace_and_action_reserved(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, segment_id, _ = seed_splittable_segment(session)
        with pytest.raises(ConflictingReplayError):
            mark(session, run_id, segment_id, key="undo:reserved")
        with pytest.raises(ValueError, match="only keep, omit or clear"):
            mark(session, run_id, segment_id, action=WordMarkAction.UNDO)


@pytest.mark.parametrize(
    ("start", "end"),
    [(-1, 1), (1, 1), (2, 1), (0, 4)],
    ids=["negative", "empty", "reversed", "out-of-range"],
)
def test_invalid_range_refused(
    session_factory: sessionmaker[Session], start: int, end: int
) -> None:
    with session_factory() as session:
        run_id, segment_id, _ = seed_splittable_segment(session)
        with pytest.raises(WordRangeError):
            mark(session, run_id, segment_id, start=start, end=end)


def test_segment_ownership_and_missing_segment(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, _, _ = seed_splittable_segment(session)
        _, foreign_segment_id, _ = seed_splittable_segment(session)
        with pytest.raises(WordRangeError, match="does not belong"):
            mark(session, run_id, foreign_segment_id)
        with pytest.raises(WordRangeError, match="no such transcript segment"):
            mark(session, run_id, uuid.uuid4())


def test_undo_missing_cross_run_and_undo_of_undo(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, segment_id, _ = seed_splittable_segment(session)
        foreign_run_id, _, _ = seed_splittable_segment(session)
        original, _ = mark(session, run_id, segment_id)
        for target_run, target_id, message in (
            (run_id, uuid.uuid4(), "no word mark"),
            (foreign_run_id, original.id, "from this run"),
        ):
            with pytest.raises(UndoError, match=message):
                undo_word_mark(
                    session,
                    run_id=target_run,
                    mark_id=target_id,
                    operator="ben",
                    user_id=None,
                    grace_seconds=300,
                )
        reversal, _ = undo(session, original)
        with pytest.raises(UndoError, match="cannot be undone"):
            undo(session, reversal)


def test_already_voided_with_another_key_refused(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, segment_id, _ = seed_splittable_segment(session)
        original, _ = mark(session, run_id, segment_id)
        session.add(
            SegmentWordMark(
                pipeline_run_id=run_id,
                segment_id=segment_id,
                start_word_index=0,
                end_word_index=1,
                action="undo",
                voids_mark_id=original.id,
                operator="ben",
                idempotency_key="legacy-undo",
            )
        )
        session.flush()
        with pytest.raises(UndoError, match="already voided"):
            undo(session, original)


@pytest.mark.parametrize("elapsed", [300, 301], ids=["boundary", "expired"])
def test_undo_grace_window(session_factory: sessionmaker[Session], elapsed: int) -> None:
    with session_factory() as session:
        run_id, segment_id, _ = seed_splittable_segment(session)
        original, _ = mark(session, run_id, segment_id)
        with pytest.raises(UndoExpiredError):
            undo_word_mark(
                session,
                run_id=run_id,
                mark_id=original.id,
                operator="ben",
                user_id=None,
                grace_seconds=300,
                now=original.created_at + timedelta(seconds=elapsed),
            )


def test_undo_default_clock_expiry(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, segment_id, _ = seed_splittable_segment(session)
        original, _ = mark(session, run_id, segment_id)
        with pytest.raises(UndoExpiredError):
            undo_word_mark(
                session,
                run_id=run_id,
                mark_id=original.id,
                operator="ben",
                user_id=None,
                grace_seconds=0,
            )


def test_undo_drift_and_voided_newer_mark(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, segment_id, _ = seed_splittable_segment(session)
        original, _ = mark(session, run_id, segment_id)
        newer, _ = mark(session, run_id, segment_id, action=WordMarkAction.KEEP, key="newer")
        with pytest.raises(UndoDriftError):
            undo(session, original)
        undo(session, newer)
        undo(session, original)
        assert effective_marks(session, run_id) == {}


def test_undo_replay_bypasses_expiry_and_drift(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, segment_id, _ = seed_splittable_segment(session)
        original, _ = mark(session, run_id, segment_id)
        reversal, replay = undo(session, original)
        assert not replay and reversal.idempotency_key == f"undo:{original.id}"
        mark(session, run_id, segment_id, action=WordMarkAction.KEEP, key="newer")
        session.commit()
        adopted, replay = undo_word_mark(
            session,
            run_id=run_id,
            mark_id=original.id,
            operator="ben",
            user_id=None,
            grace_seconds=300,
            now=original.created_at + timedelta(days=1),
        )
        assert replay and adopted.id == reversal.id
        with pytest.raises(ConflictingReplayError):
            undo(session, original, operator="other")


@pytest.mark.parametrize("action", [WordMarkAction.OMIT, WordMarkAction.CLEAR])
def test_undo_restores_previous_effective_state(
    session_factory: sessionmaker[Session], action: WordMarkAction
) -> None:
    with session_factory() as session:
        run_id, segment_id, _ = seed_splittable_segment(session)
        foreign_run_id, foreign_segment_id, _ = seed_splittable_segment(session)
        mark(session, run_id, segment_id, action=WordMarkAction.KEEP, key="previous")
        mark(session, foreign_run_id, foreign_segment_id, key="foreign")
        before = effective_marks(session, run_id)
        original, _ = mark(session, run_id, segment_id, action=action)
        assert effective_marks(session, run_id) != before
        mark(session, run_id, segment_id, start=1, end=2, key="other-unit")
        undo(session, original)
        assert effective_marks(session, run_id) == {**before, (segment_id, 1, 2): "omit"}
        assert effective_marks(session, foreign_run_id) == {(foreign_segment_id, 0, 1): "omit"}
        session.commit()


def test_concurrent_key_conflict_recovers_savepoint(
    session_factory: sessionmaker[Session], engine: Engine
) -> None:
    with session_factory() as session:
        run_id, segment_id, _ = seed_splittable_segment(session)
        other_run_id, other_segment_id, _ = seed_splittable_segment(session)
        session.commit()
    barrier = Barrier(2)
    guard = Lock()
    lookups = 0
    violations: list[str] = []

    def synchronize_lookup(
        conn: Connection,
        cursor: Any,
        statement: str,
        parameters: Any,
        context: ExecutionContext,
        executemany: bool,
    ) -> None:
        nonlocal lookups
        if (
            not statement.startswith("SELECT")
            or "segment_word_marks.idempotency_key =" not in statement
        ):
            return
        with guard:
            lookups += 1
            first_lookup = lookups <= 2
        if first_lookup:
            barrier.wait(timeout=10)

    def capture_violation(context: ExceptionContext) -> None:
        violations.append(str(context.original_exception))

    def write(target: tuple[uuid.UUID, uuid.UUID]) -> bool:
        with session_factory() as session:
            try:
                _, replay = mark(session, *target, key="racing-key")
                assert not replay
                session.commit()
                return True
            except ConflictingReplayError:
                # The failed insert rolled back its savepoint, not this transaction.
                assert session.scalar(select(1)) == 1
                return False

    event.listen(engine, "after_cursor_execute", synchronize_lookup)
    event.listen(engine, "handle_error", capture_violation)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(
                pool.map(write, [(run_id, segment_id), (other_run_id, other_segment_id)])
            )
    finally:
        event.remove(engine, "after_cursor_execute", synchronize_lookup)
        event.remove(engine, "handle_error", capture_violation)
    assert sorted(outcomes) == [False, True]
    assert any("segment_word_marks_idempotency_key" in error for error in violations)
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(SegmentWordMark)) == 1


def test_concurrent_identical_replay(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, segment_id, _ = seed_splittable_segment(session)
        session.commit()
    barrier = Barrier(2)

    def write() -> tuple[uuid.UUID, bool]:
        with session_factory() as session:
            barrier.wait(timeout=10)
            row, replay = mark(session, run_id, segment_id)
            session.commit()
            return row.id, replay

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(write) for _ in range(2)]
        outcomes = [future.result(timeout=15) for future in futures]
    assert outcomes[0][0] == outcomes[1][0]
    assert sorted(replay for _, replay in outcomes) == [False, True]
