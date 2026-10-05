"""The sole writer and pure resolver for append-only per-word marks (#757)."""

import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session, aliased

from voxint.adjudication.ledger import (
    UNDO_KEY_PREFIX,
    ConflictingReplayError,
    WordRangeError,
)
from voxint.adjudication.turns import EffectiveMarks as EffectiveMarks
from voxint.adjudication.turns import WordMarkKey as WordMarkKey
from voxint.adjudication.undo import UndoDriftError, UndoError, UndoExpiredError, _check_grace
from voxint.db.models import PipelineRun, SegmentWordMark, TranscriptSegment, WordMarkAction
from voxint.idempotency import savepoint_adopt_or_conflict


@dataclass(frozen=True, slots=True)
class WordMarkRow:
    """The immutable fields resolution needs, without a database session."""

    id: uuid.UUID
    seq: int
    segment_id: uuid.UUID
    start_word_index: int
    end_word_index: int
    action: str
    voids_mark_id: uuid.UUID | None = None


def resolve_marks(
    rows: Iterable[WordMarkRow | SegmentWordMark],
) -> dict[WordMarkKey, Literal["keep", "omit"]]:
    """Drop undo pairs, then reduce each exact token range by highest seq."""
    history = list(rows)
    voided = {row.voids_mark_id for row in history if row.action == WordMarkAction.UNDO}
    newest: dict[WordMarkKey, WordMarkRow | SegmentWordMark] = {}
    for row in history:
        if row.action == WordMarkAction.UNDO or row.id in voided:
            continue
        key = (row.segment_id, row.start_word_index, row.end_word_index)
        if key not in newest or row.seq > newest[key].seq:
            newest[key] = row
    marks: dict[WordMarkKey, Literal["keep", "omit"]] = {}
    for key, row in newest.items():
        if row.action == WordMarkAction.KEEP:
            marks[key] = "keep"
        elif row.action == WordMarkAction.OMIT:
            marks[key] = "omit"
    return marks


def effective_marks(session: Session, run_id: uuid.UUID) -> EffectiveMarks:
    """Load one run's history in sequence order and resolve its overlay."""
    rows = session.scalars(
        select(SegmentWordMark)
        .where(SegmentWordMark.pipeline_run_id == run_id)
        .order_by(SegmentWordMark.seq)
    )
    return resolve_marks(rows)


def _payload_matches(existing: SegmentWordMark, requested: SegmentWordMark) -> bool:
    return (
        existing.pipeline_run_id == requested.pipeline_run_id
        and existing.segment_id == requested.segment_id
        and existing.start_word_index == requested.start_word_index
        and existing.end_word_index == requested.end_word_index
        and existing.action == requested.action
        and existing.voids_mark_id == requested.voids_mark_id
        and existing.operator == requested.operator
        and existing.user_id == requested.user_id
    )


def _append(session: Session, row: SegmentWordMark) -> tuple[SegmentWordMark, bool]:
    def _lookup() -> tuple[SegmentWordMark, bool] | None:
        existing = session.scalar(
            select(SegmentWordMark).where(SegmentWordMark.idempotency_key == row.idempotency_key)
        )
        return None if existing is None else (existing, True)

    def _adopt_or_conflict(
        existing: tuple[SegmentWordMark, bool],
    ) -> tuple[SegmentWordMark, bool]:
        if not _payload_matches(existing[0], row):
            raise ConflictingReplayError(row.idempotency_key)
        return existing

    def _persist() -> tuple[SegmentWordMark, bool]:
        session.add(row)
        return row, False

    return savepoint_adopt_or_conflict(
        session, lookup=_lookup, adopt_or_conflict=_adopt_or_conflict, persist=_persist
    )


def _lock_run(session: Session, run_id: uuid.UUID) -> uuid.UUID | None:
    # Match restart's run-before-segment order without excluding other writers.
    return session.scalar(
        select(PipelineRun.id).where(PipelineRun.id == run_id)
        .with_for_update(read=True, key_share=True)
    )


def _lock_segment(session: Session, segment_id: uuid.UUID) -> TranscriptSegment | None:
    # Both writers take this lock: drift checks and inserts on a segment serialize,
    # and a concurrent restart cannot delete the parent between validation and flush.
    return session.scalar(
        select(TranscriptSegment).where(TranscriptSegment.id == segment_id).with_for_update()
    )


def record_word_mark(
    session: Session,
    *,
    run_id: uuid.UUID,
    segment_id: uuid.UUID,
    start: int,
    end: int,
    action: WordMarkAction,
    operator: str,
    user_id: uuid.UUID | None,
    idempotency_key: str,
) -> tuple[SegmentWordMark, bool]:
    """Append keep/omit/clear, or adopt an identical replay; caller owns commit.

    Bounds and run ownership live here. Markability and unit alignment belong
    to the projection caller; clear must remain possible for stale marks.
    """
    if idempotency_key.startswith(UNDO_KEY_PREFIX):
        raise ConflictingReplayError(idempotency_key)
    if action not in (WordMarkAction.KEEP, WordMarkAction.OMIT, WordMarkAction.CLEAR):
        raise ValueError("record_word_mark accepts only keep, omit or clear")
    if not 0 <= start < end:
        raise WordRangeError(
            "word-range must be a non-empty half-open [start, end) with start >= 0"
        )
    if _lock_run(session, run_id) is None:
        raise WordRangeError(f"no such pipeline run {run_id}")
    segment = _lock_segment(session, segment_id)
    if segment is None:
        raise WordRangeError(f"no such transcript segment {segment_id}")
    if segment.pipeline_run_id != run_id:
        raise WordRangeError(f"segment {segment_id} does not belong to run {run_id}")
    if segment.words is None:
        raise WordRangeError("a segment without words cannot be marked")
    if end > len(segment.words):
        raise WordRangeError(
            f"word-range end {end} exceeds the segment's {len(segment.words)} words"
        )
    return _append(
        session,
        SegmentWordMark(
            pipeline_run_id=run_id,
            segment_id=segment_id,
            start_word_index=start,
            end_word_index=end,
            action=action.value,
            operator=operator,
            user_id=user_id,
            idempotency_key=idempotency_key,
        ),
    )


def undo_word_mark(
    session: Session,
    *,
    run_id: uuid.UUID,
    mark_id: uuid.UUID,
    operator: str,
    user_id: uuid.UUID | None,
    grace_seconds: float,
    now: datetime | None = None,
) -> tuple[SegmentWordMark, bool]:
    """Void a fresh, still-current ruling; identical replays bypass grace/drift."""
    if _lock_run(session, run_id) is None:
        raise UndoDriftError("the run this word mark applied to no longer exists")
    original = session.get(SegmentWordMark, mark_id)
    if original is None:
        raise UndoError(f"no word mark {mark_id}")
    if original.pipeline_run_id != run_id:
        raise UndoError("only a word mark from this run can be undone")
    if original.action == WordMarkAction.UNDO:
        raise UndoError("an undo word mark cannot be undone")
    if _lock_segment(session, original.segment_id) is None:
        raise UndoDriftError("the segment this word mark applied to no longer exists")
    key = f"{UNDO_KEY_PREFIX}{mark_id}"
    requested = SegmentWordMark(
        pipeline_run_id=run_id,
        segment_id=original.segment_id,
        start_word_index=original.start_word_index,
        end_word_index=original.end_word_index,
        action=WordMarkAction.UNDO.value,
        voids_mark_id=mark_id,
        operator=operator,
        user_id=user_id,
        idempotency_key=key,
    )
    existing = session.scalar(
        select(SegmentWordMark).where(SegmentWordMark.voids_mark_id == mark_id)
    )
    if existing is not None:
        if existing.idempotency_key != key:
            raise UndoError("this word mark is already voided")
        return _append(session, requested)

    if original.created_at is None:
        session.flush()
        session.refresh(original)
    if now is None:
        _check_grace(original.created_at, grace_seconds, "the undo grace window has passed")
    else:
        created_at = original.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        if created_at + timedelta(seconds=grace_seconds) <= now:
            raise UndoExpiredError("the undo grace window has passed")

    void = aliased(SegmentWordMark)
    newer = session.scalar(
        select(SegmentWordMark.id)
        .where(
            SegmentWordMark.pipeline_run_id == run_id,
            SegmentWordMark.segment_id == original.segment_id,
            SegmentWordMark.start_word_index == original.start_word_index,
            SegmentWordMark.end_word_index == original.end_word_index,
            SegmentWordMark.seq > original.seq,
            SegmentWordMark.action != WordMarkAction.UNDO.value,
            ~select(void.id).where(void.voids_mark_id == SegmentWordMark.id).exists(),
        )
        .limit(1)
    )
    if newer is not None:
        raise UndoDriftError("this word was marked again after this ruling")
    return _append(session, requested)
