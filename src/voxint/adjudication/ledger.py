"""Append-only access to the adjudication decision ledger.

This module is the only sanctioned way to write ``adjudication_decisions``.
The table itself rejects UPDATE/DELETE via a trigger (see alembic 0001);
:func:`record_decision` adds idempotent-replay semantics on top: replaying the
same idempotency key with the same payload returns the existing row, replaying
it with a *different* payload is an error, never a silent overwrite.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from voxint.adjudication.splits import word_count
from voxint.db.models import AdjudicationDecision, Decision, TranscriptSegment


class ConflictingReplayError(Exception):
    def __init__(self, idempotency_key: str) -> None:
        super().__init__(
            f"idempotency key {idempotency_key!r} was already used with a different payload"
        )
        self.idempotency_key = idempotency_key


class WordRangeError(ValueError):
    """A word-range scope is malformed or out of the parent's word bounds.

    The DB CHECK backstops pairing and ``end > start >= 0``; this is the
    in-code guard for the upper bound (``end <= word_count``) the CHECK cannot
    express, and for a range on a segment that has no derivable children."""


def _payload_matches(
    row: AdjudicationDecision,
    pipeline_run_id: uuid.UUID,
    diarization_label: str,
    decision: Decision,
    speaker_id: uuid.UUID | None,
    operator: str,
    transcript_segment_id: uuid.UUID | None,
    start_word_index: int | None,
    end_word_index: int | None,
) -> bool:
    return (
        row.pipeline_run_id == pipeline_run_id
        and row.diarization_label == diarization_label
        and row.decision == decision.value
        and row.speaker_id == speaker_id
        and row.operator == operator
        and row.transcript_segment_id == transcript_segment_id
        and row.start_word_index == start_word_index
        and row.end_word_index == end_word_index
    )


def record_decision(
    session: Session,
    *,
    pipeline_run_id: uuid.UUID,
    diarization_label: str,
    decision: Decision,
    operator: str,
    idempotency_key: str,
    speaker_id: uuid.UUID | None = None,
    transcript_segment_id: uuid.UUID | None = None,
    start_word_index: int | None = None,
    end_word_index: int | None = None,
) -> AdjudicationDecision:
    """Append a ruling; replaying an identical request returns the existing row.

    ``transcript_segment_id`` NULL is the historical label scope (rules the whole
    ``(run, label)``); non-NULL scopes the ruling to one segment (issue #54
    Phase B). ``start_word_index``/``end_word_index`` narrow that further to a
    half-open ``[start, end)`` word-range of the parent segment (issue #59 slice
    3 — reassigning a derived split child). Scope is part of the replay identity,
    so the same key replayed with a different scope (segment OR range) is a
    conflict, not a silent adopt.

    A word-range is validated here — the sole ledger writer is the invariant home
    (extend, never bypass): both indices set together, scoping a segment whose
    derivable word count bounds ``end`` (``0 <= start < end <= word_count``). The
    DB CHECK backstops pairing and ``end > start >= 0``; the ``end <= word_count``
    upper bound lives here because the CHECK cannot count a segment's words.
    """
    _validate_word_range(
        session,
        pipeline_run_id,
        transcript_segment_id,
        start_word_index,
        end_word_index,
    )
    existing = session.execute(
        select(AdjudicationDecision).where(
            AdjudicationDecision.idempotency_key == idempotency_key
        )
    ).scalar_one_or_none()
    if existing is None:
        row = AdjudicationDecision(
            pipeline_run_id=pipeline_run_id,
            diarization_label=diarization_label,
            decision=decision.value,
            speaker_id=speaker_id,
            operator=operator,
            idempotency_key=idempotency_key,
            transcript_segment_id=transcript_segment_id,
            start_word_index=start_word_index,
            end_word_index=end_word_index,
        )
        try:
            # Savepoint, not a bare flush: callers compose this into larger
            # transactions (P5 enrollment creates the speaker + embedding in
            # the same one), and losing the race here must not roll their
            # work back with it.
            with session.begin_nested():
                session.add(row)
        except IntegrityError:
            # Only the savepoint rolled back. If a concurrent writer inserted
            # the same key between our SELECT and flush, adopt their row —
            # any other constraint violation (FK, CHECK) is not a replay and
            # must not be masked as one.
            existing = session.execute(
                select(AdjudicationDecision).where(
                    AdjudicationDecision.idempotency_key == idempotency_key
                )
            ).scalar_one_or_none()
            if existing is None:
                raise
        else:
            return row
    if _payload_matches(
        existing,
        pipeline_run_id,
        diarization_label,
        decision,
        speaker_id,
        operator,
        transcript_segment_id,
        start_word_index,
        end_word_index,
    ):
        return existing
    raise ConflictingReplayError(idempotency_key)


def _validate_word_range(
    session: Session,
    pipeline_run_id: uuid.UUID,
    transcript_segment_id: uuid.UUID | None,
    start_word_index: int | None,
    end_word_index: int | None,
) -> None:
    """Guard the word-range scope before it reaches the ledger (issue #59 slice 3).

    Both-or-neither, a segment to scope, and a non-empty half-open interval whose
    ``end`` is within the parent's derivable word count. A no-range call is a
    no-op, so whole-segment and label scopes are untouched.

    The scoped segment must also belong to ``pipeline_run_id``: a ranged row keyed
    on a foreign run's segment would be permanently unreadable (``word_range_states``
    loads it under this run, but ``attributed_transcript`` only ever looks up this
    run's own segment ids) AND, being append-only, uncleanable — so the sole writer
    (the documented invariant home) refuses it here rather than trust every caller
    to have checked ownership.
    """
    if start_word_index is None and end_word_index is None:
        return
    if (start_word_index is None) != (end_word_index is None):
        raise WordRangeError(
            "start_word_index and end_word_index must be set together or both NULL"
        )
    assert start_word_index is not None and end_word_index is not None  # narrows for mypy
    if transcript_segment_id is None:
        raise WordRangeError("a word-range scope requires a transcript_segment_id")
    if not 0 <= start_word_index < end_word_index:
        raise WordRangeError(
            f"word-range must be a non-empty half-open [start, end) with start >= 0; "
            f"got [{start_word_index}, {end_word_index})"
        )
    parent = session.get(TranscriptSegment, transcript_segment_id)
    if parent is None:
        raise WordRangeError(f"no such transcript segment {transcript_segment_id}")
    if parent.pipeline_run_id != pipeline_run_id:
        raise WordRangeError(
            f"segment {transcript_segment_id} does not belong to run {pipeline_run_id}"
        )
    count = word_count(parent)
    if count is None:
        raise WordRangeError(
            "segment has no derivable words to range-scope "
            "(no aligned word timings, or its text was enhanced)"
        )
    if end_word_index > count:
        raise WordRangeError(
            f"word-range end {end_word_index} exceeds the segment's {count} words"
        )
