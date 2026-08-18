"""The adjudication ledger's append-only + idempotent-replay contract."""

import uuid

import pytest
from sqlalchemy import delete, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session, sessionmaker

from voxint.adjudication.ledger import (
    ConflictingReplayError,
    WordRangeError,
    record_decision,
)
from voxint.db.models import (
    AdjudicationDecision,
    Decision,
    MediaItem,
    PipelineRun,
    RunStatus,
    Speaker,
    TranscriptSegment,
)
from voxint.domain_packs.base import load_default
from voxint.pipeline.engine import submit


def seed(session: Session) -> tuple[uuid.UUID, uuid.UUID]:
    media = MediaItem(source_path=f"/data/media/adj-{uuid.uuid4()}.wav")
    speaker = Speaker(display_name=f"Speaker {uuid.uuid4()}")
    session.add_all([media, speaker])
    session.flush()
    run = submit(session, media.id, domain_pack=load_default().to_mapping())
    return run.id, speaker.id


# A splittable three-word segment (tokens reconcatenate to raw_text exactly), so
# the ledger's word-range bound (end <= word_count) has real words to count.
_WORDS = [
    {"start": 0.0, "end": 0.4, "word": "Hello"},
    {"start": 0.5, "end": 0.9, "word": " there"},
    {"start": 1.0, "end": 1.4, "word": " world"},
]


def seed_splittable_segment(
    session: Session,
) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    """A completed run with one splittable 3-word segment. Returns
    ``(run_id, segment_id, speaker_id)``."""
    media = MediaItem(source_path=f"/data/media/adj-{uuid.uuid4()}.wav")
    speaker = Speaker(display_name=f"Speaker {uuid.uuid4()}")
    session.add_all([media, speaker])
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()
    seg = TranscriptSegment(
        pipeline_run_id=run.id,
        segment_index=0,
        start_seconds=0.0,
        end_seconds=2.0,
        raw_text="Hello there world",
        diarization_label="SPEAKER_00",
        words=_WORDS,
    )
    session.add(seg)
    session.flush()
    return run.id, seg.id, speaker.id


def test_identical_replay_returns_existing_row(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, speaker_id = seed(session)
        first = record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="SPEAKER_00",
            decision=Decision.ASSIGN,
            speaker_id=speaker_id,
            operator="ben",
            idempotency_key="slot-1",
        )
        replay = record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="SPEAKER_00",
            decision=Decision.ASSIGN,
            speaker_id=speaker_id,
            operator="ben",
            idempotency_key="slot-1",
        )
        assert replay.id == first.id
        session.commit()


def test_conflicting_replay_is_rejected(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, speaker_id = seed(session)
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="SPEAKER_00",
            decision=Decision.ASSIGN,
            speaker_id=speaker_id,
            operator="ben",
            idempotency_key="slot-1",
        )
        with pytest.raises(ConflictingReplayError):
            record_decision(
                session,
                pipeline_run_id=run_id,
                diarization_label="SPEAKER_00",
                decision=Decision.EXCLUDE,
                operator="ben",
                idempotency_key="slot-1",
            )
        session.rollback()


def test_update_and_delete_are_blocked_by_the_database(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, speaker_id = seed(session)
        row = record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="SPEAKER_00",
            decision=Decision.ASSIGN,
            speaker_id=speaker_id,
            operator="ben",
            idempotency_key="slot-immutable",
        )
        session.commit()
        row_id = row.id

    with session_factory() as session:
        with pytest.raises(DBAPIError, match="append-only"):
            session.execute(
                update(AdjudicationDecision)
                .where(AdjudicationDecision.id == row_id)
                .values(operator="mallory")
            )
        session.rollback()
        with pytest.raises(DBAPIError, match="append-only"):
            session.execute(
                delete(AdjudicationDecision).where(AdjudicationDecision.id == row_id)
            )
        session.rollback()


# --- Word-range scope (issue #59 slice 3) -----------------------------------


def test_word_range_scope_roundtrips(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, seg_id, speaker_id = seed_splittable_segment(session)
        row = record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="SPEAKER_00",
            decision=Decision.ASSIGN,
            speaker_id=speaker_id,
            operator="ben",
            idempotency_key="range-1",
            transcript_segment_id=seg_id,
            start_word_index=0,
            end_word_index=2,
        )
        assert (row.start_word_index, row.end_word_index) == (0, 2)
        session.commit()


def test_identical_ranged_replay_returns_existing_row(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, seg_id, speaker_id = seed_splittable_segment(session)
        kwargs = dict(
            pipeline_run_id=run_id,
            diarization_label="SPEAKER_00",
            decision=Decision.ASSIGN,
            speaker_id=speaker_id,
            operator="ben",
            idempotency_key="range-replay",
            transcript_segment_id=seg_id,
            start_word_index=0,
            end_word_index=2,
        )
        first = record_decision(session, **kwargs)  # type: ignore[arg-type]
        replay = record_decision(session, **kwargs)  # type: ignore[arg-type]
        assert replay.id == first.id  # same range replays as a no-op
        session.commit()


def test_same_key_different_range_is_a_conflict(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, seg_id, speaker_id = seed_splittable_segment(session)
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="SPEAKER_00",
            decision=Decision.ASSIGN,
            speaker_id=speaker_id,
            operator="ben",
            idempotency_key="range-key",
            transcript_segment_id=seg_id,
            start_word_index=0,
            end_word_index=2,
        )
        # The range is part of the replay identity: the same key with a different
        # range is a conflict, never a silent adopt.
        with pytest.raises(ConflictingReplayError):
            record_decision(
                session,
                pipeline_run_id=run_id,
                diarization_label="SPEAKER_00",
                decision=Decision.ASSIGN,
                speaker_id=speaker_id,
                operator="ben",
                idempotency_key="range-key",
                transcript_segment_id=seg_id,
                start_word_index=2,
                end_word_index=3,
            )
        session.rollback()


def test_range_end_beyond_word_count_is_rejected(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, seg_id, speaker_id = seed_splittable_segment(session)
        # The segment has 3 words; end=4 exceeds the count the DB CHECK cannot see.
        with pytest.raises(WordRangeError, match="exceeds"):
            record_decision(
                session,
                pipeline_run_id=run_id,
                diarization_label="SPEAKER_00",
                decision=Decision.ASSIGN,
                speaker_id=speaker_id,
                operator="ben",
                idempotency_key="range-oob",
                transcript_segment_id=seg_id,
                start_word_index=1,
                end_word_index=4,
            )
        session.rollback()


def test_half_set_range_is_rejected(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, seg_id, speaker_id = seed_splittable_segment(session)
        with pytest.raises(WordRangeError, match="together"):
            record_decision(
                session,
                pipeline_run_id=run_id,
                diarization_label="SPEAKER_00",
                decision=Decision.ASSIGN,
                speaker_id=speaker_id,
                operator="ben",
                idempotency_key="range-half",
                transcript_segment_id=seg_id,
                start_word_index=1,
                end_word_index=None,
            )
        session.rollback()


def test_empty_range_is_rejected(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, seg_id, speaker_id = seed_splittable_segment(session)
        with pytest.raises(WordRangeError):  # end must be > start
            record_decision(
                session,
                pipeline_run_id=run_id,
                diarization_label="SPEAKER_00",
                decision=Decision.ASSIGN,
                speaker_id=speaker_id,
                operator="ben",
                idempotency_key="range-empty",
                transcript_segment_id=seg_id,
                start_word_index=2,
                end_word_index=2,
            )
        session.rollback()


def test_range_without_segment_is_rejected(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, _seg_id, speaker_id = seed_splittable_segment(session)
        with pytest.raises(WordRangeError, match="transcript_segment_id"):
            record_decision(
                session,
                pipeline_run_id=run_id,
                diarization_label="SPEAKER_00",
                decision=Decision.ASSIGN,
                speaker_id=speaker_id,
                operator="ben",
                idempotency_key="range-noseg",
                transcript_segment_id=None,
                start_word_index=0,
                end_word_index=1,
            )
        session.rollback()


def test_ranged_ruling_on_a_foreign_run_segment_is_rejected(
    session_factory: sessionmaker[Session],
) -> None:
    """The sole writer refuses a ranged ruling whose segment belongs to a different
    run: such a row would be permanently unreadable (loaded under run A, but the
    read path only looks up run A's own segment ids) AND append-only-uncleanable —
    so the invariant home rejects it rather than trust the caller checked ownership.
    """
    with session_factory() as session:
        run_a, _seg_a, speaker_id = seed_splittable_segment(session)
        _run_b, seg_b, _speaker_b = seed_splittable_segment(session)
        with pytest.raises(WordRangeError, match="does not belong to run"):
            record_decision(
                session,
                pipeline_run_id=run_a,  # run A ...
                diarization_label="SPEAKER_00",
                decision=Decision.ASSIGN,
                speaker_id=speaker_id,
                operator="ben",
                idempotency_key="range-foreign",
                transcript_segment_id=seg_b,  # ... but run B's segment
                start_word_index=0,
                end_word_index=1,
            )
        session.rollback()
