"""The identity-grade attribution seam (issue #159): winning scope, canonical
speaker ids, and parity with the rendered transcript.

``attributed_intervals`` is what the speakers aggregation folds, so these pin
exactly the semantics the overview's numbers depend on: merge-chain
canonicalization, later-ruling recency, split expansion, word-range precedence,
and — the badge-critical case — a human label assignment fully displaced by
narrower overrides yielding NO interval for the displaced speaker.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy.orm import Session, sessionmaker

from voxint.adjudication.attribution import (
    AttributionScope,
    attributed_intervals,
)
from voxint.adjudication.ledger import record_decision
from voxint.adjudication.resolver import Resolution
from voxint.adjudication.transcript import TranscriptText, attributed_transcript
from voxint.db.models import (
    EMBEDDING_DIM,
    Decision,
    DiarizationTurn,
    MediaItem,
    PipelineRun,
    RunStatus,
    SegmentSplitBoundary,
    Speaker,
    SpeakerAssignment,
    TranscriptSegment,
)
from voxint.speakers.roster import merge_speakers

SPACE = "titanet-large-v1"
WORDS = ["alpha ", "beta ", "gamma ", "delta"]
RAW = "alpha beta gamma delta"


def make_completed_run(session: Session) -> uuid.UUID:
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()
    return run.id


def add_turn(session: Session, run_id: uuid.UUID, index: int, label: str) -> None:
    vector = [0.0] * EMBEDDING_DIM
    vector[index % EMBEDDING_DIM] = 1.0
    session.add(
        DiarizationTurn(
            pipeline_run_id=run_id,
            turn_index=index,
            start_seconds=float(index * 10),
            end_seconds=float(index * 10 + 8),
            label=label,
            embedding=vector,
            embedding_space=SPACE,
        )
    )


def add_segment(
    session: Session,
    run_id: uuid.UUID,
    index: int,
    label: str,
    *,
    with_words: bool = False,
    start: float | None = None,
    end: float | None = None,
) -> uuid.UUID:
    lo = float(index * 10) if start is None else start
    hi = float(index * 10 + 8) if end is None else end
    seg = TranscriptSegment(
        pipeline_run_id=run_id,
        segment_index=index,
        start_seconds=lo,
        end_seconds=hi,
        raw_text=RAW,
        diarization_label=label,
        words=(
            [
                {"word": w, "start": lo + i, "end": min(lo + i + 1, hi)}
                for i, w in enumerate(WORDS)
            ]
            if with_words
            else None
        ),
    )
    session.add(seg)
    session.flush()
    return seg.id


def add_speaker(session: Session, name: str) -> uuid.UUID:
    speaker = Speaker(display_name=name)
    session.add(speaker)
    session.flush()
    return speaker.id


def add_grounded_cosine(
    session: Session, run_id: uuid.UUID, label: str, speaker_id: uuid.UUID
) -> None:
    session.add(
        SpeakerAssignment(
            pipeline_run_id=run_id,
            diarization_label=label,
            speaker_id=speaker_id,
            method="cosine",
            confidence=0.9,
            grounded=True,
        )
    )


def test_label_precedence_and_nonassign_kinds(
    session_factory: sessionmaker[Session],
) -> None:
    """Human assign beats grounded cosine; exclude/unknown/unresolved carry no id."""
    with session_factory() as session:
        run_id = make_completed_run(session)
        for i, label in enumerate(["S0", "S1", "S2", "S3"]):
            add_turn(session, run_id, i, label)
            add_segment(session, run_id, i, label)
        alice = add_speaker(session, "Alice")
        bob = add_speaker(session, "Bob")
        carol = add_speaker(session, "Carol")
        add_grounded_cosine(session, run_id, "S0", alice)
        add_grounded_cosine(session, run_id, "S1", bob)
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S1",
            decision=Decision.ASSIGN,
            operator="op",
            idempotency_key="k-s1",
            speaker_id=carol,
        )
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S2",
            decision=Decision.EXCLUDE,
            operator="op",
            idempotency_key="k-s2",
        )
        session.commit()

        intervals = attributed_intervals(session, run_id)
        by_label = {iv.diarization_label: iv for iv in intervals}
        assert len(intervals) == 4
        s0 = by_label["S0"]
        assert (s0.scope, s0.resolution, s0.speaker_id) == (
            AttributionScope.LABEL,
            Resolution.GROUNDED_COSINE,
            alice,
        )
        assert not s0.is_human_assign
        s1 = by_label["S1"]
        assert (s1.resolution, s1.speaker_id) == (Resolution.HUMAN_ASSIGN, carol)
        assert s1.is_human_assign
        s2 = by_label["S2"]
        assert (s2.resolution, s2.speaker_id) == (Resolution.HUMAN_EXCLUDE, None)
        s3 = by_label["S3"]
        assert (s3.resolution, s3.speaker_id) == (Resolution.UNRESOLVED, None)
        assert s3.segment_id is not None
        # Interval bounds are the segment's own interval for unsplit segments.
        assert (s0.start_seconds, s0.end_seconds) == (0.0, 8.0)


def test_later_label_ruling_overrides_earlier(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = make_completed_run(session)
        add_turn(session, run_id, 0, "S0")
        add_segment(session, run_id, 0, "S0")
        alice = add_speaker(session, "Alice")
        bob = add_speaker(session, "Bob")
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S0",
            decision=Decision.ASSIGN,
            operator="op",
            idempotency_key="k-1",
            speaker_id=alice,
        )
        session.commit()
        assert attributed_intervals(session, run_id)[0].speaker_id == alice
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S0",
            decision=Decision.ASSIGN,
            operator="op",
            idempotency_key="k-2",
            speaker_id=bob,
        )
        session.commit()
        assert attributed_intervals(session, run_id)[0].speaker_id == bob
        # A later exclude removes the attribution entirely (the verified-badge flip).
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S0",
            decision=Decision.EXCLUDE,
            operator="op",
            idempotency_key="k-3",
        )
        session.commit()
        only = attributed_intervals(session, run_id)[0]
        assert (only.resolution, only.speaker_id) == (Resolution.HUMAN_EXCLUDE, None)


def test_merge_chain_canonicalizes_to_terminal_target(
    session_factory: sessionmaker[Session],
) -> None:
    """Assign to A, merge A into B, then B into C: intervals report C."""
    with session_factory() as session:
        run_id = make_completed_run(session)
        add_turn(session, run_id, 0, "S0")
        add_segment(session, run_id, 0, "S0")
        a = add_speaker(session, "A")
        b = add_speaker(session, "B")
        c = add_speaker(session, "C")
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S0",
            decision=Decision.ASSIGN,
            operator="op",
            idempotency_key="k-a",
            speaker_id=a,
        )
        session.commit()
        merge_speakers(session, a, b)
        session.commit()
        merge_speakers(session, b, c)
        session.commit()
        only = attributed_intervals(session, run_id)[0]
        assert only.speaker_id == c
        assert only.speaker_name == "C"
        assert only.is_human_assign


def test_segment_override_wins_and_inherit_restores(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = make_completed_run(session)
        add_turn(session, run_id, 0, "S0")
        add_turn(session, run_id, 1, "S0")
        seg0 = add_segment(session, run_id, 0, "S0")
        add_segment(session, run_id, 1, "S0")
        alice = add_speaker(session, "Alice")
        bob = add_speaker(session, "Bob")
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S0",
            decision=Decision.ASSIGN,
            operator="op",
            idempotency_key="k-label",
            speaker_id=alice,
        )
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S0",
            decision=Decision.ASSIGN,
            operator="op",
            idempotency_key="k-seg",
            speaker_id=bob,
            transcript_segment_id=seg0,
        )
        session.commit()
        first, second = attributed_intervals(session, run_id)
        assert (first.scope, first.speaker_id) == (AttributionScope.SEGMENT, bob)
        assert (second.scope, second.speaker_id) == (AttributionScope.LABEL, alice)
        # A newest inherit clears the override; the segment follows its label again.
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S0",
            decision=Decision.INHERIT,
            operator="op",
            idempotency_key="k-inherit",
            transcript_segment_id=seg0,
        )
        session.commit()
        first, _ = attributed_intervals(session, run_id)
        assert (first.scope, first.speaker_id) == (AttributionScope.LABEL, alice)


def test_split_children_and_word_range_precedence(
    session_factory: sessionmaker[Session],
) -> None:
    """A split parent expands per child; a range override wins for its exact
    range only; siblings inherit the parent's resolution."""
    with session_factory() as session:
        run_id = make_completed_run(session)
        add_turn(session, run_id, 0, "S0")
        seg0 = add_segment(session, run_id, 0, "S0", with_words=True)
        alice = add_speaker(session, "Alice")
        bob = add_speaker(session, "Bob")
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S0",
            decision=Decision.ASSIGN,
            operator="op",
            idempotency_key="k-label",
            speaker_id=alice,
        )
        session.add(
            SegmentSplitBoundary(
                pipeline_run_id=run_id,
                parent_segment_id=seg0,
                word_index=2,
                operator="op",
            )
        )
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S0",
            decision=Decision.ASSIGN,
            operator="op",
            idempotency_key="k-range",
            speaker_id=bob,
            transcript_segment_id=seg0,
            start_word_index=0,
            end_word_index=2,
        )
        session.commit()
        intervals = attributed_intervals(session, run_id)
        assert len(intervals) == 2
        first, second = intervals
        assert (first.scope, first.speaker_id) == (AttributionScope.WORD_RANGE, bob)
        assert (first.word_start, first.word_end) == (0, 2)
        assert (second.scope, second.speaker_id) == (AttributionScope.LABEL, alice)
        assert (second.word_start, second.word_end) == (2, 4)
        assert first.segment_id == seg0 and second.segment_id == seg0
        # Child bounds partition the parent's word span, never overlap.
        assert first.end_seconds <= second.start_seconds


def test_fully_displaced_human_label_assign_yields_no_interval(
    session_factory: sessionmaker[Session],
) -> None:
    """A human label assign whose every segment is overridden away contributes
    nothing — the displaced speaker must not count as attributed (or verified)."""
    with session_factory() as session:
        run_id = make_completed_run(session)
        add_turn(session, run_id, 0, "S0")
        seg0 = add_segment(session, run_id, 0, "S0")
        alice = add_speaker(session, "Alice")
        bob = add_speaker(session, "Bob")
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S0",
            decision=Decision.ASSIGN,
            operator="op",
            idempotency_key="k-label",
            speaker_id=alice,
        )
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S0",
            decision=Decision.ASSIGN,
            operator="op",
            idempotency_key="k-seg",
            speaker_id=bob,
            transcript_segment_id=seg0,
        )
        session.commit()
        intervals = attributed_intervals(session, run_id)
        assert [iv.speaker_id for iv in intervals] == [bob]


def test_zero_duration_segment_emits_zero_length_interval(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = make_completed_run(session)
        add_turn(session, run_id, 0, "S0")
        add_segment(session, run_id, 0, "S0", start=5.0, end=5.0)
        session.commit()
        only = attributed_intervals(session, run_id)[0]
        assert only.start_seconds == only.end_seconds == 5.0


def test_parity_with_attributed_transcript(
    session_factory: sessionmaker[Session],
) -> None:
    """Intervals and rendered lines come from ONE walk: same count, same order,
    same identity — the display projection can never disagree with the
    aggregation projection."""
    with session_factory() as session:
        run_id = make_completed_run(session)
        for i, label in enumerate(["S0", "S1", "S0"]):
            add_turn(session, run_id, i, label)
        seg0 = add_segment(session, run_id, 0, "S0", with_words=True)
        seg1 = add_segment(session, run_id, 1, "S1")
        add_segment(session, run_id, 2, "S0")
        alice = add_speaker(session, "Alice")
        bob = add_speaker(session, "Bob")
        add_grounded_cosine(session, run_id, "S0", alice)
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S1",
            decision=Decision.ASSIGN,
            operator="op",
            idempotency_key="k-s1",
            speaker_id=bob,
            transcript_segment_id=seg1,
        )
        session.add(
            SegmentSplitBoundary(
                pipeline_run_id=run_id,
                parent_segment_id=seg0,
                word_index=2,
                operator="op",
            )
        )
        session.commit()
        intervals = attributed_intervals(session, run_id)
        lines = attributed_transcript(session, run_id, text=TranscriptText.CORRECTED)
        assert len(intervals) == len(lines)
        for iv, ln in zip(intervals, lines, strict=True):
            assert iv.segment_id == ln.segment_id
            assert (iv.start_seconds, iv.end_seconds) == (
                ln.start_seconds,
                ln.end_seconds,
            )
            assert (iv.word_start, iv.word_end) == (ln.word_start, ln.word_end)
            if iv.speaker_id is not None:
                # An attributed interval's name is exactly the rendered speaker.
                assert ln.speaker == (iv.speaker_name or ln.diarization_label)


def test_attribution_verified_is_not_review_verified(
    session_factory: sessionmaker[Session],
) -> None:
    """Segment review verification (issue #53) never leaks into attribution:
    a review-verified but unattributed segment stays UNRESOLVED."""
    from voxint.db.models import SegmentReviewState

    with session_factory() as session:
        run_id = make_completed_run(session)
        add_turn(session, run_id, 0, "S0")
        seg0 = add_segment(session, run_id, 0, "S0")
        session.add(
            SegmentReviewState(
                transcript_segment_id=seg0,
                pipeline_run_id=run_id,
                verified_at=datetime.now(UTC),
            )
        )
        session.commit()
        only = attributed_intervals(session, run_id)
        assert only[0].resolution is Resolution.UNRESOLVED
        assert not only[0].is_human_assign
