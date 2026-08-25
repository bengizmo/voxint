"""Per-speaker aggregation from effective resolution (issue #159).

Hand-computed fixtures: merge chains, override ordering, splits and word-range
reassignments (no double-count), the verified flip on a later exclude, the
canonical latest-run-per-media collapse, and the surviving-grounded evidence
keys the tier module consumes.
"""

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session, sessionmaker

from voxint.adjudication.ledger import record_decision
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
    SpeakerEmbedding,
    TranscriptSegment,
)
from voxint.speakers.aggregate import (
    aggregate_for_speaker,
    aggregate_speakers,
    enrollment_count,
)
from voxint.speakers.roster import merge_speakers

SPACE = "titanet-large-v1"
WORDS = ["alpha ", "beta ", "gamma ", "delta"]
RAW = "alpha beta gamma delta"
BASE = datetime(2026, 1, 1, tzinfo=UTC)


def make_media(session: Session, *, created_at: datetime) -> uuid.UUID:
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    media.created_at = created_at
    session.flush()
    return media.id


def make_run(
    session: Session,
    media_id: uuid.UUID,
    *,
    created_at: datetime,
    status: str = RunStatus.COMPLETED.value,
    archived: bool = False,
) -> uuid.UUID:
    run = PipelineRun(media_item_id=media_id, status=status)
    session.add(run)
    session.flush()
    run.created_at = created_at
    if archived:
        run.archived_at = created_at
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
    duration: float = 8.0,
) -> uuid.UUID:
    lo = float(index * 10)
    hi = lo + duration
    seg = TranscriptSegment(
        pipeline_run_id=run_id,
        segment_index=index,
        start_seconds=lo,
        end_seconds=hi,
        raw_text=RAW,
        diarization_label=label,
        words=(
            # Word timings tile the whole interval, so split children partition
            # the parent's full duration.
            [
                {
                    "word": w,
                    "start": lo + i * duration / len(WORDS),
                    "end": lo + (i + 1) * duration / len(WORDS),
                }
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


def add_grounded(
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


def assign(
    session: Session,
    run_id: uuid.UUID,
    label: str,
    speaker_id: uuid.UUID | None,
    *,
    decision: Decision = Decision.ASSIGN,
    segment_id: uuid.UUID | None = None,
    words: tuple[int, int] | None = None,
) -> None:
    record_decision(
        session,
        pipeline_run_id=run_id,
        diarization_label=label,
        decision=decision,
        operator="op",
        idempotency_key=f"k-{uuid.uuid4()}",
        speaker_id=speaker_id,
        transcript_segment_id=segment_id,
        start_word_index=words[0] if words else None,
        end_word_index=words[1] if words else None,
    )


def test_hand_computed_totals_two_media(
    session_factory: sessionmaker[Session],
) -> None:
    """Alice: grounded on media1 (2 segs x 8 s) + human on media2 (1 seg x 8 s)
    = 3 segments, 24 s, 2 files, first seen media1's created_at, verified."""
    with session_factory() as session:
        media1 = make_media(session, created_at=BASE)
        media2 = make_media(session, created_at=BASE + timedelta(days=2))
        run1 = make_run(session, media1, created_at=BASE)
        run2 = make_run(session, media2, created_at=BASE + timedelta(days=2))
        alice = add_speaker(session, "Alice")
        bob = add_speaker(session, "Bob")
        for i, label in enumerate(["S0", "S0"]):
            add_turn(session, run1, i, label)
            add_segment(session, run1, i, label)
        add_grounded(session, run1, "S0", alice)
        add_turn(session, run2, 0, "S1")
        add_turn(session, run2, 1, "S2")
        add_segment(session, run2, 0, "S1")
        add_segment(session, run2, 1, "S2")
        assign(session, run2, "S1", alice)
        assign(session, run2, "S2", bob)
        session.commit()

        result = aggregate_speakers(session)
        assert result.runs_scanned == 2
        agg = result.by_speaker[alice]
        assert agg.files == 2
        assert agg.segments == 3
        assert agg.seconds == 24.0
        assert agg.first_seen == BASE
        assert agg.last_seen == BASE + timedelta(days=2)
        assert agg.verified  # the human assign on run2
        # Grounded evidence: only run1's surviving grounded label.
        assert agg.grounded_keys == ((run1, "S0"),)
        bob_agg = result.by_speaker[bob]
        assert (bob_agg.files, bob_agg.segments, bob_agg.seconds) == (1, 1, 8.0)
        # Appearances are newest-media-first.
        assert [a.media_id for a in agg.appearances] == [media2, media1]


def test_only_latest_completed_run_per_media_counts(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        media = make_media(session, created_at=BASE)
        old = make_run(session, media, created_at=BASE)
        new = make_run(session, media, created_at=BASE + timedelta(hours=1))
        # A newer but archived run and a newer running run are both ignored.
        make_run(
            session, media, created_at=BASE + timedelta(hours=2), archived=True
        )
        make_run(
            session,
            media,
            created_at=BASE + timedelta(hours=3),
            status=RunStatus.RUNNING.value,
        )
        alice = add_speaker(session, "Alice")
        for run_id in (old, new):
            add_turn(session, run_id, 0, "S0")
            add_segment(session, run_id, 0, "S0")
            add_grounded(session, run_id, "S0", alice)
        session.commit()

        result = aggregate_speakers(session)
        assert result.runs_scanned == 1
        agg = result.by_speaker[alice]
        # One canonical run: reprocessing did not double the minutes.
        assert (agg.files, agg.segments, agg.seconds) == (1, 1, 8.0)
        assert agg.appearances[0].run_id == new


def test_merge_chain_aggregates_under_terminal_target(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        media = make_media(session, created_at=BASE)
        run = make_run(session, media, created_at=BASE)
        a = add_speaker(session, "A")
        b = add_speaker(session, "B")
        c = add_speaker(session, "C")
        add_turn(session, run, 0, "S0")
        add_segment(session, run, 0, "S0")
        assign(session, run, "S0", a)
        session.add(
            SpeakerEmbedding(
                speaker_id=a,
                embedding_space=SPACE,
                embedding=[1.0] + [0.0] * (EMBEDDING_DIM - 1),
            )
        )
        session.commit()
        merge_speakers(session, a, b)
        session.commit()
        merge_speakers(session, b, c)
        session.commit()

        result = aggregate_speakers(session)
        assert a not in result.by_speaker
        assert b not in result.by_speaker
        assert result.by_speaker[c].verified
        # aggregate_for_speaker canonicalizes a tombstone id.
        assert aggregate_for_speaker(session, a).speaker_id == c
        # Enrollments count across aliases (the embedding moved on merge anyway).
        assert enrollment_count(session, a) == 1
        assert enrollment_count(session, c) == 1


def test_later_exclude_flips_verified_off(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        media = make_media(session, created_at=BASE)
        run = make_run(session, media, created_at=BASE)
        alice = add_speaker(session, "Alice")
        add_turn(session, run, 0, "S0")
        add_segment(session, run, 0, "S0")
        assign(session, run, "S0", alice)
        session.commit()
        assert aggregate_for_speaker(session, alice).verified
        assign(session, run, "S0", None, decision=Decision.EXCLUDE)
        session.commit()
        agg = aggregate_for_speaker(session, alice)
        assert not agg.verified
        assert (agg.files, agg.seconds) == (0, 0.0)


def test_displaced_human_label_assign_does_not_verify(
    session_factory: sessionmaker[Session],
) -> None:
    """A human label assign whose only segment is overridden away verifies the
    override's speaker, never the displaced one."""
    with session_factory() as session:
        media = make_media(session, created_at=BASE)
        run = make_run(session, media, created_at=BASE)
        alice = add_speaker(session, "Alice")
        bob = add_speaker(session, "Bob")
        add_turn(session, run, 0, "S0")
        seg = add_segment(session, run, 0, "S0")
        assign(session, run, "S0", alice)
        assign(session, run, "S0", bob, segment_id=seg)
        session.commit()
        result = aggregate_speakers(session)
        assert alice not in result.by_speaker
        assert result.by_speaker[bob].verified


def test_word_range_reassignment_not_double_counted(
    session_factory: sessionmaker[Session],
) -> None:
    """A split segment's seconds partition between the range's speaker and the
    parent's speaker — the total equals the parent duration exactly once."""
    with session_factory() as session:
        media = make_media(session, created_at=BASE)
        run = make_run(session, media, created_at=BASE)
        alice = add_speaker(session, "Alice")
        bob = add_speaker(session, "Bob")
        add_turn(session, run, 0, "S0")
        seg = add_segment(session, run, 0, "S0", with_words=True)
        assign(session, run, "S0", alice)
        session.add(
            SegmentSplitBoundary(
                pipeline_run_id=run, parent_segment_id=seg, word_index=2, operator="op"
            )
        )
        assign(session, run, "S0", bob, segment_id=seg, words=(0, 2))
        session.commit()

        result = aggregate_speakers(session)
        alice_agg = result.by_speaker[alice]
        bob_agg = result.by_speaker[bob]
        assert alice_agg.segments == 1 and bob_agg.segments == 1
        total = alice_agg.seconds + bob_agg.seconds
        assert abs(total - 8.0) < 1e-6
        assert bob_agg.verified and alice_agg.verified
        # A fully human-attributed run leaves no grounded evidence keys.
        assert alice_agg.grounded_keys == () and bob_agg.grounded_keys == ()


def test_overridden_grounded_label_leaves_no_evidence_key(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        media = make_media(session, created_at=BASE)
        run = make_run(session, media, created_at=BASE)
        alice = add_speaker(session, "Alice")
        bob = add_speaker(session, "Bob")
        add_turn(session, run, 0, "S0")
        seg = add_segment(session, run, 0, "S0")
        add_grounded(session, run, "S0", alice)
        assign(session, run, "S0", bob, segment_id=seg)
        session.commit()
        result = aggregate_speakers(session)
        assert alice not in result.by_speaker
        assert result.by_speaker[bob].grounded_keys == ()


def test_empty_and_unknown_speaker_aggregate(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        lonely = add_speaker(session, "Lonely")
        session.commit()
        agg = aggregate_for_speaker(session, lonely)
        assert (agg.files, agg.seconds, agg.segments) == (0, 0.0, 0)
        assert agg.first_seen is None and agg.last_seen is None
        assert not agg.verified and agg.appearances == ()


def test_tier_evidence_batch_load_and_unavailable(
    session_factory: sessionmaker[Session],
) -> None:
    """evidence_for loads recorded diagnostics in one batch and reports a key
    with no match_candidates row (pre-0032 run) as unavailable."""
    from voxint.db.models import MatchCandidate
    from voxint.speakers.tiers import evidence_for

    with session_factory() as session:
        media = make_media(session, created_at=BASE)
        run = make_run(session, media, created_at=BASE)
        alice = add_speaker(session, "Alice")
        session.add(
            MatchCandidate(
                pipeline_run_id=run,
                diarization_label="S0",
                decision="accepted",
                reason="accepted",
                embedding_space=SPACE,
                top_speaker_id=alice,
                similarity=0.82,
                margin=0.12,
                vote_agreement=0.9,
                grounded=True,
                eligible_turns=5,
                eligible_seconds=40.0,
                roster_size=3,
            )
        )
        session.commit()

        recorded, absent = evidence_for(session, [(run, "S0"), (run, "S9")])
        assert recorded.available and recorded.similarity == 0.82
        assert recorded.roster_size == 3
        assert not absent.available and absent.label == "S9"
