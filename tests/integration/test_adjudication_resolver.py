"""Read-time attribution: decision precedence, queue membership, correction order."""

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session, sessionmaker

from voxint.adjudication.ledger import record_decision
from voxint.adjudication.resolver import (
    Resolution,
    adjudication_queue,
    effective_decisions,
    label_states,
    review_backlog_count,
)
from voxint.adjudication.slots import claim_run
from voxint.db.models import (
    EMBEDDING_DIM,
    Decision,
    DiarizationTurn,
    MediaItem,
    MediaSourceMetadata,
    PipelineRun,
    RunStatus,
    Speaker,
    SpeakerAssignment,
)

SPACE = "titanet-large-v1"


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


def add_speaker(session: Session, name: str) -> uuid.UUID:
    speaker = Speaker(display_name=name)
    session.add(speaker)
    session.flush()
    return speaker.id


def test_precedence_human_over_grounded_over_unresolved(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = make_completed_run(session)
        for index, label in enumerate(["S0", "S0", "S1", "S1", "S2", "S2", "S3"]):
            add_turn(session, run_id, index, label)
        alice = add_speaker(session, "Alice")
        bob = add_speaker(session, "Bob")
        carol = add_speaker(session, "Carol")
        # S0: grounded cosine — machine identity stands.
        session.add(
            SpeakerAssignment(
                pipeline_run_id=run_id,
                diarization_label="S0",
                speaker_id=alice,
                method="cosine",
                confidence=0.9,
                grounded=True,
            )
        )
        # S1: grounded cosine says Bob, but a human ruled Carol — human wins.
        session.add(
            SpeakerAssignment(
                pipeline_run_id=run_id,
                diarization_label="S1",
                speaker_id=bob,
                method="cosine",
                confidence=0.85,
                grounded=True,
            )
        )
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S1",
            decision=Decision.ASSIGN,
            operator="ben",
            idempotency_key="k-s1",
            speaker_id=carol,
        )
        # S2: only a non-grounded cosine + an llm hint — unresolved.
        session.add(
            SpeakerAssignment(
                pipeline_run_id=run_id,
                diarization_label="S2",
                speaker_id=bob,
                method="cosine",
                confidence=0.65,
                grounded=False,
            )
        )
        session.add(
            SpeakerAssignment(
                pipeline_run_id=run_id,
                diarization_label="S2",
                method="llm_hint",
                proposed_name="Dave",
                grounded=False,
            )
        )
        session.commit()

        by_label = {s.label: s for s in label_states(session, run_id)}
        assert by_label["S0"].resolution is Resolution.GROUNDED_COSINE
        assert by_label["S0"].speaker_name == "Alice"
        assert by_label["S1"].resolution is Resolution.HUMAN_ASSIGN
        assert by_label["S1"].speaker_name == "Carol"
        assert by_label["S1"].cosine_speaker_name == "Bob"  # evidence still shown
        assert by_label["S2"].resolution is Resolution.UNRESOLVED
        assert by_label["S2"].llm_hint_name == "Dave"
        assert by_label["S2"].speaker_name is None  # hints are never identity
        assert by_label["S3"].resolution is Resolution.UNRESOLVED


def test_exclude_and_unknown_resolve_without_identity(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = make_completed_run(session)
        add_turn(session, run_id, 0, "S0")
        add_turn(session, run_id, 1, "S1")
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S0",
            decision=Decision.EXCLUDE,
            operator="ben",
            idempotency_key="k-ex",
        )
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S1",
            decision=Decision.UNKNOWN,
            operator="ben",
            idempotency_key="k-unk",
        )
        session.commit()
        by_label = {s.label: s for s in label_states(session, run_id)}
        assert by_label["S0"].resolution is Resolution.HUMAN_EXCLUDE
        assert by_label["S1"].resolution is Resolution.HUMAN_UNKNOWN
        assert by_label["S0"].speaker_id is None
        assert by_label["S1"].speaker_id is None


def test_correction_newest_decision_wins(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id = make_completed_run(session)
        add_turn(session, run_id, 0, "S0")
        alice = add_speaker(session, "Alice")
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S0",
            decision=Decision.UNKNOWN,
            operator="ben",
            idempotency_key="k-1",
        )
        # Separate transaction: created_at is transaction time, and the ledger
        # trigger forbids editing timestamps — a real correction is always a
        # later transaction.
        session.commit()
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S0",
            decision=Decision.ASSIGN,
            operator="ben",
            idempotency_key="k-2",
            speaker_id=alice,
        )
        session.commit()
        effective = effective_decisions(session, run_id)
        assert effective["S0"].decision == Decision.ASSIGN.value
        state = label_states(session, run_id)[0]
        assert state.resolution is Resolution.HUMAN_ASSIGN
        assert state.speaker_name == "Alice"


def test_queue_lists_only_completed_runs_with_unresolved_labels(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        # Unresolved + completed: queued.
        queued_run = make_completed_run(session)
        add_turn(session, queued_run, 0, "S0")
        # Completed but fully ruled: not queued.
        ruled_run = make_completed_run(session)
        add_turn(session, ruled_run, 0, "S0")
        record_decision(
            session,
            pipeline_run_id=ruled_run,
            diarization_label="S0",
            decision=Decision.EXCLUDE,
            operator="ben",
            idempotency_key="k-ruled",
        )
        # Unresolved but still running: not queued.
        media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
        session.add(media)
        session.flush()
        running = PipelineRun(
            media_item_id=media.id,
            status=RunStatus.RUNNING.value,
            current_stage="prepare",
        )
        session.add(running)
        session.flush()
        add_turn(session, running.id, 0, "S0")
        session.commit()

        entries = adjudication_queue(session)
        assert [e.run_id for e in entries] == [queued_run]
        assert entries[0].unresolved_labels == 1
        assert entries[0].claimed_by is None

        # A live claim surfaces in the queue row.
        claim_run(session, queued_run, reviewer="ben", ttl_seconds=600)
        session.commit()
        assert adjudication_queue(session)[0].claimed_by == "ben"


def test_queue_entry_carries_display_context(
    session_factory: sessionmaker[Session],
) -> None:
    """The queue row exposes title, probed duration, and created_at (issue #56)."""
    with session_factory() as session:
        media = MediaItem(
            source_path=f"incoming/{uuid.uuid4()}.wav",
            duration_seconds=125.0,
        )
        session.add(media)
        session.flush()
        session.add(
            MediaSourceMetadata(
                media_item_id=media.id,
                source_kind="ytdlp",
                title="City Council 2026-08",
                raw={"id": "abc"},
                raw_schema_version=1,
                acquired_at=datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
            )
        )
        run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
        session.add(run)
        session.flush()
        add_turn(session, run.id, 0, "S0")
        session.commit()

        entry = adjudication_queue(session)[0]
        assert entry.title == "City Council 2026-08"
        assert entry.duration_seconds == 125.0
        assert entry.created_at is not None

    # An upload with no metadata snapshot leaves title/duration None, no error.
    with session_factory() as session:
        bare = make_completed_run(session)
        add_turn(session, bare, 0, "S0")
        session.commit()
        entry = next(e for e in adjudication_queue(session) if e.run_id == bare)
        assert entry.title is None
        assert entry.duration_seconds is None
        assert entry.created_at is not None


def test_queue_sidecar_title_wins_over_scraped_title(
    session_factory: sessionmaker[Session],
) -> None:
    """The frozen sidecar title (issue #104, operator intent) beats the
    acquisition-metadata title; a tampered snapshot degrades to the scraped one."""
    with session_factory() as session:
        media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
        session.add(media)
        session.flush()
        session.add(
            MediaSourceMetadata(
                media_item_id=media.id,
                source_kind="ytdlp",
                title="Scraped title",
                raw={"id": "abc"},
                raw_schema_version=1,
                acquired_at=datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
            )
        )
        run = PipelineRun(
            media_item_id=media.id,
            status=RunStatus.COMPLETED.value,
            sidecar={"title": "Operator title", "content_item_id": 1},
        )
        session.add(run)
        session.flush()
        add_turn(session, run.id, 0, "S0")
        session.commit()
        entry = next(e for e in adjudication_queue(session) if e.run_id == run.id)
        assert entry.title == "Operator title"

    # A sidecar without a usable title falls back to the scraped one.
    with session_factory() as session:
        media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
        session.add(media)
        session.flush()
        session.add(
            MediaSourceMetadata(
                media_item_id=media.id,
                source_kind="ytdlp",
                title="Scraped title",
                raw={"id": "abc"},
                raw_schema_version=1,
                acquired_at=datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
            )
        )
        run = PipelineRun(
            media_item_id=media.id,
            status=RunStatus.COMPLETED.value,
            sidecar={"notes": "no title here"},
        )
        session.add(run)
        session.flush()
        add_turn(session, run.id, 0, "S0")
        session.commit()
        entry = next(e for e in adjudication_queue(session) if e.run_id == run.id)
        assert entry.title == "Scraped title"


def test_queue_sort_unresolved_orders_by_voice_count(
    session_factory: sessionmaker[Session],
) -> None:
    """sort="unresolved" surfaces the most-unresolved runs first, oldest-tie-broken."""
    with session_factory() as session:
        # Created oldest→newest: one-voice, then two-voice, then another one-voice.
        # Explicit, distinct created_at values give a well-defined oldest-first
        # order (same-transaction rows share the server default now(), so their
        # order is otherwise decided only by the id tie-breaker).
        base = datetime(2026, 8, 1, 12, 0, 0, tzinfo=UTC)
        one_a = make_completed_run(session)
        add_turn(session, one_a, 0, "S0")
        two = make_completed_run(session)
        add_turn(session, two, 0, "S0")
        add_turn(session, two, 1, "S1")
        one_b = make_completed_run(session)
        add_turn(session, one_b, 0, "S0")
        for offset, rid in enumerate((one_a, two, one_b)):
            session.get(PipelineRun, rid).created_at = base + timedelta(minutes=offset)
        session.commit()

        # Default: oldest-first (FIFO), unchanged behaviour.
        assert [e.run_id for e in adjudication_queue(session)] == [one_a, two, one_b]

        # Unresolved-first: the two-voice run leads; the two one-voice runs keep
        # their oldest-first order among the tie (stable sort).
        by_work = adjudication_queue(session, sort="unresolved")
        assert [e.run_id for e in by_work] == [two, one_a, one_b]
        assert [e.unresolved_labels for e in by_work] == [2, 1, 1]

        # An unknown sort degrades to the default order rather than erroring.
        assert [e.run_id for e in adjudication_queue(session, sort="bogus")] == [
            one_a,
            two,
            one_b,
        ]


def test_review_backlog_count_matches_queue_length(
    session_factory: sessionmaker[Session],
) -> None:
    """The dashboard headline (issue #117) equals the queue it links to, by
    construction, across every eligibility outcome — completed-and-unresolved,
    fully-ruled, still-running, and archived-with-unresolved."""
    # Empty system: no runs, no backlog.
    with session_factory() as session:
        assert review_backlog_count(session) == len(adjudication_queue(session)) == 0

    with session_factory() as session:
        # Eligible: completed with an unresolved voice.
        eligible = make_completed_run(session)
        add_turn(session, eligible, 0, "S0")
        # Completed but fully ruled: not eligible.
        ruled = make_completed_run(session)
        add_turn(session, ruled, 0, "S0")
        record_decision(
            session,
            pipeline_run_id=ruled,
            diarization_label="S0",
            decision=Decision.EXCLUDE,
            operator="ben",
            idempotency_key="k-ruled-backlog",
        )
        # Unresolved but still running: not eligible.
        media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
        session.add(media)
        session.flush()
        running = PipelineRun(
            media_item_id=media.id,
            status=RunStatus.RUNNING.value,
            current_stage="prepare",
        )
        session.add(running)
        session.flush()
        add_turn(session, running.id, 0, "S0")
        # Archived completed run with an unresolved voice: hidden from the queue.
        archived = make_completed_run(session)
        add_turn(session, archived, 0, "S0")
        session.get(PipelineRun, archived).archived_at = datetime.now(tz=UTC)
        session.commit()

        assert review_backlog_count(session) == len(adjudication_queue(session)) == 1

        # Resolving the one eligible run drops the backlog to zero, still in step.
        record_decision(
            session,
            pipeline_run_id=eligible,
            diarization_label="S0",
            decision=Decision.EXCLUDE,
            operator="ben",
            idempotency_key="k-eligible-backlog",
        )
        session.commit()
        assert review_backlog_count(session) == len(adjudication_queue(session)) == 0
