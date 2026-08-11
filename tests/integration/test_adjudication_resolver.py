"""Read-time attribution: decision precedence, queue membership, correction order."""

import uuid

from sqlalchemy.orm import Session, sessionmaker

from voxint.adjudication.ledger import record_decision
from voxint.adjudication.resolver import (
    Resolution,
    adjudication_queue,
    effective_decisions,
    label_states,
)
from voxint.adjudication.slots import claim_run
from voxint.db.models import (
    EMBEDDING_DIM,
    Decision,
    DiarizationTurn,
    MediaItem,
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
