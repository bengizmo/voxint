"""Enrollment: centroid creation, provenance, replay safety, refusal paths."""

import math
import uuid

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from voxint.adjudication.enrollment import EnrollmentError, enroll_new_speaker
from voxint.adjudication.ledger import ConflictingReplayError, record_decision
from voxint.db.models import (
    EMBEDDING_DIM,
    AdjudicationDecision,
    Decision,
    DiarizationTurn,
    MediaItem,
    PipelineRun,
    RunStatus,
    Speaker,
    SpeakerEmbedding,
)
from voxint.speakers.matching import MatchingGates
from voxint.speakers.roster import rename_speaker

SPACE = "titanet-large-v2"
GATES = MatchingGates()


def unit(dim: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    vector[dim] = 1.0
    return vector


def make_completed_run(session: Session) -> uuid.UUID:
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()
    return run.id


def add_turn(
    session: Session,
    run_id: uuid.UUID,
    index: int,
    label: str,
    *,
    vector: list[float] | None = None,
    duration: float = 8.0,
    overlap_seconds: float = 0.0,
    space: str = SPACE,
) -> None:
    session.add(
        DiarizationTurn(
            pipeline_run_id=run_id,
            turn_index=index,
            start_seconds=float(index * 20),
            end_seconds=float(index * 20) + duration,
            label=label,
            overlap=overlap_seconds > 0,
            overlap_seconds=overlap_seconds,
            embedding=vector or unit(index % EMBEDDING_DIM),
            embedding_space=space,
        )
    )


def test_enrollment_creates_speaker_centroid_and_ruling(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = make_completed_run(session)
        add_turn(session, run_id, 0, "S0", vector=unit(0))
        add_turn(session, run_id, 1, "S0", vector=unit(0))
        session.commit()

        result = enroll_new_speaker(
            session,
            run_id=run_id,
            diarization_label="S0",
            display_name="  Alice  ",
            operator="ben",
            idempotency_key="k-enroll",
            gates=GATES,
        )
        session.commit()

        assert result.created_speaker
        speaker = session.get(Speaker, result.speaker_id)
        assert speaker is not None and speaker.display_name == "Alice"

        embedding = session.execute(
            select(SpeakerEmbedding).where(SpeakerEmbedding.speaker_id == speaker.id)
        ).scalar_one()
        assert embedding.embedding_space == SPACE
        assert embedding.source_pipeline_run_id == run_id
        assert embedding.source_diarization_label == "S0"
        assert embedding.source_adjudication_decision_id == result.decision_id
        vec = np.asarray(embedding.embedding, dtype=np.float64)
        assert math.isclose(float(np.linalg.norm(vec)), 1.0, rel_tol=1e-6)
        assert math.isclose(float(vec[0]), 1.0, rel_tol=1e-6)  # both turns point at e0

        decision = session.get(AdjudicationDecision, result.decision_id)
        assert decision is not None
        assert decision.decision == Decision.ASSIGN.value
        assert decision.speaker_id == speaker.id


def test_replayed_enrollment_returns_original_without_new_rows(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = make_completed_run(session)
        add_turn(session, run_id, 0, "S0")
        add_turn(session, run_id, 1, "S0")
        session.commit()
        first = enroll_new_speaker(
            session,
            run_id=run_id,
            diarization_label="S0",
            display_name="Alice",
            operator="ben",
            idempotency_key="k-replay",
            gates=GATES,
        )
        session.commit()
        replay = enroll_new_speaker(
            session,
            run_id=run_id,
            diarization_label="S0",
            display_name="Alice",
            operator="ben",
            idempotency_key="k-replay",
            gates=GATES,
        )
        session.commit()
        assert not replay.created_speaker
        assert replay.speaker_id == first.speaker_id
        assert replay.decision_id == first.decision_id
        assert len(session.execute(select(Speaker)).scalars().all()) == 1
        assert len(session.execute(select(SpeakerEmbedding)).scalars().all()) == 1


def test_duplicate_display_name_refused(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id = make_completed_run(session)
        add_turn(session, run_id, 0, "S0")
        add_turn(session, run_id, 1, "S0")
        session.add(Speaker(display_name="Alice"))
        session.commit()
        with pytest.raises(EnrollmentError, match="already exists"):
            enroll_new_speaker(
                session,
                run_id=run_id,
                diarization_label="S0",
                display_name="Alice",
                operator="ben",
                idempotency_key="k-dup",
                gates=GATES,
            )


def test_label_without_eligible_turns_refused(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = make_completed_run(session)
        # Overlap ratio 100% — ineligible under the default 20% gate.
        add_turn(session, run_id, 0, "S0", duration=8.0, overlap_seconds=8.0)
        session.commit()
        with pytest.raises(EnrollmentError, match="no speaker audio"):
            enroll_new_speaker(
                session,
                run_id=run_id,
                diarization_label="S0",
                display_name="Alice",
                operator="ben",
                idempotency_key="k-inelig",
                gates=GATES,
            )


def test_blank_and_oversized_names_refused(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = make_completed_run(session)
        add_turn(session, run_id, 0, "S0")
        session.commit()
        for bad in ("   ", "x" * 121):
            with pytest.raises(EnrollmentError, match="display name"):
                enroll_new_speaker(
                    session,
                    run_id=run_id,
                    diarization_label="S0",
                    display_name=bad,
                    operator="ben",
                    idempotency_key="k-bad",
                    gates=GATES,
                )


def test_key_reused_with_different_payload_conflicts(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = make_completed_run(session)
        add_turn(session, run_id, 0, "S0")
        add_turn(session, run_id, 1, "S0")
        add_turn(session, run_id, 2, "S1")
        add_turn(session, run_id, 3, "S1")
        session.commit()
        enroll_new_speaker(
            session,
            run_id=run_id,
            diarization_label="S0",
            display_name="Alice",
            operator="ben",
            idempotency_key="k-payload",
            gates=GATES,
        )
        session.commit()
        # Same key, different label: NOT a replay — must conflict, never
        # silently "succeed" pointing at the S0 enrollment.
        with pytest.raises(ConflictingReplayError):
            enroll_new_speaker(
                session,
                run_id=run_id,
                diarization_label="S1",
                display_name="Alice",
                operator="ben",
                idempotency_key="k-payload",
                gates=GATES,
            )
        session.rollback()
        # Same key, same run/label/operator but a different NAME is still a
        # replay: display_name is mutable (roster rename), so it is deliberately
        # NOT part of the replay comparison — the durable identity is the ledger
        # row. The original outcome comes back; no second speaker is minted.
        replayed = enroll_new_speaker(
            session,
            run_id=run_id,
            diarization_label="S0",
            display_name="Someone Else",
            operator="ben",
            idempotency_key="k-payload",
            gates=GATES,
        )
        assert replayed.created_speaker is False
        alice_id = session.execute(
            select(Speaker.id).where(Speaker.display_name == "Alice")
        ).scalar_one()
        assert replayed.speaker_id == alice_id


def test_replay_still_succeeds_after_rename(
    session_factory: sessionmaker[Session],
) -> None:
    """The rename-breaks-replay bug: a replayed enrollment POST arriving after
    the speaker was renamed must return the original outcome, not conflict."""
    with session_factory() as session:
        run_id = make_completed_run(session)
        add_turn(session, run_id, 0, "S0")
        add_turn(session, run_id, 1, "S0")
        session.commit()
        original = enroll_new_speaker(
            session,
            run_id=run_id,
            diarization_label="S0",
            display_name="Alice",
            operator="ben",
            idempotency_key="k-rename-replay",
            gates=GATES,
        )
        session.commit()
        rename_speaker(session, original.speaker_id, "Alice Verified")
        session.commit()
        replayed = enroll_new_speaker(
            session,
            run_id=run_id,
            diarization_label="S0",
            display_name="Alice",  # the stale form still carries the old name
            operator="ben",
            idempotency_key="k-rename-replay",
            gates=GATES,
        )
        assert replayed.created_speaker is False
        assert replayed.speaker_id == original.speaker_id
        assert replayed.decision_id == original.decision_id


def test_key_reused_from_non_assign_decision_refused(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = make_completed_run(session)
        add_turn(session, run_id, 0, "S0")
        add_turn(session, run_id, 1, "S0")
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S0",
            decision=Decision.EXCLUDE,
            operator="ben",
            idempotency_key="k-mixed",
        )
        session.commit()
        with pytest.raises(EnrollmentError, match="conflicts with a previous"):
            enroll_new_speaker(
                session,
                run_id=run_id,
                diarization_label="S0",
                display_name="Alice",
                operator="ben",
                idempotency_key="k-mixed",
                gates=GATES,
            )
