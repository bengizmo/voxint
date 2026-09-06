"""Compensating undo preserves the ledger and restores prior resolution."""

import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from voxint.adjudication.ledger import record_decision
from voxint.adjudication.resolver import effective_decisions
from voxint.adjudication.undo import (
    UndoDriftError,
    UndoExpiredError,
    undo_enrollment,
    undo_merge,
)
from voxint.db.models import (
    EMBEDDING_DIM,
    AdjudicationDecision,
    Decision,
    MediaItem,
    PipelineRun,
    RunStatus,
    Speaker,
    SpeakerEmbedding,
)


def _run(session: Session) -> uuid.UUID:
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()
    return run.id


def _decision(
    session: Session,
    run_id: uuid.UUID,
    label: str,
    decision: Decision,
    key: str,
    speaker_id: uuid.UUID | None = None,
) -> AdjudicationDecision:
    return record_decision(
        session,
        pipeline_run_id=run_id,
        diarization_label=label,
        decision=decision,
        speaker_id=speaker_id,
        operator="ben",
        idempotency_key=key,
    )


def _embedding(
    session: Session, speaker_id: uuid.UUID, decision: AdjudicationDecision
) -> None:
    session.add(
        SpeakerEmbedding(
            speaker_id=speaker_id,
            embedding_space="test-space",
            embedding=[0.0] * EMBEDDING_DIM,
            source_pipeline_run_id=decision.pipeline_run_id,
            source_diarization_label=decision.diarization_label,
            source_adjudication_decision_id=decision.id,
        )
    )


def test_undo_enrollment_restores_prior_ruling_and_replays(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = _run(session)
        previous = _decision(session, run_id, "S0", Decision.UNKNOWN, "previous")
        session.commit()
        speaker = Speaker(display_name="Undo enrollment")
        session.add(speaker)
        session.flush()
        enrolled = _decision(
            session, run_id, "S0", Decision.ASSIGN, "enrollment", speaker.id
        )
        _embedding(session, speaker.id, enrolled)
        session.commit()

        result = undo_enrollment(
            session,
            run_id=run_id,
            decision_id=enrolled.id,
            operator="ben",
            idempotency_key="undo-enrollment",
        )
        session.commit()

        assert effective_decisions(session, run_id)["S0"].id == previous.id
        assert session.get(AdjudicationDecision, enrolled.id) is not None
        assert session.execute(select(func.count()).select_from(SpeakerEmbedding)).scalar_one() == 0
        assert session.get(Speaker, speaker.id).deleted_at is not None  # type: ignore[union-attr]
        replay = undo_enrollment(
            session,
            run_id=run_id,
            decision_id=enrolled.id,
            operator="ben",
            idempotency_key="another-replay-key",
        )
        assert replay["is_replay"] is True
        assert replay["revoke_decision_id"] == result["revoke_decision_id"]


def test_undo_enrollment_rejects_a_later_ruling(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = _run(session)
        speaker = Speaker(display_name="Undo drift")
        session.add(speaker)
        session.flush()
        enrolled = _decision(session, run_id, "S0", Decision.ASSIGN, "enroll", speaker.id)
        session.commit()
        _decision(session, run_id, "S0", Decision.EXCLUDE, "later")
        session.commit()

        with pytest.raises(UndoDriftError):
            undo_enrollment(
                session,
                run_id=run_id,
                decision_id=enrolled.id,
                operator="ben",
                idempotency_key="undo-drift",
            )


def test_undo_merge_restores_all_labels_and_archives_created_speaker(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = _run(session)
        previous = {
            label: _decision(session, run_id, label, Decision.UNKNOWN, f"old-{label}")
            for label in ("S0", "S1")
        }
        session.commit()
        speaker = Speaker(display_name="Undo merge")
        session.add(speaker)
        session.flush()
        children = {
            label: _decision(
                session,
                run_id,
                label,
                Decision.ASSIGN,
                f"merge:merge-nonce:digest:{label}",
                speaker.id,
            )
            for label in ("S0", "S1")
        }
        _embedding(session, speaker.id, children["S0"])
        session.commit()

        result = undo_merge(
            session,
            run_id=run_id,
            merge_nonce="merge-nonce",
            operator="ben",
            idempotency_key="undo-merge-request",
            grace_seconds=300,
        )
        session.commit()

        effective = effective_decisions(session, run_id)
        assert {label: effective[label].id for label in previous} == {
            label: row.id for label, row in previous.items()
        }
        assert result["is_replay"] is False
        assert session.get(Speaker, speaker.id).deleted_at is not None  # type: ignore[union-attr]
        assert all(session.get(AdjudicationDecision, row.id) for row in children.values())


def test_undo_merge_enforces_grace_window(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = _run(session)
        speaker = Speaker(display_name="Expired merge")
        session.add(speaker)
        session.flush()
        for label in ("S0", "S1"):
            _decision(
                session,
                run_id,
                label,
                Decision.ASSIGN,
                f"merge:expired:digest:{label}",
                speaker.id,
            )
        session.commit()

        with pytest.raises(UndoExpiredError):
            undo_merge(
                session,
                run_id=run_id,
                merge_nonce="expired",
                operator="ben",
                idempotency_key="undo-expired",
                grace_seconds=0,
            )
