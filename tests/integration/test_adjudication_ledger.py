"""The adjudication ledger's append-only + idempotent-replay contract."""

import uuid

import pytest
from sqlalchemy import delete, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session, sessionmaker

from voxint.adjudication.ledger import ConflictingReplayError, record_decision
from voxint.db.models import AdjudicationDecision, Decision, MediaItem, Speaker
from voxint.domain_packs.base import load_default
from voxint.pipeline.engine import submit


def seed(session: Session) -> tuple[uuid.UUID, uuid.UUID]:
    media = MediaItem(source_path=f"/data/media/adj-{uuid.uuid4()}.wav")
    speaker = Speaker(display_name=f"Speaker {uuid.uuid4()}")
    session.add_all([media, speaker])
    session.flush()
    run = submit(session, media.id, domain_pack=load_default().to_mapping())
    return run.id, speaker.id


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
