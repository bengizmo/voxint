"""Compensating undo preserves the ledger and restores prior resolution."""

import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from voxint.adjudication.ledger import ConflictingReplayError, record_decision
from voxint.adjudication.resolver import effective_decisions
from voxint.adjudication.undo import (
    UndoDriftError,
    UndoError,
    UndoExpiredError,
    undo_decision,
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
    TranscriptSegment,
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


def _undo_decision(
    session: Session,
    run_id: uuid.UUID,
    decision_id: uuid.UUID,
    key: str,
    grace_seconds: float = 300,
) -> dict[str, object]:
    return undo_decision(
        session,
        run_id=run_id,
        decision_id=decision_id,
        operator="ben",
        idempotency_key=key,
        grace_seconds=grace_seconds,
    )


@pytest.mark.parametrize("ruling", [Decision.ASSIGN, Decision.EXCLUDE, Decision.UNKNOWN])
def test_undo_decision_restores_prior_ruling_and_keeps_the_ledger(
    session_factory: sessionmaker[Session], ruling: Decision
) -> None:
    with session_factory() as session:
        run_id = _run(session)
        speaker = Speaker(display_name="Undo decision")
        session.add(speaker)
        session.flush()
        previous = _decision(session, run_id, "S0", Decision.UNKNOWN, "previous")
        session.commit()
        original = _decision(
            session,
            run_id,
            "S0",
            ruling,
            f"ruling-{ruling.value}",
            speaker.id if ruling is Decision.ASSIGN else None,
        )
        session.commit()

        result = _undo_decision(session, run_id, original.id, "undo-ruling")
        session.commit()

        assert result["is_replay"] is False
        assert result["voided_decision_id"] == original.id
        assert effective_decisions(session, run_id)["S0"].id == previous.id
        revoke = session.get(AdjudicationDecision, result["revoke_decision_id"])
        assert revoke is not None
        assert revoke.decision == Decision.REVOKE.value
        assert revoke.voids_decision_id == original.id
        assert revoke.diarization_label == "S0"
        assert revoke.transcript_segment_id is None
        # Append-only: the voided ruling is still in the ledger.
        assert session.get(AdjudicationDecision, original.id) is not None
        # A plain ASSIGN undo never touches the roster.
        assert session.get(Speaker, speaker.id).deleted_at is None  # type: ignore[union-attr]


def test_undo_decision_with_no_prior_ruling_leaves_the_label_unruled(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = _run(session)
        original = _decision(session, run_id, "S0", Decision.EXCLUDE, "only")
        session.commit()

        _undo_decision(session, run_id, original.id, "undo-only")
        session.commit()

        assert "S0" not in effective_decisions(session, run_id)


def test_undo_decision_replays_under_any_key(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = _run(session)
        original = _decision(session, run_id, "S0", Decision.EXCLUDE, "exclude")
        session.commit()
        first = _undo_decision(session, run_id, original.id, "undo-first")
        session.commit()

        same_key = _undo_decision(session, run_id, original.id, "undo-first")
        other_key = _undo_decision(session, run_id, original.id, "undo-second")

        for replay in (same_key, other_key):
            assert replay["is_replay"] is True
            assert replay["revoke_decision_id"] == first["revoke_decision_id"]
        revokes = session.execute(
            select(func.count())
            .select_from(AdjudicationDecision)
            .where(AdjudicationDecision.voids_decision_id == original.id)
        ).scalar_one()
        assert revokes == 1


def test_undo_decision_replay_succeeds_after_the_grace_window(
    session_factory: sessionmaker[Session],
) -> None:
    # The replay check runs before the window check: a retried undo whose first
    # attempt committed must report success, not "expired".
    with session_factory() as session:
        run_id = _run(session)
        original = _decision(session, run_id, "S0", Decision.EXCLUDE, "late-replay")
        session.commit()
        first = _undo_decision(session, run_id, original.id, "undo-late")
        session.commit()

        replay = _undo_decision(session, run_id, original.id, "undo-late", grace_seconds=0)

        assert replay["is_replay"] is True
        assert replay["revoke_decision_id"] == first["revoke_decision_id"]


def test_undo_decision_enforces_grace_window(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = _run(session)
        original = _decision(session, run_id, "S0", Decision.EXCLUDE, "expired")
        session.commit()

        with pytest.raises(UndoExpiredError):
            _undo_decision(session, run_id, original.id, "undo-expired", grace_seconds=0)
        session.rollback()
        assert effective_decisions(session, run_id)["S0"].id == original.id


def test_undo_decision_rejects_a_later_ruling(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = _run(session)
        original = _decision(session, run_id, "S0", Decision.EXCLUDE, "first")
        session.commit()
        later = _decision(session, run_id, "S0", Decision.UNKNOWN, "later")
        session.commit()

        with pytest.raises(UndoDriftError):
            _undo_decision(session, run_id, original.id, "undo-drift")
        session.rollback()
        assert effective_decisions(session, run_id)["S0"].id == later.id


def test_undo_decision_ignores_rulings_on_other_labels(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = _run(session)
        original = _decision(session, run_id, "S0", Decision.EXCLUDE, "s0-ruling")
        session.commit()
        other = _decision(session, run_id, "S1", Decision.UNKNOWN, "s1-ruling")
        session.commit()

        _undo_decision(session, run_id, original.id, "undo-s0")
        session.commit()

        effective = effective_decisions(session, run_id)
        assert "S0" not in effective
        assert effective["S1"].id == other.id


def test_undo_decision_rejects_a_reused_key_for_another_ruling(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = _run(session)
        first = _decision(session, run_id, "S0", Decision.EXCLUDE, "s0")
        second = _decision(session, run_id, "S1", Decision.EXCLUDE, "s1")
        session.commit()
        _undo_decision(session, run_id, first.id, "shared-undo-key")
        session.commit()

        with pytest.raises(ConflictingReplayError):
            _undo_decision(session, run_id, second.id, "shared-undo-key")


def test_undo_decision_refuses_unknown_and_foreign_decisions(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = _run(session)
        other_run_id = _run(session)
        foreign = _decision(session, other_run_id, "S0", Decision.EXCLUDE, "foreign")
        session.commit()

        with pytest.raises(UndoError, match="no adjudication decision"):
            _undo_decision(session, run_id, uuid.uuid4(), "undo-missing")
        with pytest.raises(UndoError, match="plain label-scope decision"):
            _undo_decision(session, run_id, foreign.id, "undo-foreign")


def test_undo_decision_refuses_revoke_and_segment_scope_rows(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = _run(session)
        speaker = Speaker(display_name="Segment owner")
        session.add(speaker)
        segment = TranscriptSegment(
            pipeline_run_id=run_id,
            segment_index=0,
            start_seconds=0.0,
            end_seconds=1.0,
            raw_text="hello there",
            diarization_label="S0",
        )
        session.add(segment)
        session.flush()
        segment_rule = record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S0",
            decision=Decision.ASSIGN,
            speaker_id=speaker.id,
            transcript_segment_id=segment.id,
            operator="ben",
            idempotency_key="segment-rule",
        )
        ruling = _decision(session, run_id, "S1", Decision.EXCLUDE, "label-rule")
        session.commit()
        revoke_id = _undo_decision(session, run_id, ruling.id, "undo-label")[
            "revoke_decision_id"
        ]
        session.commit()
        assert isinstance(revoke_id, uuid.UUID)

        with pytest.raises(UndoError, match="plain label-scope decision"):
            _undo_decision(session, run_id, segment_rule.id, "undo-segment")
        with pytest.raises(UndoError, match="plain label-scope decision"):
            _undo_decision(session, run_id, revoke_id, "undo-revoke")


def test_undo_decision_routes_merge_and_enrollment_rows_elsewhere(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = _run(session)
        speaker = Speaker(display_name="Routed elsewhere")
        session.add(speaker)
        session.flush()
        merge_child = _decision(
            session, run_id, "S0", Decision.ASSIGN, "merge:nonce:digest:S0", speaker.id
        )
        enrolled = _decision(session, run_id, "S1", Decision.ASSIGN, "enroll", speaker.id)
        _embedding(session, speaker.id, enrolled)
        session.commit()

        with pytest.raises(UndoError, match="merge undo endpoint"):
            _undo_decision(session, run_id, merge_child.id, "undo-merge-child")
        with pytest.raises(UndoError, match="enrollment undo endpoint"):
            _undo_decision(session, run_id, enrolled.id, "undo-enrolled")
        session.rollback()
        effective = effective_decisions(session, run_id)
        assert effective["S0"].id == merge_child.id
        assert effective["S1"].id == enrolled.id
