"""Integration behavior of the profile-review writer + derived state (issue #37).

Covers: accept/reject recording + derived state, terminal decisions, idempotent
and conflicting replays, stale-candidate protection, the accepted-claims
collection, Python/SQL derivation parity — and the load-bearing invariant that
accepting a name changes neither the roster nor attribution resolution.
"""

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from voxint.adjudication.resolver import Resolution, label_states
from voxint.db.models import (
    AdjudicationDecision,
    ClaimField,
    EnrichmentCandidate,
    ProfileDecision,
    ProfileReviewDecision,
    Speaker,
)
from voxint.enrichment.drafts import (
    CandidateDraft,
    EnrichmentScope,
    UrlEvidence,
    record_producer_run,
)
from voxint.enrichment.queries import (
    CandidateState,
    accepted_claims,
    candidates_for_speaker,
    effective_state_sql,
)
from voxint.enrichment.review import (
    ConflictingReplayError,
    StaleCandidateError,
    record_profile_decision,
)

NOW = datetime.now(tz=UTC)


@pytest.fixture()
def session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    with session_factory() as sess:
        yield sess


@pytest.fixture()
def speaker_id(session: Session) -> uuid.UUID:
    speaker = Speaker(display_name="Review Target Speaker")
    session.add(speaker)
    session.commit()
    return speaker.id


def _draft(
    speaker_id: uuid.UUID, field: ClaimField = ClaimField.BIO, value: str = "A host."
) -> CandidateDraft:
    return CandidateDraft(
        target=EnrichmentScope.speaker(speaker_id),
        field=field,
        value=value,
        evidence=(UrlEvidence(url="https://example.com/about"),),
    )


def _record_candidates(
    session: Session, speaker_id: uuid.UUID, *drafts: CandidateDraft
) -> list[uuid.UUID]:
    record_producer_run(
        session,
        producer="web_researcher",
        producer_version="1.0",
        scope=EnrichmentScope.speaker(speaker_id),
        covered_fields=(ClaimField.NAME, ClaimField.BIO, ClaimField.LINK),
        candidates=drafts,
        idempotency_key=f"pr-{uuid.uuid4()}",
        started_at=NOW,
        completed_at=NOW,
    )
    session.commit()
    # created_at is the transaction timestamp, identical across one insert
    # batch, so listing order tie-breaks on random UUIDs — map ids back to
    # drafts by content instead of position.
    by_content = {
        (v.candidate.field, v.candidate.value): v.candidate.id
        for v in candidates_for_speaker(session, speaker_id)
        if v.state is CandidateState.PROPOSED
    }
    return [by_content[(d.field.value, d.value)] for d in drafts]


def test_accept_and_reject_flow(session: Session, speaker_id: uuid.UUID) -> None:
    bio_id, link_id = _record_candidates(
        session,
        speaker_id,
        _draft(speaker_id, ClaimField.BIO, "Longtime host."),
        _draft(speaker_id, ClaimField.LINK, "https://example.com/jane"),
    )
    accepted = record_profile_decision(
        session,
        candidate_id=bio_id,
        decision=ProfileDecision.ACCEPT,
        operator="ben",
        idempotency_key="d-accept",
        note="matches the episode metadata",
    )
    rejected = record_profile_decision(
        session,
        candidate_id=link_id,
        decision=ProfileDecision.REJECT,
        operator="ben",
        idempotency_key="d-reject",
    )
    session.commit()
    assert accepted.note == "matches the episode metadata"
    assert rejected.note is None

    states = {
        v.candidate.id: v.state for v in candidates_for_speaker(session, speaker_id)
    }
    assert states[bio_id] is CandidateState.ACCEPTED
    assert states[link_id] is CandidateState.REJECTED

    claims = accepted_claims(session, speaker_id)
    assert [(v.candidate.field, v.candidate.value) for v in claims] == [
        ("bio", "Longtime host.")
    ]


def test_decisions_are_terminal(session: Session, speaker_id: uuid.UUID) -> None:
    (candidate_id,) = _record_candidates(session, speaker_id, _draft(speaker_id))
    record_profile_decision(
        session,
        candidate_id=candidate_id,
        decision=ProfileDecision.ACCEPT,
        operator="ben",
        idempotency_key="d1",
    )
    session.commit()
    with pytest.raises(StaleCandidateError, match="already decided"):
        record_profile_decision(
            session,
            candidate_id=candidate_id,
            decision=ProfileDecision.REJECT,
            operator="ben",
            idempotency_key="d2",
        )


def test_idempotent_and_conflicting_replay(
    session: Session, speaker_id: uuid.UUID
) -> None:
    (candidate_id,) = _record_candidates(session, speaker_id, _draft(speaker_id))
    first = record_profile_decision(
        session,
        candidate_id=candidate_id,
        decision=ProfileDecision.ACCEPT,
        operator="ben",
        idempotency_key="replay",
    )
    session.commit()
    replay = record_profile_decision(
        session,
        candidate_id=candidate_id,
        decision=ProfileDecision.ACCEPT,
        operator="ben",
        idempotency_key="replay",
    )
    assert replay.id == first.id
    with pytest.raises(ConflictingReplayError):
        record_profile_decision(
            session,
            candidate_id=candidate_id,
            decision=ProfileDecision.REJECT,
            operator="ben",
            idempotency_key="replay",
        )


def test_deciding_superseded_or_missing_candidate_raises(
    session: Session, speaker_id: uuid.UUID
) -> None:
    (candidate_id,) = _record_candidates(session, speaker_id, _draft(speaker_id))
    # a newer generation retires the proposal
    record_producer_run(
        session,
        producer="web_researcher",
        producer_version="1.0",
        scope=EnrichmentScope.speaker(speaker_id),
        covered_fields=(ClaimField.BIO,),
        candidates=(),
        idempotency_key="gen2",
        started_at=NOW,
        completed_at=NOW,
    )
    session.commit()
    with pytest.raises(StaleCandidateError, match="superseded"):
        record_profile_decision(
            session,
            candidate_id=candidate_id,
            decision=ProfileDecision.ACCEPT,
            operator="ben",
            idempotency_key="d1",
        )
    with pytest.raises(StaleCandidateError, match="no such candidate"):
        record_profile_decision(
            session,
            candidate_id=uuid.uuid4(),
            decision=ProfileDecision.ACCEPT,
            operator="ben",
            idempotency_key="d2",
        )


def test_accepting_a_name_never_grounds_identity(
    session: Session, speaker_id: uuid.UUID
) -> None:
    """The invariant: suggestions ABOUT identity, never identity.

    Accepting a name claim must leave the roster row and attribution
    resolution exactly as they were — no display_name change, no
    adjudication ledger rows, no resolved labels.
    """
    (candidate_id,) = _record_candidates(
        session, speaker_id, _draft(speaker_id, ClaimField.NAME, "Jane Interviewee")
    )
    before_name = session.get(Speaker, speaker_id)
    assert before_name is not None
    record_profile_decision(
        session,
        candidate_id=candidate_id,
        decision=ProfileDecision.ACCEPT,
        operator="ben",
        idempotency_key="d1",
    )
    session.commit()

    speaker = session.get(Speaker, speaker_id)
    assert speaker is not None
    assert speaker.display_name == "Review Target Speaker"
    assert session.execute(select(AdjudicationDecision)).scalars().all() == []
    # the accepted claim is visible as a claim — and only as a claim
    assert [
        v.candidate.value for v in accepted_claims(session, speaker_id)
    ] == ["Jane Interviewee"]


def test_effective_state_python_sql_parity(
    session: Session, speaker_id: uuid.UUID
) -> None:
    """The Python rule and its SQL mirror must agree on every state."""
    proposed_id, accepted_id, rejected_id = _record_candidates(
        session,
        speaker_id,
        _draft(speaker_id, ClaimField.BIO, "proposed"),
        _draft(speaker_id, ClaimField.BIO, "accepted"),
        _draft(speaker_id, ClaimField.BIO, "rejected"),
    )
    record_profile_decision(
        session,
        candidate_id=accepted_id,
        decision=ProfileDecision.ACCEPT,
        operator="ben",
        idempotency_key="p1",
    )
    record_profile_decision(
        session,
        candidate_id=rejected_id,
        decision=ProfileDecision.REJECT,
        operator="ben",
        idempotency_key="p2",
    )
    session.commit()
    # supersede the remaining proposed row (accepted/rejected survive)
    record_producer_run(
        session,
        producer="web_researcher",
        producer_version="1.0",
        scope=EnrichmentScope.speaker(speaker_id),
        covered_fields=(ClaimField.BIO,),
        candidates=(_draft(speaker_id, ClaimField.BIO, "fresh"),),
        idempotency_key="gen2",
        started_at=NOW,
        completed_at=NOW,
    )
    session.commit()

    sql_states = dict(
        session.execute(
            select(EnrichmentCandidate.id, effective_state_sql()).outerjoin(
                ProfileReviewDecision,
                ProfileReviewDecision.candidate_id == EnrichmentCandidate.id,
            )
        ).all()
    )
    python_states = {
        v.candidate.id: v.state for v in candidates_for_speaker(session, speaker_id)
    }
    assert {cid: state.value for cid, state in python_states.items()} == sql_states
    assert python_states[proposed_id] is CandidateState.SUPERSEDED
    assert python_states[accepted_id] is CandidateState.ACCEPTED
    assert python_states[rejected_id] is CandidateState.REJECTED
    assert CandidateState.PROPOSED in python_states.values()


def test_attribution_resolver_ignores_enrichment(
    session: Session, speaker_id: uuid.UUID
) -> None:
    """Belt and braces: label_states derives from the adjudication world only,
    so a run with enrichment activity but no proposals/decisions stays fully
    unresolved."""
    from voxint.db.models import DiarizationTurn, MediaItem, PipelineRun

    media = MediaItem(source_path="incoming/enrichment/resolver.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id)
    session.add(run)
    session.flush()
    session.add(
        DiarizationTurn(
            pipeline_run_id=run.id,
            turn_index=0,
            start_seconds=0.0,
            end_seconds=4.0,
            label="SPEAKER_00",
            skip_reason="too_short",
        )
    )
    session.commit()
    record_producer_run(
        session,
        producer="name_miner",
        producer_version="1.0",
        scope=EnrichmentScope.run(run.id),
        covered_fields=(ClaimField.NAME,),
        candidates=(
            CandidateDraft(
                target=EnrichmentScope.run_label(run.id, "SPEAKER_00"),
                field=ClaimField.NAME,
                value="Jane Interviewee",
                evidence=(UrlEvidence(url="https://example.com/x"),),
            ),
        ),
        idempotency_key="res-1",
        started_at=NOW,
        completed_at=NOW,
    )
    session.commit()
    (state,) = label_states(session, run.id)
    assert state.label == "SPEAKER_00"
    assert state.resolution is Resolution.UNRESOLVED
