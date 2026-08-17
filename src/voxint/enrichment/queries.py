"""Read side of the enrichment draft layer: derived state + listing helpers.

There is **no stored review state** — the effective state of a candidate is
derived at read time by one rule (mirroring ``adjudication/resolver.py``):

1. a ``profile_review_decisions`` row wins — ``accepted``/``rejected``,
   terminal;
2. else a set ``superseded_by_producer_run_id`` means ``superseded``;
3. else the claim is ``proposed``.

:func:`effective_state` is the Python rule; :func:`effective_state_sql` is the
SQL mirror used by listing queries. They are parity-tested together — change
one, change both.

``accepted_claims`` deliberately returns a *collection*: several accepted
names or bios may coexist (different producers, different evidence) and this
layer does not pick a canonical one — profile canonicalization is console
work, and identity is never resolved here at all.
"""

import enum
import uuid
from dataclasses import dataclass

from sqlalchemy import ColumnElement, case, select
from sqlalchemy.orm import Session, selectinload

from voxint.db.models import (
    EnrichmentCandidate,
    EnrichmentCandidateEvidence,
    EnrichmentProducerRun,
    ProfileDecision,
    ProfileReviewDecision,
)
from voxint.enrichment.drafts import EnrichmentScope, _scope_filter


class CandidateState(enum.StrEnum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


def effective_state(
    candidate: EnrichmentCandidate, decision: ProfileReviewDecision | None
) -> CandidateState:
    """The one derivation rule — keep in lockstep with :func:`effective_state_sql`."""
    if decision is not None:
        return (
            CandidateState.ACCEPTED
            if decision.decision == ProfileDecision.ACCEPT.value
            else CandidateState.REJECTED
        )
    if candidate.superseded_by_producer_run_id is not None:
        return CandidateState.SUPERSEDED
    return CandidateState.PROPOSED


def effective_state_sql() -> ColumnElement[str]:
    """SQL mirror of :func:`effective_state` over an outer-joined decision row."""
    return case(
        (
            ProfileReviewDecision.id.is_not(None),
            case(
                (
                    ProfileReviewDecision.decision == ProfileDecision.ACCEPT.value,
                    CandidateState.ACCEPTED.value,
                ),
                else_=CandidateState.REJECTED.value,
            ),
        ),
        (
            EnrichmentCandidate.superseded_by_producer_run_id.is_not(None),
            CandidateState.SUPERSEDED.value,
        ),
        else_=CandidateState.PROPOSED.value,
    )


@dataclass(frozen=True)
class CandidateView:
    candidate: EnrichmentCandidate
    evidence: tuple[EnrichmentCandidateEvidence, ...]
    decision: ProfileReviewDecision | None
    state: CandidateState


def _views(
    session: Session, *criteria: ColumnElement[bool]
) -> list[CandidateView]:
    rows = session.execute(
        select(EnrichmentCandidate, ProfileReviewDecision)
        .outerjoin(
            ProfileReviewDecision,
            ProfileReviewDecision.candidate_id == EnrichmentCandidate.id,
        )
        .options(
            selectinload(EnrichmentCandidate.evidence),
            # Triage (#42) reads the producing producer per candidate for the
            # name-match adapter and cross-producer agreement — eager-load to
            # avoid an N+1 across a run's / speaker's candidate list.
            selectinload(EnrichmentCandidate.producer_run),
        )
        .where(*criteria)
        .order_by(EnrichmentCandidate.created_at, EnrichmentCandidate.id)
    ).all()
    return [
        CandidateView(
            candidate=candidate,
            evidence=tuple(candidate.evidence),
            decision=decision,
            state=effective_state(candidate, decision),
        )
        for candidate, decision in rows
    ]


def candidates_for_run(
    session: Session, pipeline_run_id: uuid.UUID
) -> list[CandidateView]:
    """All claims targeting a run or its diarization labels, with derived state."""
    return _views(session, EnrichmentCandidate.pipeline_run_id == pipeline_run_id)


def candidates_for_speaker(
    session: Session, speaker_id: uuid.UUID
) -> list[CandidateView]:
    return _views(session, EnrichmentCandidate.speaker_id == speaker_id)


def accepted_claims(session: Session, speaker_id: uuid.UUID) -> list[CandidateView]:
    """Accepted claims about a speaker — a collection, never a canonical profile."""
    return [
        view
        for view in candidates_for_speaker(session, speaker_id)
        if view.state is CandidateState.ACCEPTED
    ]


def latest_producer_run(
    session: Session, producer: str, scope: EnrichmentScope
) -> EnrichmentProducerRun | None:
    """The newest generation recorded for this producer + scope, if any."""
    return session.execute(
        select(EnrichmentProducerRun)
        .where(EnrichmentProducerRun.producer == producer, *_scope_filter(scope))
        .order_by(EnrichmentProducerRun.generation.desc())
        .limit(1)
    ).scalar_one_or_none()
