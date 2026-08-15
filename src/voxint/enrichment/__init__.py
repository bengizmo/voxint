"""Enrichment draft layer (issue #37): evidence-backed machine-derived claims.

Producers write reviewable drafts through :mod:`voxint.enrichment.drafts`;
humans accept/reject through :mod:`voxint.enrichment.review`; readers derive
effective state through :mod:`voxint.enrichment.queries`. Drafts are
suggestions *about* identity, never identity — nothing in this package writes
``speakers``, ``speaker_assignments``, or ``adjudication_decisions``.
"""

from voxint.enrichment.drafts import (
    CandidateDraft,
    EnrichmentDraftError,
    EnrichmentScope,
    MetadataEvidence,
    TranscriptEvidence,
    UrlEvidence,
    record_producer_run,
)
from voxint.enrichment.queries import (
    CandidateState,
    CandidateView,
    accepted_claims,
    candidates_for_run,
    candidates_for_speaker,
    effective_state,
    latest_producer_run,
)
from voxint.enrichment.review import (
    ConflictingReplayError,
    StaleCandidateError,
    record_profile_decision,
)

__all__ = [
    "CandidateDraft",
    "CandidateState",
    "CandidateView",
    "ConflictingReplayError",
    "EnrichmentDraftError",
    "EnrichmentScope",
    "MetadataEvidence",
    "StaleCandidateError",
    "TranscriptEvidence",
    "UrlEvidence",
    "accepted_claims",
    "candidates_for_run",
    "candidates_for_speaker",
    "effective_state",
    "latest_producer_run",
    "record_producer_run",
    "record_profile_decision",
]
