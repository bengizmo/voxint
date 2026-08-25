"""Tier grading and folding (issue #159): gate boundaries, the NULL-margin
one-speaker-roster rule, unavailable vs weak, and best-appearance-wins."""

import uuid

from voxint.speakers.matching import MatchingGates
from voxint.speakers.tiers import MatchTier, TierEvidence, grade, tier_for

GATES = MatchingGates()  # library defaults mirror Settings


def _evidence(
    *,
    similarity: float | None = 0.75,
    margin: float | None = 0.10,
    vote_agreement: float | None = 0.80,
    eligible_turns: int = 5,
    eligible_seconds: float = 30.0,
    roster_size: int | None = 4,
    available: bool = True,
) -> TierEvidence:
    return TierEvidence(
        run_id=uuid.uuid4(),
        label="S0",
        available=available,
        similarity=similarity,
        margin=margin,
        vote_agreement=vote_agreement,
        eligible_turns=eligible_turns,
        eligible_seconds=eligible_seconds,
        roster_size=roster_size,
    )


def test_grounded_gate_clears_strong() -> None:
    assert grade(_evidence(), GATES) is MatchTier.STRONG


def test_exact_grounded_boundaries_are_inclusive() -> None:
    at_gate = _evidence(
        similarity=GATES.grounded_min_cosine,
        margin=GATES.grounded_min_margin,
        vote_agreement=GATES.grounded_min_vote_agreement,
        eligible_turns=GATES.grounded_min_turns,
        eligible_seconds=GATES.grounded_min_seconds,
    )
    assert grade(at_gate, GATES) is MatchTier.STRONG


def test_accept_but_not_grounded_is_moderate() -> None:
    # Clears accept (0.60/0.05/0.60/2/6) but sits below the grounded cosine.
    assert (
        grade(_evidence(similarity=0.65, margin=0.06, vote_agreement=0.62), GATES)
        is MatchTier.MODERATE
    )


def test_below_accept_is_weak() -> None:
    assert grade(_evidence(similarity=0.55), GATES) is MatchTier.WEAK
    assert grade(_evidence(margin=0.01), GATES) is MatchTier.WEAK
    assert grade(_evidence(vote_agreement=0.40), GATES) is MatchTier.WEAK
    assert grade(_evidence(eligible_turns=1, eligible_seconds=3.0), GATES) is MatchTier.WEAK


def test_null_margin_passes_only_for_one_speaker_roster() -> None:
    # One-speaker roster: margin is undefined (infinite) — passes.
    assert grade(_evidence(margin=None, roster_size=1), GATES) is MatchTier.STRONG
    # NULL margin with a multi-speaker roster is malformed — never passes.
    assert grade(_evidence(margin=None, roster_size=3), GATES) is MatchTier.WEAK


def test_missing_diagnostics_are_unavailable_never_weak() -> None:
    absent = TierEvidence(run_id=uuid.uuid4(), label="S0", available=False)
    assert grade(absent, GATES) is None
    summary = tier_for([absent], GATES)
    assert summary.tier is None
    assert summary.unavailable == 1
    assert summary.weak == 0
    assert summary.has_voice_evidence


def test_null_numbers_are_unavailable() -> None:
    assert grade(_evidence(similarity=None), GATES) is None
    assert grade(_evidence(vote_agreement=None), GATES) is None


def test_fold_best_appearance_wins_with_honest_counts() -> None:
    strong = _evidence()
    weak = _evidence(similarity=0.50)
    absent = TierEvidence(run_id=uuid.uuid4(), label="S1", available=False)
    summary = tier_for([weak, strong, absent], GATES)
    assert summary.tier is MatchTier.STRONG
    assert (summary.strong, summary.moderate, summary.weak, summary.unavailable) == (
        1,
        0,
        1,
        1,
    )
    assert len(summary.evidence) == 3


def test_fold_without_evidence_is_none() -> None:
    summary = tier_for([], GATES)
    assert summary.tier is None
    assert not summary.has_voice_evidence
