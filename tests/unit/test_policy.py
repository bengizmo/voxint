"""Unit tests for the three-band speaker-match policy."""

import math
import uuid

import pytest

from voxint.speakers.matching import MatchingGates
from voxint.speakers.policy import BandResult, MatchBand, band_for

GATES = MatchingGates()


def _band(**overrides: object) -> BandResult:
    """Return a policy result from valid, strongly grounded evidence."""
    values = {
        "decision": "accepted",
        "reason": "accepted",
        "similarity": 0.80,
        "margin": 0.12,
        "vote_agreement": 0.80,
        "eligible_turns": 5,
        "eligible_seconds": 20.0,
        "roster_size": 4,
        "top_speaker_id": uuid.uuid4(),
        "gates": GATES,
    }
    values.update(overrides)
    return band_for(**values)  # type: ignore[arg-type]


def test_clear_grounded_evidence_is_auto_attributed() -> None:
    speaker_id = uuid.uuid4()

    result = _band(top_speaker_id=speaker_id)

    assert result == BandResult(
        band=MatchBand.AUTO_ATTRIBUTE,
        reason="Strong enough to trust on its own.",
        candidate_speaker_id=speaker_id,
        candidate_prompt_allowed=True,
    )


def test_clear_accepted_but_ungrounded_evidence_requires_review() -> None:
    speaker_id = uuid.uuid4()

    result = _band(similarity=0.65, top_speaker_id=speaker_id)

    assert result == BandResult(
        band=MatchBand.REVIEW,
        reason="Not strong enough to confirm without your check.",
        candidate_speaker_id=speaker_id,
        candidate_prompt_allowed=True,
    )


def test_clear_rejection_abstains() -> None:
    speaker_id = uuid.uuid4()

    result = _band(
        decision="rejected",
        reason="below_cosine",
        similarity=0.40,
        top_speaker_id=speaker_id,
    )

    assert result == BandResult(
        band=MatchBand.ABSTAIN,
        reason="Voice not distinctive enough to match.",
        candidate_speaker_id=speaker_id,
        candidate_prompt_allowed=False,
    )


def test_exact_grounded_gate_boundaries_are_auto_attributed() -> None:
    result = _band(
        similarity=GATES.grounded_min_cosine,
        margin=GATES.grounded_min_margin,
        vote_agreement=GATES.grounded_min_vote_agreement,
        eligible_turns=GATES.grounded_min_turns,
        eligible_seconds=GATES.grounded_min_seconds,
    )

    assert result.band is MatchBand.AUTO_ATTRIBUTE


@pytest.mark.parametrize(
    ("field", "just_below"),
    [
        (
            "similarity",
            math.nextafter(GATES.grounded_min_cosine, -math.inf),
        ),
        ("margin", math.nextafter(GATES.grounded_min_margin, -math.inf)),
        (
            "vote_agreement",
            math.nextafter(GATES.grounded_min_vote_agreement, -math.inf),
        ),
        ("eligible_turns", GATES.grounded_min_turns - 1),
        (
            "eligible_seconds",
            math.nextafter(GATES.grounded_min_seconds, -math.inf),
        ),
    ],
)
def test_just_below_each_grounded_gate_requires_review(
    field: str, just_below: float | int
) -> None:
    evidence = {
        "similarity": GATES.grounded_min_cosine,
        "margin": GATES.grounded_min_margin,
        "vote_agreement": GATES.grounded_min_vote_agreement,
        "eligible_turns": GATES.grounded_min_turns,
        "eligible_seconds": GATES.grounded_min_seconds,
    }
    evidence[field] = just_below

    assert _band(**evidence).band is MatchBand.REVIEW


def test_review_suppresses_candidate_prompt_for_ambiguous_margin() -> None:
    speaker_id = uuid.uuid4()

    result = _band(
        margin=math.nextafter(GATES.grounded_min_margin, -math.inf),
        top_speaker_id=speaker_id,
    )

    assert result.band is MatchBand.REVIEW
    assert result.reason == "Similar voices found."
    assert result.candidate_speaker_id == speaker_id
    assert result.candidate_prompt_allowed is False


def test_review_allows_candidate_prompt_when_margin_clears_grounded_gate() -> None:
    result = _band(
        similarity=math.nextafter(GATES.grounded_min_cosine, -math.inf),
        margin=GATES.grounded_min_margin,
    )

    assert result.band is MatchBand.REVIEW
    assert result.reason == "Not strong enough to confirm without your check."
    assert result.candidate_prompt_allowed is True


def test_null_margin_is_infinite_for_single_speaker_roster() -> None:
    result = _band(margin=None, roster_size=1)

    assert result.band is MatchBand.AUTO_ATTRIBUTE
    assert result.candidate_prompt_allowed is True


@pytest.mark.parametrize("roster_size", [None, 0, 2])
def test_null_margin_does_not_pass_for_non_single_speaker_roster(
    roster_size: int | None,
) -> None:
    result = _band(margin=None, roster_size=roster_size)

    assert result.band is MatchBand.ABSTAIN
    assert result.candidate_prompt_allowed is False


@pytest.mark.parametrize(
    ("reason", "expected_copy"),
    [
        ("no_eligible_turns", "Not enough clear speech to match."),
        ("too_few_turns", "Not enough clear speech to match."),
        ("too_little_speech", "Not enough clear speech to match."),
        ("no_roster", "No known speakers to compare against."),
        ("degenerate_centroid", "Voice not distinctive enough to match."),
    ],
)
def test_ineligible_decisions_abstain_with_plain_language_reason(
    reason: str, expected_copy: str
) -> None:
    result = _band(
        decision="ineligible",
        reason=reason,
        similarity=None,
        margin=None,
        vote_agreement=None,
        eligible_turns=0,
        eligible_seconds=0.0,
        roster_size=None,
        top_speaker_id=None,
    )

    assert result.band is MatchBand.ABSTAIN
    assert result.reason == expected_copy
    assert result.candidate_speaker_id is None
    assert result.candidate_prompt_allowed is False


@pytest.mark.parametrize(
    ("reason", "expected_copy"),
    [
        ("below_cosine", "Voice not distinctive enough to match."),
        ("below_margin", "No clear winner among known speakers."),
        ("below_vote_agreement", "No clear winner among known speakers."),
    ],
)
def test_rejected_decisions_abstain_with_plain_language_reason(
    reason: str, expected_copy: str
) -> None:
    speaker_id = uuid.uuid4()

    result = _band(
        decision="rejected",
        reason=reason,
        similarity=0.40,
        margin=0.02,
        vote_agreement=0.30,
        eligible_turns=1,
        eligible_seconds=3.0,
        top_speaker_id=speaker_id,
    )

    assert result.band is MatchBand.ABSTAIN
    assert result.reason == expected_copy
    assert result.candidate_speaker_id == speaker_id
    assert result.candidate_prompt_allowed is False


def test_none_decision_represents_missing_match_candidates_row() -> None:
    result = _band(
        decision=None,
        reason=None,
        similarity=None,
        margin=None,
        vote_agreement=None,
        eligible_turns=0,
        eligible_seconds=0.0,
        roster_size=None,
        top_speaker_id=None,
    )

    assert result == BandResult(
        band=MatchBand.ABSTAIN,
        reason="Not enough evidence to suggest a speaker.",
        candidate_speaker_id=None,
        candidate_prompt_allowed=False,
    )


def test_unknown_reason_code_uses_fallback_copy() -> None:
    result = _band(
        decision="rejected",
        reason="future_reason",
        similarity=0.40,
        margin=0.02,
        vote_agreement=0.30,
        eligible_turns=1,
        eligible_seconds=3.0,
    )
    assert result.band is MatchBand.ABSTAIN
    assert result.reason == "Not enough evidence to suggest a speaker."


def test_current_default_grounded_evidence_remains_auto_attributed() -> None:
    """Characterize compatibility with today's default grounded gates."""
    gates = MatchingGates()
    assert gates.grounded_min_cosine == pytest.approx(0.70)
    assert gates.grounded_min_margin == pytest.approx(0.08)
    assert gates.grounded_min_vote_agreement == pytest.approx(0.67)
    assert gates.grounded_min_turns == 3
    assert gates.grounded_min_seconds == pytest.approx(10.0)

    result = _band(
        similarity=0.70,
        margin=0.08,
        vote_agreement=0.67,
        eligible_turns=3,
        eligible_seconds=10.0,
        gates=gates,
    )

    assert result.band is MatchBand.AUTO_ATTRIBUTE
