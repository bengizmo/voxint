"""Three-band confidence policy for speaker-match display (issue #114).

Classifies each diarization label's match evidence into one of three bands:

- ``AUTO_ATTRIBUTE``: safe to display without operator review (the confident
  band). Maps to the grounded gate in ``MatchingGates``.
- ``REVIEW``: the uncertain tail routed to the operator. Maps to accepted-
  but-not-grounded evidence, or near-misses worth surfacing.
- ``ABSTAIN``: insufficient evidence to propose an identity. Ineligible
  labels, clear rejects, or missing data.

The band is computed at read time from recorded ``match_candidates`` evidence
and the current ``MatchingGates``, so recalibrating thresholds retroactively
reclassifies all runs without a backfill.

``band_for`` shares gate predicates with :func:`~voxint.speakers.tiers.grade`
so tier chips and policy bands can never disagree on what constitutes grounded
or accepted evidence.
"""

import enum
import uuid
from dataclasses import dataclass

from voxint.speakers.matching import MatchingGates
from voxint.speakers.tiers import passes_accept, passes_grounded


class MatchBand(enum.StrEnum):
    AUTO_ATTRIBUTE = "auto_attribute"
    REVIEW = "review"
    ABSTAIN = "abstain"


# Operator-facing plain language keyed to match_candidates reason codes.
_REASON_COPY: dict[str | None, str] = {
    "too_few_turns": "Not enough clear speech to match.",
    "too_little_speech": "Not enough clear speech to match.",
    "no_eligible_turns": "Not enough clear speech to match.",
    "no_roster": "No known speakers to compare against.",
    "below_cosine": "Voice not distinctive enough to match.",
    "degenerate_centroid": "Voice not distinctive enough to match.",
    "below_margin": "No clear winner among known speakers.",
    "below_vote_agreement": "No clear winner among known speakers.",
}


@dataclass(frozen=True)
class BandResult:
    """The band, the plain-language reason, and whether to name the candidate."""

    band: MatchBand
    reason: str
    candidate_speaker_id: uuid.UUID | None
    candidate_prompt_allowed: bool


def band_for(
    *,
    decision: str | None,
    reason: str | None,
    similarity: float | None,
    margin: float | None,
    vote_agreement: float | None,
    eligible_turns: int,
    eligible_seconds: float,
    roster_size: int | None,
    top_speaker_id: uuid.UUID | None,
    gates: MatchingGates,
) -> BandResult:
    """Classify one label's match evidence into a confidence band.

    Uses the shared gate predicates from :mod:`~voxint.speakers.tiers` so the
    band boundaries are identical to the tier chip boundaries.  The classification
    is based on the **current** gates applied to the recorded evidence numbers,
    not on the historical ``decision`` field — so recalibrating gate values
    retroactively reclassifies runs without a backfill.

    When no ``match_candidates`` row exists (pre-migration-0032 run or a label
    never evaluated), pass all-None/zero values — the function returns ABSTAIN.
    """
    fields = (similarity, margin, vote_agreement, eligible_turns,
              eligible_seconds, roster_size)

    # Classify using current gates against recorded numeric evidence.
    if passes_grounded(*fields, gates):
        return BandResult(
            band=MatchBand.AUTO_ATTRIBUTE,
            reason="Strong enough to trust on its own.",
            candidate_speaker_id=top_speaker_id,
            candidate_prompt_allowed=True,
        )

    if passes_accept(*fields, gates):
        ambiguous = (
            margin is not None
            and margin < gates.grounded_min_margin
        )
        return BandResult(
            band=MatchBand.REVIEW,
            reason=(
                "Similar voices found." if ambiguous
                else "Not strong enough to confirm without your check."
            ),
            candidate_speaker_id=top_speaker_id,
            candidate_prompt_allowed=not ambiguous,
        )

    # Below the accept gate or evidence unavailable — ABSTAIN.
    return BandResult(
        band=MatchBand.ABSTAIN,
        reason=_REASON_COPY.get(reason, "Not enough evidence to suggest a speaker."),
        candidate_speaker_id=top_speaker_id if decision == "rejected" else None,
        candidate_prompt_allowed=False,
    )
