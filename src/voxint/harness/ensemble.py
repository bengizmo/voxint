"""Two-voter verdict fusion — the ONLY place voter agreement is decided.

Each embedding voter (a model with its own embedding space) is scored
independently by :mod:`voxint.harness.agreement`; only its typed
:class:`~voxint.harness.agreement.LabelResult` crosses into this module.
Nothing here imports numpy or accepts a vector, so cross-embedding-space
cosine is structurally impossible at the ensemble layer — the invariant the
guardrail tests pin.

Both combiners are conservative AND-gates: a silver label requires both voters
to agree; any contradiction flag or single-voter confidence routes to human
review rather than a guess.
"""

from dataclasses import dataclass

from voxint.harness.agreement import (
    CONFIDENT_HOST_PRESENT,
    NO_CURATED_HOST_DETECTED,
    LabelResult,
)

# Ensemble verdicts (distinct from the single-voter agreement verdicts).
SILVER_HOST_PRESENT = "SILVER_HOST_PRESENT"
SILVER_NO_HOST = "SILVER_NO_HOST"
FLAG_REVIEW = "FLAG_REVIEW"
ENSEMBLE_ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class EnsembleDecision:
    """A fused two-voter decision + the agreement class behind it."""

    verdict: str
    reason: str
    agreement: str


def combine_curated(
    voter_a: LabelResult,
    voter_b: LabelResult,
    *,
    voter_a_name: str = "voter_a",
    voter_b_name: str = "voter_b",
) -> EnsembleDecision:
    """AND-gate two curated-item voter verdicts into an ensemble decision.

    Precedence: contradiction (either voter) > both confident on the same slot
    (silver positive) > both confident on different slots (review) > exactly
    one confident (review) > agree-abstain.
    """
    if voter_a.contradiction or voter_b.contradiction:
        return EnsembleDecision(
            verdict=FLAG_REVIEW, reason="contradiction", agreement="contradiction"
        )
    va, vb = voter_a.verdict, voter_b.verdict
    if va == CONFIDENT_HOST_PRESENT and vb == CONFIDENT_HOST_PRESENT:
        if voter_a.host_slot == voter_b.host_slot:
            return EnsembleDecision(
                verdict=SILVER_HOST_PRESENT,
                reason="both_confident_same_slot",
                agreement="agree_present",
            )
        return EnsembleDecision(
            verdict=FLAG_REVIEW,
            reason="both_confident_slot_mismatch",
            agreement="slot_disagree",
        )
    if va == CONFIDENT_HOST_PRESENT or vb == CONFIDENT_HOST_PRESENT:
        only = voter_a_name if va == CONFIDENT_HOST_PRESENT else voter_b_name
        return EnsembleDecision(
            verdict=FLAG_REVIEW,
            reason=f"single_voter_confident:{only}",
            agreement="one_confident",
        )
    return EnsembleDecision(
        verdict=ENSEMBLE_ABSTAIN,
        reason="no_confident_signal",
        agreement="agree_abstain",
    )


def combine_neg_control(voter_a: LabelResult, voter_b: LabelResult) -> EnsembleDecision:
    """AND-gate two negative-control voter verdicts into an ensemble decision.

    Precedence: contradiction (a curated host cleared the present-gates on a
    no-host channel — a false-accept suspect) > both confidently absent (silver
    negative) > abstain.
    """
    if voter_a.contradiction or voter_b.contradiction:
        return EnsembleDecision(
            verdict=FLAG_REVIEW,
            reason="candidate_host_present_on_neg_control",
            agreement="false_accept_suspect",
        )
    if (
        voter_a.verdict == NO_CURATED_HOST_DETECTED
        and voter_b.verdict == NO_CURATED_HOST_DETECTED
    ):
        return EnsembleDecision(
            verdict=SILVER_NO_HOST, reason="both_no_host", agreement="agree_absent"
        )
    return EnsembleDecision(
        verdict=ENSEMBLE_ABSTAIN,
        reason="no_confident_absence",
        agreement="one_or_both_abstain",
    )
