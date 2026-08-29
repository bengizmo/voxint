"""Calibration tooling for the three-band confidence policy (#114 Phase 2).

Pure, DB-free: operates on exported trial data. Reuses gate predicates from
:mod:`~voxint.speakers.tiers` so calibration boundaries match production.

Trial taxonomy:

- Genuine: machine top speaker == human-assigned speaker (post-canonicalization).
- Impostor: machine top speaker != human-assigned speaker (both are real people).
- Unscoreable: no human ruling, EXCLUDE, UNKNOWN, AUTO_ENROLL, or ineligible
  (no machine evidence to threshold on).

Truth anchoring tags whether the human decision was made before or after seeing
the machine proposal. Production confirms are anchoring-biased and cannot serve
as primary calibration truth (4-model consult, issue #114).
"""

import enum
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from voxint.harness.name_accuracy import wilson_ci
from voxint.speakers.matching import MatchingGates
from voxint.speakers.tiers import passes_accept, passes_grounded

MIN_INDEPENDENT_CLUSTERS = 50


class TrialKind(enum.StrEnum):
    GENUINE = "genuine"
    IMPOSTOR = "impostor"
    UNSCOREABLE = "unscoreable"


@dataclass(frozen=True)
class Trial:
    """One (run, label) calibration trial with machine evidence and human truth."""

    run_id: str
    label: str
    similarity: float | None
    margin: float | None
    vote_agreement: float | None
    eligible_turns: int
    eligible_seconds: float
    roster_size: int | None
    top_speaker_id: str | None
    kind: TrialKind
    truth_anchoring: str
    cluster_id: str


@dataclass(frozen=True)
class SweepPoint:
    """Metrics at one (cosine, margin) grid point.

    ``far`` is the traditional false accept rate: impostor trials auto-attributed
    divided by total impostor trials. ``far_ci_upper`` is the Wilson score 95% CI
    upper bound, DESCRIPTIVE-ONLY (assumes independent labels; with clustered data
    the true interval may be wider -- see :func:`check_independence`).
    """

    cosine: float
    margin: float
    auto_correct: int
    auto_wrong_person: int
    review_count: int
    abstain_count: int
    n_scoreable: int
    far: float
    far_ci_upper: float


@dataclass(frozen=True)
class BandChange:
    """A label whose band moved between baseline and candidate gates."""

    run_id: str
    label: str
    old_band: str
    new_band: str
    top_speaker_id: str | None
    similarity: float | None
    kind: TrialKind


@dataclass(frozen=True)
class CompareResult:
    """Aggregate diff between baseline and candidate gate configurations."""

    baseline_auto_correct: int
    baseline_auto_wrong: int
    baseline_review: int
    baseline_abstain: int
    candidate_auto_correct: int
    candidate_auto_wrong: int
    candidate_review: int
    candidate_abstain: int
    changes: tuple[BandChange, ...]
    n_scoreable: int


@dataclass(frozen=True)
class IndependenceReport:
    """Cluster independence check for a trial set."""

    n_clusters: int
    n_trials: int
    sufficient: bool
    cluster_sizes: dict[str, int]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classify_trial(
    *,
    run_id: str,
    label: str,
    mc_decision: str | None,
    similarity: float | None,
    margin: float | None,
    vote_agreement: float | None,
    eligible_turns: int,
    eligible_seconds: float,
    roster_size: int | None,
    top_speaker_id: str | None,
    human_decision: str | None,
    human_speaker_id: str | None,
    truth_anchoring: str,
) -> Trial:
    """Build a Trial from raw DB-extracted evidence, classifying genuine/impostor.

    ``top_speaker_id`` and ``human_speaker_id`` must already be canonicalized
    through merge tombstones so that a merged speaker compares equal to its
    canonical successor.
    """
    _has_truth = human_decision == "assign" and human_speaker_id is not None
    _has_evidence = mc_decision in ("accepted", "rejected") and top_speaker_id is not None

    if _has_truth and _has_evidence:
        kind = (
            TrialKind.GENUINE
            if human_speaker_id == top_speaker_id
            else TrialKind.IMPOSTOR
        )
    else:
        kind = TrialKind.UNSCOREABLE

    if kind != TrialKind.UNSCOREABLE and human_speaker_id is not None:
        cluster_id = human_speaker_id
    else:
        cluster_id = f"__unscoreable__:{run_id}:{label}"

    return Trial(
        run_id=run_id,
        label=label,
        similarity=similarity,
        margin=margin,
        vote_agreement=vote_agreement,
        eligible_turns=eligible_turns,
        eligible_seconds=eligible_seconds,
        roster_size=roster_size,
        top_speaker_id=top_speaker_id,
        kind=kind,
        truth_anchoring=truth_anchoring,
        cluster_id=cluster_id,
    )


# ---------------------------------------------------------------------------
# Band computation (reuses tiers.py shared predicates)
# ---------------------------------------------------------------------------
def _band_label(trial: Trial, gates: MatchingGates) -> str:
    """Classify a trial's evidence into a band string."""
    fields = (
        trial.similarity,
        trial.margin,
        trial.vote_agreement,
        trial.eligible_turns,
        trial.eligible_seconds,
        trial.roster_size,
    )
    if passes_grounded(*fields, gates):
        return "auto_attribute"
    if passes_accept(*fields, gates):
        return "review"
    return "abstain"


def _grounded_gates(
    base: MatchingGates, *, cosine: float, margin: float
) -> MatchingGates:
    """Clone ``base`` with overridden grounded cosine and margin thresholds."""
    return MatchingGates(
        max_overlap_ratio=base.max_overlap_ratio,
        turn_weight_cap_seconds=base.turn_weight_cap_seconds,
        min_turns=base.min_turns,
        min_seconds=base.min_seconds,
        min_cosine=base.min_cosine,
        min_margin=base.min_margin,
        min_vote_agreement=base.min_vote_agreement,
        grounded_min_turns=base.grounded_min_turns,
        grounded_min_seconds=base.grounded_min_seconds,
        grounded_min_cosine=cosine,
        grounded_min_margin=margin,
        grounded_min_vote_agreement=base.grounded_min_vote_agreement,
    )


def _scoreable(trials: Sequence[Trial]) -> list[Trial]:
    """Filter to trials with machine evidence and human truth."""
    return [t for t in trials if t.kind != TrialKind.UNSCOREABLE]


def _tally(
    trials: Sequence[Trial], gates: MatchingGates
) -> tuple[int, int, int, int]:
    """Count (auto_correct, auto_wrong, review, abstain) for scoreable trials."""
    auto_correct = 0
    auto_wrong = 0
    review = 0
    abstain = 0
    for trial in trials:
        band = _band_label(trial, gates)
        if band == "auto_attribute":
            if trial.kind == TrialKind.GENUINE:
                auto_correct += 1
            else:
                auto_wrong += 1
        elif band == "review":
            review += 1
        else:
            abstain += 1
    return auto_correct, auto_wrong, review, abstain


# ---------------------------------------------------------------------------
# Gate sweep
# ---------------------------------------------------------------------------
def sweep(
    trials: Sequence[Trial],
    *,
    cosine_grid: Sequence[float],
    margin_grid: Sequence[float],
    base_gates: MatchingGates,
    roster_stratum: str | None = None,
) -> list[SweepPoint]:
    """2D gate sweep over grounded cosine x grounded margin.

    The accept gate and eligibility floors are held fixed from ``base_gates``;
    only the grounded (AUTO_ATTRIBUTE) boundary moves. ``roster_stratum``
    filters trials: ``"R=1"`` (single-speaker roster), ``"R>=2"``
    (multi-speaker), or ``None`` (all).

    Callers are responsible for filtering trials by ``truth_anchoring`` if
    anchoring-biased (post-proposal) data must not drive the sweep. Mixed
    anchoring is permitted for exploratory analysis but not for threshold
    certification.
    """
    scoreable = _scoreable(trials)
    if roster_stratum == "R=1":
        scoreable = [t for t in scoreable if t.roster_size == 1]
    elif roster_stratum == "R>=2":
        scoreable = [
            t for t in scoreable if t.roster_size is not None and t.roster_size >= 2
        ]

    n_impostor = sum(1 for t in scoreable if t.kind == TrialKind.IMPOSTOR)

    points: list[SweepPoint] = []
    for cosine in cosine_grid:
        for margin_val in margin_grid:
            gates = _grounded_gates(base_gates, cosine=cosine, margin=margin_val)
            ac, aw, rv, ab = _tally(scoreable, gates)
            if n_impostor > 0:
                far = aw / n_impostor
                _, far_upper = wilson_ci(aw, n_impostor)
            else:
                far = 0.0
                far_upper = 1.0
            points.append(
                SweepPoint(
                    cosine=cosine,
                    margin=margin_val,
                    auto_correct=ac,
                    auto_wrong_person=aw,
                    review_count=rv,
                    abstain_count=ab,
                    n_scoreable=ac + aw + rv + ab,
                    far=far,
                    far_ci_upper=far_upper,
                )
            )
    return points


# ---------------------------------------------------------------------------
# PRE/POST comparator
# ---------------------------------------------------------------------------
def compare(
    trials: Sequence[Trial],
    *,
    baseline_gates: MatchingGates,
    candidate_gates: MatchingGates,
) -> CompareResult:
    """Band-change diff between baseline and candidate gate configurations.

    Reports which labels changed band, aggregate tallies for both configs,
    and the scoreable trial count.
    """
    scoreable = _scoreable(trials)
    b_ac, b_aw, b_rv, b_ab = _tally(scoreable, baseline_gates)
    c_ac, c_aw, c_rv, c_ab = _tally(scoreable, candidate_gates)

    changes: list[BandChange] = []
    for trial in scoreable:
        old = _band_label(trial, baseline_gates)
        new = _band_label(trial, candidate_gates)
        if old != new:
            changes.append(
                BandChange(
                    run_id=trial.run_id,
                    label=trial.label,
                    old_band=old,
                    new_band=new,
                    top_speaker_id=trial.top_speaker_id,
                    similarity=trial.similarity,
                    kind=trial.kind,
                )
            )

    return CompareResult(
        baseline_auto_correct=b_ac,
        baseline_auto_wrong=b_aw,
        baseline_review=b_rv,
        baseline_abstain=b_ab,
        candidate_auto_correct=c_ac,
        candidate_auto_wrong=c_aw,
        candidate_review=c_rv,
        candidate_abstain=c_ab,
        changes=tuple(changes),
        n_scoreable=len(scoreable),
    )


# ---------------------------------------------------------------------------
# Independence check
# ---------------------------------------------------------------------------
def check_independence(trials: Sequence[Trial]) -> IndependenceReport:
    """Verify the trial set has enough independent clusters (>= 50).

    Independence = speaker clusters. Trials sharing a human-assigned speaker
    are correlated (same voice); resampling must be at the cluster level.
    Fewer than 50 clusters means the calibration cannot produce a reliable
    decision (NO_DECISION is a valid first-class outcome).
    """
    scoreable = _scoreable(trials)
    counts = Counter(t.cluster_id for t in scoreable)
    return IndependenceReport(
        n_clusters=len(counts),
        n_trials=len(scoreable),
        sufficient=len(counts) >= MIN_INDEPENDENT_CLUSTERS,
        cluster_sizes=dict(counts),
    )


# ---------------------------------------------------------------------------
# Serialization (JSONL round-trip)
# ---------------------------------------------------------------------------
def trial_to_dict(trial: Trial) -> dict[str, Any]:
    """Serialize a Trial for JSONL output."""
    return {
        "run_id": trial.run_id,
        "label": trial.label,
        "similarity": trial.similarity,
        "margin": trial.margin,
        "vote_agreement": trial.vote_agreement,
        "eligible_turns": trial.eligible_turns,
        "eligible_seconds": trial.eligible_seconds,
        "roster_size": trial.roster_size,
        "top_speaker_id": trial.top_speaker_id,
        "kind": trial.kind.value,
        "truth_anchoring": trial.truth_anchoring,
        "cluster_id": trial.cluster_id,
    }


def trial_from_dict(d: dict[str, Any]) -> Trial:
    """Deserialize a Trial from a JSONL record."""
    return Trial(
        run_id=d["run_id"],
        label=d["label"],
        similarity=d.get("similarity"),
        margin=d.get("margin"),
        vote_agreement=d.get("vote_agreement"),
        eligible_turns=d.get("eligible_turns", 0),
        eligible_seconds=d.get("eligible_seconds", 0.0),
        roster_size=d.get("roster_size"),
        top_speaker_id=d.get("top_speaker_id"),
        kind=TrialKind(d["kind"]),
        truth_anchoring=d["truth_anchoring"],
        cluster_id=d["cluster_id"],
    )


def sweep_point_to_dict(point: SweepPoint) -> dict[str, Any]:
    """Serialize a SweepPoint for JSON output."""
    return {
        "cosine": point.cosine,
        "margin": point.margin,
        "auto_correct": point.auto_correct,
        "auto_wrong_person": point.auto_wrong_person,
        "review_count": point.review_count,
        "abstain_count": point.abstain_count,
        "n_scoreable": point.n_scoreable,
        "far": round(point.far, 6),
        "far_ci_upper": round(point.far_ci_upper, 6),
    }


def gates_to_dict(gates: MatchingGates) -> dict[str, Any]:
    """Serialize MatchingGates to a JSON-compatible dict."""
    return {
        "max_overlap_ratio": gates.max_overlap_ratio,
        "turn_weight_cap_seconds": gates.turn_weight_cap_seconds,
        "min_turns": gates.min_turns,
        "min_seconds": gates.min_seconds,
        "min_cosine": gates.min_cosine,
        "min_margin": gates.min_margin,
        "min_vote_agreement": gates.min_vote_agreement,
        "grounded_min_turns": gates.grounded_min_turns,
        "grounded_min_seconds": gates.grounded_min_seconds,
        "grounded_min_cosine": gates.grounded_min_cosine,
        "grounded_min_margin": gates.grounded_min_margin,
        "grounded_min_vote_agreement": gates.grounded_min_vote_agreement,
    }


def gates_from_dict(d: dict[str, Any]) -> MatchingGates:
    """Deserialize MatchingGates from a JSON dict."""
    return MatchingGates(
        max_overlap_ratio=d["max_overlap_ratio"],
        turn_weight_cap_seconds=d["turn_weight_cap_seconds"],
        min_turns=d["min_turns"],
        min_seconds=d["min_seconds"],
        min_cosine=d["min_cosine"],
        min_margin=d["min_margin"],
        min_vote_agreement=d["min_vote_agreement"],
        grounded_min_turns=d["grounded_min_turns"],
        grounded_min_seconds=d["grounded_min_seconds"],
        grounded_min_cosine=d["grounded_min_cosine"],
        grounded_min_margin=d["grounded_min_margin"],
        grounded_min_vote_agreement=d["grounded_min_vote_agreement"],
    )
