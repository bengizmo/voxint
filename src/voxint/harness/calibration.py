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

Pre-registered selection and certification rule:

- Candidate family: grounded cosine in
  {0.66,0.68,0.70,0.72,0.74,0.76,0.78,0.80} x grounded margin in
  {0.05,0.08,0.10,0.12}; all other gates held at the base gates (defaults or
  --gates file). Accept tier and eligibility floors fixed.
- Feasible on DEV: auto_wrong == 0 AND one-sided 95% Wilson upper bound of
  cluster-level FAR (impostor clusters with >= 1 auto_wrong over impostor
  clusters) <= 0.05.
- Selection among feasible: maximise genuine cluster coverage (fraction of
  genuine clusters with >= 1 auto_correct), then minimise review count, then
  stricter (higher cosine, then higher margin). If no feasible point: outcome
  NO_DECISION reason "no_feasible_candidate".
- Certification on CONFIRM with the locked candidate (single scoring, no
  reselection): CERTIFIED iff impostor clusters >= MIN_INDEPENDENT_CLUSTERS
  (50) AND auto_wrong == 0 AND one-sided upper FAR <= 0.05; else NO_DECISION
  with the failing reasons listed (insufficient_impostor_clusters /
  auto_wrong_observed / far_bound_exceeded). Also report the PRE (base gates)
  vs POST (candidate) tallies on the same confirm trials, the full band-change
  listing (reuse `compare`), genuine-cluster coverage and trial-level FRR for
  both, descriptive two-sided Wilson CIs, and the confirm genuine cluster count
  (descriptive; no floor, per the asymmetric bar).
"""

import enum
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from voxint.harness.name_accuracy import wilson_ci, wilson_upper_one_sided
from voxint.speakers.matching import MatchingGates
from voxint.speakers.tiers import passes_accept, passes_grounded

MIN_INDEPENDENT_CLUSTERS = 50


class TrialKind(enum.StrEnum):
    GENUINE = "genuine"
    IMPOSTOR = "impostor"
    UNSCOREABLE = "unscoreable"


class TrialDetail(enum.StrEnum):
    GENUINE = "genuine"
    IMPOSTOR_CLOSED = "impostor_closed"
    IMPOSTOR_OPEN = "impostor_open"
    UNSCOREABLE = "unscoreable"


_DETAIL_BY_KIND = {
    TrialKind.GENUINE: TrialDetail.GENUINE,
    TrialKind.IMPOSTOR: TrialDetail.IMPOSTOR_CLOSED,
    TrialKind.UNSCOREABLE: TrialDetail.UNSCOREABLE,
}
_DETAIL_UNSET: Any = object()


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
    detail: TrialDetail = field(default=_DETAIL_UNSET)
    split: str | None = None

    def __post_init__(self) -> None:
        if self.detail is _DETAIL_UNSET:
            object.__setattr__(self, "detail", _DETAIL_BY_KIND[self.kind])


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
    n_impostor_clusters: int
    n_genuine_clusters: int
    far_upper_one_sided: float
    genuine_cluster_coverage: float


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
    """Per-kind cluster independence check for a trial set."""

    n_clusters: int
    n_trials: int
    sufficient: bool
    cluster_sizes: dict[str, int]
    n_genuine_clusters: int
    n_impostor_clusters: int
    genuine_cluster_sizes: dict[str, int]
    impostor_cluster_sizes: dict[str, int]


@dataclass(frozen=True)
class SelectionRule:
    """The fixed candidate family and safety thresholds for issue #114."""

    cosine_grid: tuple[float, ...] = (
        0.66,
        0.68,
        0.70,
        0.72,
        0.74,
        0.76,
        0.78,
        0.80,
    )
    margin_grid: tuple[float, ...] = (0.05, 0.08, 0.10, 0.12)
    max_far_upper_one_sided: float = 0.05
    confidence: float = 0.95
    min_impostor_clusters: int = MIN_INDEPENDENT_CLUSTERS


@dataclass(frozen=True)
class SelectionPoint:
    """One pre-registered DEV candidate plus its feasibility decision."""

    metrics: SweepPoint
    feasible: bool


@dataclass(frozen=True)
class SelectionResult:
    """DEV selection outcome and the complete candidate search record."""

    outcome: str
    reason: str | None
    chosen_gates: MatchingGates | None
    points: tuple[SelectionPoint, ...]
    n_genuine_clusters: int
    n_impostor_clusters: int


@dataclass(frozen=True)
class PolicyEvaluation:
    """Safety and utility metrics for one policy on one fixed trial set."""

    auto_correct: int
    auto_wrong: int
    review: int
    abstain: int
    n_scoreable: int
    n_genuine_trials: int
    n_impostor_trials: int
    n_genuine_clusters: int
    n_impostor_clusters: int
    far: float
    far_ci: tuple[float, float]
    far_upper_one_sided: float
    frr: float
    frr_ci: tuple[float, float]
    genuine_cluster_coverage: float
    coverage: float


@dataclass(frozen=True)
class CertificationResult:
    """Locked-candidate CONFIRM result with paired PRE/POST evidence."""

    outcome: str
    reasons: tuple[str, ...]
    pre: PolicyEvaluation
    post: PolicyEvaluation
    changes: tuple[BandChange, ...]
    candidate_gates: MatchingGates


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
        detail=_DETAIL_BY_KIND[kind],
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

    impostors = [t for t in scoreable if t.kind == TrialKind.IMPOSTOR]
    genuines = [t for t in scoreable if t.kind == TrialKind.GENUINE]
    n_impostor = len(impostors)
    impostor_clusters = {t.cluster_id for t in impostors}
    genuine_clusters = {t.cluster_id for t in genuines}

    points: list[SweepPoint] = []
    for cosine in cosine_grid:
        for margin_val in margin_grid:
            gates = _grounded_gates(base_gates, cosine=cosine, margin=margin_val)
            ac, aw, rv, ab = _tally(scoreable, gates)
            wrong_clusters = {
                t.cluster_id
                for t in impostors
                if _band_label(t, gates) == "auto_attribute"
            }
            covered_genuine_clusters = {
                t.cluster_id
                for t in genuines
                if _band_label(t, gates) == "auto_attribute"
            }
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
                    n_impostor_clusters=len(impostor_clusters),
                    n_genuine_clusters=len(genuine_clusters),
                    far_upper_one_sided=wilson_upper_one_sided(
                        len(wrong_clusters), len(impostor_clusters)
                    ),
                    genuine_cluster_coverage=(
                        len(covered_genuine_clusters) / len(genuine_clusters)
                        if genuine_clusters
                        else 0.0
                    ),
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
    """Verify the trial set has enough independent impostor clusters (>= 50).

    Independence = speaker clusters. Trials sharing a human-assigned speaker
    are correlated (same voice); resampling must be at the cluster level.
    Fewer than 50 impostor clusters means the calibration cannot produce a
    reliable FAR decision (NO_DECISION is a valid first-class outcome).
    """
    scoreable = _scoreable(trials)
    counts = Counter(t.cluster_id for t in scoreable)
    genuine_counts = Counter(
        t.cluster_id for t in scoreable if t.kind == TrialKind.GENUINE
    )
    impostor_counts = Counter(
        t.cluster_id for t in scoreable if t.kind == TrialKind.IMPOSTOR
    )
    return IndependenceReport(
        n_clusters=len(counts),
        n_trials=len(scoreable),
        sufficient=len(impostor_counts) >= MIN_INDEPENDENT_CLUSTERS,
        cluster_sizes=dict(counts),
        n_genuine_clusters=len(genuine_counts),
        n_impostor_clusters=len(impostor_counts),
        genuine_cluster_sizes=dict(genuine_counts),
        impostor_cluster_sizes=dict(impostor_counts),
    )


# ---------------------------------------------------------------------------
# Pre-registered DEV selection and locked CONFIRM certification
# ---------------------------------------------------------------------------
def load_attribution_trials(data: dict[str, Any]) -> list[Trial]:
    """Load split-aware trials from an ``eval_attribution align`` document."""
    if data.get("kind") != "attribution_trials":
        raise ValueError(
            f"expected kind 'attribution_trials', got {data.get('kind')!r}"
        )
    raw_trials = data.get("trials")
    if not isinstance(raw_trials, list):
        raise ValueError("attribution trials document requires a 'trials' list")
    trials: list[Trial] = []
    for index, raw in enumerate(raw_trials):
        if not isinstance(raw, dict):
            raise ValueError(f"trial {index}: expected an object")
        split = raw.get("meeting_split")
        if split is None:
            raise ValueError(f"trial {index}: missing required meeting_split")
        if split not in ("dev", "confirm"):
            raise ValueError(f"trial {index}: invalid meeting_split {split!r}")
        payload = dict(raw)
        payload["split"] = split
        try:
            trials.append(trial_from_dict(payload))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"trial {index}: {exc}") from exc
    return trials


def filter_split(trials: Sequence[Trial], split: str) -> list[Trial]:
    """Return trials belonging to one pre-registered meeting split."""
    if split not in ("dev", "confirm"):
        raise ValueError(f"split must be 'dev' or 'confirm', got {split!r}")
    return [trial for trial in trials if trial.split == split]


def rule_to_dict(rule: SelectionRule) -> dict[str, Any]:
    """Serialize the complete pre-registered rule."""
    return {
        "cosine_grid": list(rule.cosine_grid),
        "margin_grid": list(rule.margin_grid),
        "max_far_upper_one_sided": rule.max_far_upper_one_sided,
        "confidence": rule.confidence,
        "min_impostor_clusters": rule.min_impostor_clusters,
    }


def select_candidate(
    dev_trials: Sequence[Trial],
    base_gates: MatchingGates,
    rule: SelectionRule,
) -> SelectionResult:
    """Apply the pre-registered candidate ordering once on DEV trials."""
    points_list: list[SelectionPoint] = []
    for point in sweep(
        dev_trials,
        cosine_grid=rule.cosine_grid,
        margin_grid=rule.margin_grid,
        base_gates=base_gates,
    ):
        gates = _grounded_gates(
            base_gates, cosine=point.cosine, margin=point.margin
        )
        evaluation = _evaluate_policy(dev_trials, gates, rule.confidence)
        metrics = replace(
            point, far_upper_one_sided=evaluation.far_upper_one_sided
        )
        points_list.append(
            SelectionPoint(
                metrics=metrics,
                feasible=(
                    metrics.auto_wrong_person == 0
                    and metrics.far_upper_one_sided
                    <= rule.max_far_upper_one_sided
                ),
            )
        )
    points = tuple(points_list)
    feasible = [point for point in points if point.feasible]
    independence = check_independence(dev_trials)
    if not feasible:
        return SelectionResult(
            outcome="NO_DECISION",
            reason="no_feasible_candidate",
            chosen_gates=None,
            points=points,
            n_genuine_clusters=independence.n_genuine_clusters,
            n_impostor_clusters=independence.n_impostor_clusters,
        )
    chosen = max(
        feasible,
        key=lambda point: (
            point.metrics.genuine_cluster_coverage,
            -point.metrics.review_count,
            point.metrics.cosine,
            point.metrics.margin,
        ),
    )
    return SelectionResult(
        outcome="SELECTED",
        reason=None,
        chosen_gates=_grounded_gates(
            base_gates,
            cosine=chosen.metrics.cosine,
            margin=chosen.metrics.margin,
        ),
        points=points,
        n_genuine_clusters=independence.n_genuine_clusters,
        n_impostor_clusters=independence.n_impostor_clusters,
    )


def _evaluate_policy(
    trials: Sequence[Trial], gates: MatchingGates, confidence: float
) -> PolicyEvaluation:
    scoreable = _scoreable(trials)
    genuines = [trial for trial in scoreable if trial.kind == TrialKind.GENUINE]
    impostors = [trial for trial in scoreable if trial.kind == TrialKind.IMPOSTOR]
    auto_correct, auto_wrong, review, abstain = _tally(scoreable, gates)
    genuine_clusters = {trial.cluster_id for trial in genuines}
    impostor_clusters = {trial.cluster_id for trial in impostors}
    covered_genuine_clusters = {
        trial.cluster_id
        for trial in genuines
        if _band_label(trial, gates) == "auto_attribute"
    }
    wrong_impostor_clusters = {
        trial.cluster_id
        for trial in impostors
        if _band_label(trial, gates) == "auto_attribute"
    }
    false_rejects = len(genuines) - auto_correct
    auto_total = auto_correct + auto_wrong
    return PolicyEvaluation(
        auto_correct=auto_correct,
        auto_wrong=auto_wrong,
        review=review,
        abstain=abstain,
        n_scoreable=len(scoreable),
        n_genuine_trials=len(genuines),
        n_impostor_trials=len(impostors),
        n_genuine_clusters=len(genuine_clusters),
        n_impostor_clusters=len(impostor_clusters),
        far=auto_wrong / len(impostors) if impostors else 0.0,
        far_ci=wilson_ci(auto_wrong, len(impostors)),
        far_upper_one_sided=wilson_upper_one_sided(
            len(wrong_impostor_clusters),
            len(impostor_clusters),
            confidence=confidence,
        ),
        frr=false_rejects / len(genuines) if genuines else 0.0,
        frr_ci=wilson_ci(false_rejects, len(genuines)),
        genuine_cluster_coverage=(
            len(covered_genuine_clusters) / len(genuine_clusters)
            if genuine_clusters
            else 0.0
        ),
        coverage=auto_total / len(scoreable) if scoreable else 0.0,
    )


def certify(
    confirm_trials: Sequence[Trial],
    base_gates: MatchingGates,
    candidate_gates: MatchingGates,
    rule: SelectionRule,
) -> CertificationResult:
    """Score one locked candidate on CONFIRM without reselection."""
    pre = _evaluate_policy(confirm_trials, base_gates, rule.confidence)
    post = _evaluate_policy(confirm_trials, candidate_gates, rule.confidence)
    reasons: list[str] = []
    if post.n_impostor_clusters < rule.min_impostor_clusters:
        reasons.append("insufficient_impostor_clusters")
    if post.auto_wrong != 0:
        reasons.append("auto_wrong_observed")
    if post.far_upper_one_sided > rule.max_far_upper_one_sided:
        reasons.append("far_bound_exceeded")
    paired = compare(
        confirm_trials,
        baseline_gates=base_gates,
        candidate_gates=candidate_gates,
    )
    return CertificationResult(
        outcome="CERTIFIED" if not reasons else "NO_DECISION",
        reasons=tuple(reasons),
        pre=pre,
        post=post,
        changes=paired.changes,
        candidate_gates=candidate_gates,
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
        "detail": trial.detail.value,
        "truth_anchoring": trial.truth_anchoring,
        "cluster_id": trial.cluster_id,
        "split": trial.split,
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
        detail=(
            TrialDetail(d["detail"])
            if "detail" in d
            else _DETAIL_BY_KIND[TrialKind(d["kind"])]
        ),
        split=d.get("split"),
    )


def selection_result_to_dict(result: SelectionResult, rule: SelectionRule) -> dict[str, Any]:
    """Serialize a DEV selection result without non-finite JSON values."""
    return {
        "kind": "calibration_selection",
        "outcome": result.outcome,
        "reason": result.reason,
        "rule": rule_to_dict(rule),
        "chosen_gates": (
            gates_to_dict(result.chosen_gates)
            if result.chosen_gates is not None
            else None
        ),
        "n_genuine_clusters": result.n_genuine_clusters,
        "n_impostor_clusters": result.n_impostor_clusters,
        "points": [
            {
                **sweep_point_to_dict(point.metrics),
                "auto_wrong": point.metrics.auto_wrong_person,
                "feasible": point.feasible,
            }
            for point in result.points
        ],
    }


def policy_evaluation_to_dict(evaluation: PolicyEvaluation) -> dict[str, Any]:
    """Serialize one PRE or POST policy evaluation."""
    return {
        "auto_correct": evaluation.auto_correct,
        "auto_wrong": evaluation.auto_wrong,
        "review": evaluation.review,
        "abstain": evaluation.abstain,
        "n_scoreable": evaluation.n_scoreable,
        "n_genuine_trials": evaluation.n_genuine_trials,
        "n_impostor_trials": evaluation.n_impostor_trials,
        "n_genuine_clusters": evaluation.n_genuine_clusters,
        "n_impostor_clusters": evaluation.n_impostor_clusters,
        "far": evaluation.far,
        "far_ci": list(evaluation.far_ci),
        "far_upper_one_sided": evaluation.far_upper_one_sided,
        "frr": evaluation.frr,
        "frr_ci": list(evaluation.frr_ci),
        "genuine_cluster_coverage": evaluation.genuine_cluster_coverage,
        "coverage": evaluation.coverage,
    }


def certification_result_to_dict(
    result: CertificationResult, rule: SelectionRule
) -> dict[str, Any]:
    """Serialize the locked CONFIRM decision and complete paired evidence."""
    return {
        "kind": "calibration_certification",
        "outcome": result.outcome,
        "reasons": list(result.reasons),
        "rule": rule_to_dict(rule),
        "candidate_gates": gates_to_dict(result.candidate_gates),
        "pre": policy_evaluation_to_dict(result.pre),
        "post": policy_evaluation_to_dict(result.post),
        "changes": [
            {
                "run_id": change.run_id,
                "label": change.label,
                "old_band": change.old_band,
                "new_band": change.new_band,
                "top_speaker_id": change.top_speaker_id,
                "similarity": change.similarity,
                "kind": change.kind.value,
            }
            for change in result.changes
        ],
    }


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
        "n_impostor_clusters": point.n_impostor_clusters,
        "n_genuine_clusters": point.n_genuine_clusters,
        "far_upper_one_sided": round(point.far_upper_one_sided, 6),
        "genuine_cluster_coverage": round(point.genuine_cluster_coverage, 6),
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
