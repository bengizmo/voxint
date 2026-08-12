"""Speaker name-accuracy scoring core (pure, DB-free).

Structural diarization metrics (DER/JER, permutation-optimal WER variants) are
blind to *name* correctness: they optimally relabel speakers before scoring.
This module scores the thing the user actually sees — whether the display name
on a diarized slot is the right person.

``person_name_match`` is deliberately stricter than symmetric substring
matching: a bare first name, bare surname, or short token is NOT proof of
identity. Matching prefers identity-id equality, then an explicit alias table,
then exact full-name equality, then a surname-required token match; substring
containment is downgraded to a ``weak`` signal that never counts as a match.

Names are compared under Unicode NFKC + casefold, so accented, hyphenated, and
non-ASCII names compare correctly. Identity ids are opaque non-empty strings.

``scipy`` is imported lazily inside :func:`mcnemar` (its only user) so that
importing this module — and the CLI paths that use it — costs numpy only.
"""

import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

# Display-name prefixes that mark a placeholder, not a real person name.
PLACEHOLDER_PREFIXES: tuple[str, ...] = ("speaker_", "auto_", "unknown")

# person_name_match strength codes, strongest first.
STRENGTH_ID = "id"
STRENGTH_ALIAS = "alias"
STRENGTH_EXACT = "exact"
STRENGTH_SURNAME_GIVEN = "surname_given"
STRENGTH_WEAK = "weak"
STRENGTH_NONE = "none"

# Ground-truth sentinels (a truth is either a real name string or one of these).
ABSTAIN = "__ABSTAIN__"  # no name should be assigned to this slot
NEITHER_DETERMINABLE = "__NEITHER_DETERMINABLE__"  # unscoreable -> excluded

# Slot verdicts.
TP = "TP"  # assigned the correct person
FP_WRONG = "FP_WRONG"  # assigned a name, truth is a different real person
FP_OVERNAME = "FP_OVERNAME"  # assigned a name, truth is ABSTAIN (over-naming)
FN = "FN"  # assigned nothing, truth is a real person (missed)
TN = "TN"  # assigned nothing, truth is ABSTAIN (correct abstain)
EXCLUDED = "EXCLUDED"  # truth is NEITHER_DETERMINABLE / unscoreable

VERDICTS = frozenset({TP, FP_WRONG, FP_OVERNAME, FN, TN, EXCLUDED})


@dataclass(frozen=True)
class NameMatch:
    """Outcome of comparing two display names for the *same person*."""

    matched: bool
    strength: str


@dataclass(frozen=True)
class McNemarResult:
    """Paired discordant-pair comparison of baseline vs candidate correctness."""

    baseline_correct_candidate_wrong: int  # b — candidate regressed this slot
    baseline_wrong_candidate_correct: int  # c — candidate fixed this slot
    n_discordant: int  # b + c
    net: int  # c - b (positive = candidate net-improves)
    p_value: float  # exact two-sided McNemar (binomial)


@dataclass(frozen=True)
class BootstrapResult:
    """Item-clustered bootstrap CI on the mean per-slot paired delta."""

    point: float
    lo: float
    hi: float


@dataclass(frozen=True)
class Aggregate:
    """Slot-verdict tallies + derived precision/recall/F1 and confusion matrix."""

    tp: int
    fp_wrong: int
    fp_overname: int
    fn: int
    tn: int
    excluded: int
    precision: float
    recall: float
    f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    confusion: dict[str, dict[str, int]]


@dataclass(frozen=True)
class RiskCoverage:
    """Descriptive selective-prediction curve. NEVER a gate input (uncalibrated)."""

    points: list[tuple[float, float, float | None]]  # (coverage, accuracy, threshold)
    chow_coverage: float | None
    descriptive: bool = True
    calibrated: bool = False


# --------------------------------------------------------------------------- #
# Name-compare primitives
# --------------------------------------------------------------------------- #
def is_named(dn: object) -> bool:
    """True iff ``dn`` is a real (non-placeholder, non-blank) display name."""
    if not dn or not isinstance(dn, str):
        return False
    norm = dn.strip().casefold()
    return bool(norm) and not norm.startswith(PLACEHOLDER_PREFIXES)


def _norm_name(name: str | None) -> str:
    """NFKC-normalize, casefold, collapse internal whitespace, strip."""
    if not name or not isinstance(name, str):
        return ""
    folded = unicodedata.normalize("NFKC", name).casefold()
    return re.sub(r"\s+", " ", folded.strip())


def _name_tokens(name: str | None) -> list[str]:
    """Word tokens of a normalized name (Unicode-aware, punctuation dropped)."""
    return [t for t in re.split(r"[^\w]+", _norm_name(name)) if t]


def _alias_match(a_norm: str, b_norm: str, aliases: Mapping[str, Iterable[str]]) -> bool:
    """True iff both names appear under one canonical entry of the alias table
    (symmetric: either side may be the canonical key)."""
    for canonical, alts in aliases.items():
        alt_norms = {_norm_name(x) for x in alts} | {_norm_name(canonical)}
        if a_norm in alt_norms and b_norm in alt_norms:
            return True
    return False


def _given_tokens_match(ga: str, gb: str) -> bool:
    """Given-name tokens match if equal or one is the other's initial."""
    if ga == gb:
        return True
    if len(ga) == 1 and gb.startswith(ga):
        return True
    return len(gb) == 1 and ga.startswith(gb)


def person_name_match(
    a: str | None,
    b: str | None,
    *,
    aliases: Mapping[str, Iterable[str]] | None = None,
    id_a: str | None = None,
    id_b: str | None = None,
) -> NameMatch:
    """Compare two names for identity. Stricter than substring (see module doc).

    Resolution order (strongest first):
      1. ``id`` — both identity ids present (non-empty strings) and equal.
      2. ``alias`` — both names appear under one entry of the alias table.
      3. ``exact`` — normalized full strings equal.
      4. ``surname_given`` — both multi-token; surnames equal AND given names
         match (equal or initial-of). Tolerates middle names.
      5. ``weak`` — single-token containment (bare first name / surname / short
         token): flagged but does NOT count as a match.
      6. ``none`` — no relationship.
    """
    if id_a and id_b and id_a == id_b:
        return NameMatch(True, STRENGTH_ID)

    a_norm, b_norm = _norm_name(a), _norm_name(b)
    if not a_norm or not b_norm:
        return NameMatch(False, STRENGTH_NONE)

    if aliases and _alias_match(a_norm, b_norm, aliases):
        return NameMatch(True, STRENGTH_ALIAS)

    if a_norm == b_norm:
        return NameMatch(True, STRENGTH_EXACT)

    ta, tb = _name_tokens(a), _name_tokens(b)
    if len(ta) >= 2 and len(tb) >= 2:
        if ta[-1] == tb[-1] and _given_tokens_match(ta[0], tb[0]):
            return NameMatch(True, STRENGTH_SURNAME_GIVEN)
        return NameMatch(False, STRENGTH_NONE)

    # At least one side is a single token: a bare first name / surname / short
    # token. Containment is the legacy-substring signal, downgraded to weak.
    if set(ta) & set(tb) or a_norm in b_norm or b_norm in a_norm:
        return NameMatch(False, STRENGTH_WEAK)
    return NameMatch(False, STRENGTH_NONE)


def norm_channel(c: str | None) -> str:
    """Normalize a channel name: casefold, word characters only ('' if blank).

    Collapses punctuation/spacing variants of the same channel so
    'Acme Audio' matches 'Acme-Audio'.
    """
    if not c:
        return ""
    return re.sub(r"[^\w]", "", unicodedata.normalize("NFKC", str(c)).casefold())


def curated_hits(names: Sequence[str], curated_hosts: Iterable[str]) -> list[str]:
    """Curated hosts *strongly* present among ``names`` (preserves host order).

    Uses the strict :func:`person_name_match`, so a bare first name does not
    over-flag a curated host.
    """
    return [
        host
        for host in curated_hosts
        if any(person_name_match(host, n).matched for n in names)
    ]


def is_overnaming(
    host: str,
    channel: str | None,
    host_home_channels: Mapping[str, Iterable[str]],
) -> bool:
    """A curated host assigned off its home channel = cross-show over-naming.

    A null/blank channel is NOT flagged (unknown provenance is not proof of an
    off-channel assignment). Channel comparison is normalized so spelling
    variants of the same channel are equal. A host with no recorded home
    channel, placed on any real channel, is off-channel by definition.
    """
    norm = norm_channel(channel)
    if not norm:
        return False
    homes = {norm_channel(h) for h in host_home_channels.get(host, ())}
    return norm not in homes


# --------------------------------------------------------------------------- #
# Verdicts + aggregates
# --------------------------------------------------------------------------- #
def slot_verdict(
    assigned_name: str | None,
    truth: str | None,
    *,
    aliases: Mapping[str, Iterable[str]] | None = None,
) -> str:
    """Classify one slot's assigned name against its ground truth.

    ``truth`` is a real name string, :data:`ABSTAIN` (no name should be
    assigned), or :data:`NEITHER_DETERMINABLE`/None (unscoreable ->
    :data:`EXCLUDED`). A placeholder / blank ``assigned_name`` counts as an
    abstention.
    """
    if truth is None or truth == NEITHER_DETERMINABLE:
        return EXCLUDED

    assigned = is_named(assigned_name)

    if truth == ABSTAIN:
        return FP_OVERNAME if assigned else TN

    if not assigned:
        return FN
    return TP if person_name_match(assigned_name, truth, aliases=aliases).matched else FP_WRONG


def _prf1(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    """Precision, recall, F1 from weighted-or-count TP/FP/FN (0.0 on empty denom)."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def aggregate(verdicts: Sequence[str | tuple[str, float]]) -> Aggregate:
    """Tally slot verdicts into counts, P/R/F1, duration-weighted P/R/F1, confusion.

    Each element is a verdict string (weight 1.0) or a ``(verdict, weight)``
    tuple where the weight is the slot's duration for the weighted variant.
    Weights must be finite and non-negative. ``FP_OVERNAME`` counts against
    precision; ``EXCLUDED`` slots are tallied but ignored by the metrics.
    Empty/all-excluded input yields zeroed metrics rather than raising.
    """
    counts = {TP: 0, FP_WRONG: 0, FP_OVERNAME: 0, FN: 0, TN: 0, EXCLUDED: 0}
    w = {TP: 0.0, FP_WRONG: 0.0, FP_OVERNAME: 0.0, FN: 0.0, TN: 0.0}

    for item in verdicts:
        if isinstance(item, tuple):
            verdict, weight = item[0], float(item[1])
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"verdict weight must be finite and >= 0, got {weight}")
        else:
            verdict, weight = item, 1.0
        if verdict not in counts:
            raise ValueError(f"unknown verdict {verdict!r}")
        counts[verdict] += 1
        if verdict in w:
            w[verdict] += weight

    fp = counts[FP_WRONG] + counts[FP_OVERNAME]
    precision, recall, f1 = _prf1(counts[TP], fp, counts[FN])
    wp, wr, wf1 = _prf1(w[TP], w[FP_WRONG] + w[FP_OVERNAME], w[FN])

    confusion = {
        "true_name": {
            "named_correct": counts[TP],
            "named_wrong": counts[FP_WRONG],
            "abstained": counts[FN],
        },
        "true_abstain": {
            "named": counts[FP_OVERNAME],
            "abstained": counts[TN],
        },
    }

    return Aggregate(
        tp=counts[TP],
        fp_wrong=counts[FP_WRONG],
        fp_overname=counts[FP_OVERNAME],
        fn=counts[FN],
        tn=counts[TN],
        excluded=counts[EXCLUDED],
        precision=precision,
        recall=recall,
        f1=f1,
        weighted_precision=wp,
        weighted_recall=wr,
        weighted_f1=wf1,
        confusion=confusion,
    )


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a single binomial rate (default 95% two-sided).

    ``n == 0`` returns ``(0.0, 1.0)`` (no information). Bounds are clamped to
    ``[0, 1]``. Used only for single-rate reporting — paired comparisons use
    :func:`mcnemar` / :func:`clustered_bootstrap_delta`.
    """
    if n <= 0:
        return (0.0, 1.0)
    if not math.isfinite(z) or z <= 0:
        raise ValueError(f"z must be a positive finite number, got {z}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be within [0, n], got {successes}/{n}")
    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar(
    baseline_correct: Sequence[bool], candidate_correct: Sequence[bool]
) -> McNemarResult:
    """Exact (binomial) McNemar test on paired per-slot correctness.

    ``b`` = slots correct at baseline but wrong for the candidate (regressions);
    ``c`` = wrong at baseline but correct for the candidate (fixes). The
    two-sided exact test is the binomial of ``b`` among the ``b + c`` discordant
    pairs at p=0.5. No discordant pairs -> p_value 1.0.
    """
    if len(baseline_correct) != len(candidate_correct):
        raise ValueError("baseline and candidate sequences must be the same length")
    paired = list(zip(baseline_correct, candidate_correct, strict=True))
    b = sum(1 for pre, post in paired if pre and not post)
    c = sum(1 for pre, post in paired if post and not pre)
    n_disc = b + c
    from scipy import stats  # lazy: keeps scipy off the module import path

    p_value = 1.0 if n_disc == 0 else float(stats.binomtest(b, n_disc, 0.5).pvalue)
    return McNemarResult(
        baseline_correct_candidate_wrong=b,
        baseline_wrong_candidate_correct=c,
        n_discordant=n_disc,
        net=c - b,
        p_value=p_value,
    )


def clustered_bootstrap_delta(
    items: Sequence[Sequence[tuple[bool, bool]]],
    *,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Item-clustered bootstrap CI on the mean per-slot paired delta.

    Each item is a cluster of ``(baseline_correct, candidate_correct)`` slot
    pairs; resampling is over *items* (slots within an item stay together — the
    correct unit of independence). The statistic is the mean of
    ``int(candidate) - int(baseline)`` over all slots in the resample.
    Deterministic given ``seed``.
    """
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci}")
    if n_boot <= 0:
        raise ValueError(f"n_boot must be positive, got {n_boot}")
    per_item_deltas: list[list[int]] = [
        [int(post) - int(pre) for (pre, post) in item] for item in items
    ]
    flat = [d for item in per_item_deltas for d in item]
    if not flat:
        return BootstrapResult(point=0.0, lo=0.0, hi=0.0)

    point = float(np.mean(flat))
    rng = np.random.default_rng(seed)
    n_items = len(per_item_deltas)
    boot_means: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_items, size=n_items)
        resampled = [d for i in idx for d in per_item_deltas[i]]
        if resampled:
            boot_means.append(float(np.mean(resampled)))
    if not boot_means:
        return BootstrapResult(point=point, lo=point, hi=point)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boot_means, 100 * alpha))
    hi = float(np.percentile(boot_means, 100 * (1.0 - alpha)))
    return BootstrapResult(point=point, lo=lo, hi=hi)


def combine_confidences(
    values: Iterable[object], *, method: str = "max"
) -> float | None:
    """Collapse a slot's confidence readings into one value (``max`` or ``mean``).

    Non-numeric / non-finite / boolean entries are ignored; ``None`` when no
    valid value remains. Feeds only the *descriptive* risk-coverage curve —
    never a gate input.
    """
    if method not in ("max", "mean"):
        raise ValueError(f"method must be 'max' or 'mean', got {method!r}")
    vals = [
        float(v)
        for v in values
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    ]
    if not vals:
        return None
    return float(np.mean(vals)) if method == "mean" else max(vals)


def risk_coverage(
    items: Sequence[tuple[float | None, bool]],
    *,
    target_accuracy: float | None = None,
) -> RiskCoverage:
    """Descriptive accuracy-vs-coverage (selective-prediction) curve + Chow point.

    ``items`` are ``(confidence, correct)`` pairs; ``None`` confidence sorts
    last (least trusted). Points are ``(coverage, accuracy, threshold)`` for
    each cumulative top-k by confidence. ``chow_coverage`` (when
    ``target_accuracy`` is given) is the maximum coverage whose selective
    accuracy still clears the target.

    DESCRIPTIVE-ONLY: ``calibrated`` is False — confidence is not proven
    calibrated, so this must never be a pass/fail gate input.
    """
    if not items:
        return RiskCoverage(points=[], chow_coverage=None)

    ordered = sorted(
        items,
        key=lambda it: (it[0] is not None, it[0] if it[0] is not None else 0.0),
        reverse=True,
    )
    n = len(ordered)
    points: list[tuple[float, float, float | None]] = []
    correct_so_far = 0
    for k, (conf, correct) in enumerate(ordered, start=1):
        correct_so_far += 1 if correct else 0
        points.append((k / n, correct_so_far / k, conf))

    chow_coverage: float | None = None
    if target_accuracy is not None:
        for coverage, accuracy, _ in points:
            if accuracy >= target_accuracy:
                chow_coverage = coverage
    return RiskCoverage(points=points, chow_coverage=chow_coverage)
