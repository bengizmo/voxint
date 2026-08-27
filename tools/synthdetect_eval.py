#!/usr/bin/env python3
"""Host-side scorer for the synthdetect eval harness (issue #144).

A MAINTAINER instrument, never shipped to users and never installed into a
service image. It consumes a raw-score JOURNAL produced by the GPU inference
runner (``synthdetect_infer.py``, a later session) joined to a corpus MANIFEST
(``synthdetect_corpus.py``) and produces the numbers the numerics doctrine needs
as measured evidence: detection-error-rate metrics, paired per-clip regression,
a calibration policy, and a house-style report.

The raw score is the durable contract. Everything here derives from it; nothing
here runs a model. The scorer fixes ONE score polarity: a higher ``raw_score``
means MORE likely synthetic (the positive class is ``spoof``). The runner is
responsible for emitting scores in that polarity; a checkpoint that natively
emits bona-fide logits is inverted upstream, not here, so the stored raw score
is always comparable across models.

Subcommands:

* ``score`` -- EER (via sklearn ``roc_curve`` + a unit-tested crossing
  interpolation), a seeded-bootstrap CI, AUC, operating points at fixed FPRs,
  and per-stratum breakdowns, for ONE corpus/split.
* ``compare`` -- paired per-clip diff of two journals over the SAME clips (gate-2
  implementation parity and subset regression): max-abs score delta, rank
  correlation, decision agreement. Aggregate EER matching is NEVER accepted as
  equivalence evidence, so this reads scores per clip, not summary numbers.
* ``verdict-windowing`` -- S5 windowing verdict: paired comparison of upstream vs
  production windowing on bona fide clips. Reports per-clip score deltas, FPR at
  a threshold sweep, per-stratum breakdown, and per-window analysis.
* ``calibrate`` -- fit a Platt policy on RAW logits over the calibration split;
  the primary shipped threshold is at FPR 5% (FPR 1% is a noisy diagnostic from
  ~1000 bona fide clips), with a reliability curve + Brier score.
* ``report`` -- metrics JSON -> dated Markdown in house style, with the model's
  ``license_class`` beside every number.

The heavy dependency (scikit-learn ``roc_curve`` / ``roc_auc_score``) is loaded
LAZILY so importing this module and running the pure helpers works in the dev
lane without the ``synthdetect-eval`` extra. Run scoring with the extra::

    uv run --isolated --extra synthdetect-eval \\
        tools/synthdetect_eval.py score --journal <j.jsonl> --manifest <m.json> \\
        --split eval --out <metrics.json>
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from synthdetect_corpus import ClipEntry, Manifest, load_manifest  # noqa: E402
from synthdetect_sources import MODELS, SELECTION_SEED  # noqa: E402

JOURNAL_SCHEMA_VERSION = 1
METRICS_SCHEMA_VERSION = 1

# The positive class for detection metrics is the synthetic one: a "false
# positive" is a bona fide clip flagged synthetic (the field friction driver),
# a "false negative" is a missed spoof. Encoded as 1=spoof / 0=bona_fide.
POSITIVE_LABEL = "spoof"

# The shipped operating point (primary) and the noisy diagnostic one. FPR 1%
# from ~1000 bona fide clips is quantile noise; it is reported, never shipped.
PRIMARY_FPR = 0.05
DIAGNOSTIC_FPR = 0.01

# Seeded-bootstrap defaults; the seed is domain-separated from SELECTION_SEED so
# a bootstrap reshuffle is tied to the one corpus seed, not an ad-hoc constant.
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_ALPHA = 0.05


class EvalError(Exception):
    """A user-facing input problem (bad journal/manifest, degenerate scores)."""


# The sklearn ROC stack is loaded once, lazily, so `import synthdetect_eval`,
# the pure metric helpers, and their unit tests all work without the extra.
_roc_curve: Any = None
_roc_auc_score: Any = None


def _load_sklearn() -> None:
    global _roc_curve, _roc_auc_score
    if _roc_curve is not None:
        return
    from sklearn.metrics import roc_auc_score, roc_curve

    _roc_curve, _roc_auc_score = roc_curve, roc_auc_score


def _seed_int(*parts: str) -> int:
    """A stable 63-bit integer seed from SELECTION_SEED + context parts."""
    digest = hashlib.sha256(("\x00".join((SELECTION_SEED, *parts))).encode()).hexdigest()
    return int(digest[:16], 16) & ((1 << 63) - 1)


# --------------------------------------------------------------------------- #
# Pure metric helpers (no sklearn; unit-tested directly on arrays)
# --------------------------------------------------------------------------- #
def eer_from_roc(fpr: Any, fnr: Any, thresholds: Any = None) -> tuple[float, float | None]:
    """EER as the fpr==fnr crossing, linearly interpolated. Pure (no sklearn).

    ``fpr`` is non-decreasing and ``fnr`` non-increasing along the ROC (as the
    decision threshold sweeps). The equal-error point is where the two curves
    cross; between the last point with ``fpr < fnr`` and the first with
    ``fpr >= fnr`` the crossing is interpolated linearly. Returns
    ``(eer, threshold_at_eer)`` -- the threshold is interpolated on the same
    segment when ``thresholds`` is given, else None. Degenerate inputs (a curve
    that never crosses) fall back to the nearest endpoint rather than raising, so
    a caller can still surface a number with the diagnostic that it is degenerate.
    """
    fpr = np.asarray(fpr, dtype=float)
    fnr = np.asarray(fnr, dtype=float)
    if fpr.shape != fnr.shape or fpr.ndim != 1 or fpr.size == 0:
        raise EvalError("eer_from_roc: fpr and fnr must be equal-length 1-D arrays")
    crossings = np.where(fpr >= fnr)[0]
    if crossings.size == 0:
        return float(fnr[-1]), (float(thresholds[-1]) if thresholds is not None else None)
    i = int(crossings[0])
    if i == 0:
        eer = float((fpr[0] + fnr[0]) / 2.0)
        thr0 = float(thresholds[0]) if thresholds is not None else None
        # sklearn's roc_curve prepends an inf threshold sentinel; never surface it.
        return eer, (thr0 if thr0 is not None and math.isfinite(thr0) else None)
    x0, x1 = float(fpr[i - 1]), float(fpr[i])
    y0, y1 = float(fnr[i - 1]), float(fnr[i])
    denom = (x1 - x0) - (y1 - y0)
    t = 0.0 if denom == 0 else (y0 - x0) / denom
    t = min(max(t, 0.0), 1.0)
    eer = x0 + t * (x1 - x0)
    thr: float | None = None
    if thresholds is not None:
        th = np.asarray(thresholds, dtype=float)
        t0, t1 = float(th[i - 1]), float(th[i])
        # Interpolate only across finite thresholds; the inf sentinel that
        # roc_curve prepends must never poison the operating point into a NaN.
        if math.isfinite(t0) and math.isfinite(t1):
            thr = t0 + t * (t1 - t0)
        elif math.isfinite(t1):
            thr = t1
        elif math.isfinite(t0):
            thr = t0
    # An EER is a rate; clamp to [0, 1] so float noise on a pathological curve
    # can never surface a rate outside the unit interval.
    return float(min(1.0, max(0.0, eer))), thr


def _sigmoid(z: Any) -> Any:
    # Numerically stable logistic; avoids overflow for large |z|.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def fit_platt(
    scores: Any, labels: Any, max_iter: int = 100, sample_weights: Any = None,
) -> tuple[float, float]:
    """Fit Platt scaling ``P(spoof|s)=sigmoid(A*s+B)`` on RAW scores. Pure numpy.

    Uses the Lin-Lin-Weng (2007) target-smoothed, regularized Newton procedure
    (the numerically stable form of Platt's original), so a separable calibration
    set does not drive the parameters to infinity. ``labels`` are 1=spoof /
    0=bona_fide; ``scores`` are raw logits (higher = more synthetic). Returns
    ``(A, B)``. A is not constrained in sign here, but on well-behaved detector
    scores it comes out positive (higher score -> higher spoof probability).

    ``sample_weights``, when given, assigns per-clip importance (e.g.
    ``1/partition_group_size``). Weights enter the effective class priors, the
    target smoothing, all gradient/Hessian/NLL accumulations, and the
    initialisation. Pass ``None`` for uniform weighting.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    if s.shape != y.shape or s.ndim != 1 or s.size == 0:
        raise EvalError("fit_platt: scores and labels must be equal-length 1-D arrays")
    if not np.all(np.isfinite(s)):
        raise EvalError("fit_platt: raw scores must be finite")
    if sample_weights is not None:
        sw = np.asarray(sample_weights, dtype=float)
        if sw.shape != s.shape:
            raise EvalError("fit_platt: sample_weights must match scores in length")
        if not np.all(np.isfinite(sw)) or np.any(sw < 0):
            raise EvalError("fit_platt: sample_weights must be finite and non-negative")
    else:
        sw = np.ones_like(s)
    prior1 = float(np.sum(sw[y == 1.0]))
    prior0 = float(np.sum(sw[y == 0.0]))
    if prior1 == 0.0 or prior0 == 0.0:
        raise EvalError("fit_platt: calibration set needs both bona_fide and spoof clips")
    hi = (prior1 + 1.0) / (prior1 + 2.0)
    lo = 1.0 / (prior0 + 2.0)
    t = np.where(y == 1.0, hi, lo)

    a, b = 0.0, math.log((prior1 + 1.0) / (prior0 + 1.0))
    min_step, sigma = 1e-10, 1e-12
    for _ in range(max_iter):
        z = a * s + b
        p = _sigmoid(z)
        d1 = (p - t) * sw
        grad_a = float(np.sum(d1 * s))
        grad_b = float(np.sum(d1))
        pw = p * (1.0 - p) * sw
        h11 = float(np.sum(pw * s * s)) + sigma
        h22 = float(np.sum(pw)) + sigma
        h12 = float(np.sum(pw * s))
        det = h11 * h22 - h12 * h12
        if det == 0.0:
            break
        da = -(h22 * grad_a - h12 * grad_b) / det
        db = -(-h12 * grad_a + h11 * grad_b) / det
        gd = grad_a * da + grad_b * db
        if gd >= 0:
            break
        step = 1.0
        eps = 1e-12
        while step >= min_step:
            new_a, new_b = a + step * da, b + step * db
            znew = new_a * s + new_b
            pnew = _sigmoid(znew)
            log_l = t * np.log(pnew + eps) + (1.0 - t) * np.log(1.0 - pnew + eps)
            nll = -float(np.sum(sw * log_l))
            zold = a * s + b
            pold = _sigmoid(zold)
            log_l_old = t * np.log(pold + eps) + (1.0 - t) * np.log(1.0 - pold + eps)
            nll_old = -float(np.sum(sw * log_l_old))
            if nll < nll_old + 1e-4 * step * gd:
                break
            step /= 2.0
        a, b = a + step * da, b + step * db
        if abs(step * da) < 1e-9 and abs(step * db) < 1e-9:
            break
    return float(a), float(b)


def apply_platt(scores: Any, a: float, b: float) -> Any:
    """Map raw scores to spoof probabilities under a fitted Platt ``(A, B)``."""
    return _sigmoid(a * np.asarray(scores, dtype=float) + b)


def brier_score(
    probabilities: Any, labels: Any, sample_weights: Any = None,
) -> float:
    """Weighted mean squared error of calibrated probabilities vs 1=spoof / 0=bona_fide."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.shape != y.shape or p.size == 0:
        raise EvalError("brier_score: probabilities and labels must be equal-length arrays")
    sq = (p - y) ** 2
    if sample_weights is not None:
        sw = np.asarray(sample_weights, dtype=float)
        return float(np.sum(sw * sq) / np.sum(sw))
    return float(np.mean(sq))


def reliability_curve(probabilities: Any, labels: Any, n_bins: int = 10) -> list[dict[str, float]]:
    """Equal-width reliability bins: (mean predicted prob, empirical spoof rate).

    Empty bins are omitted (they carry no reliability signal). Each kept bin
    reports its count so a reader can weight sparsely populated bins.
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float)
    if n_bins < 1:
        raise EvalError("reliability_curve: n_bins must be >= 1")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict[str, float]] = []
    for lo, hi in itertools.pairwise(edges):
        # The last bin is closed on the right so a probability of exactly 1.0 lands.
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        count = int(np.sum(mask))
        if count == 0:
            continue
        out.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "mean_predicted": float(np.mean(p[mask])),
                "empirical_rate": float(np.mean(y[mask])),
                "count": float(count),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# sklearn-backed scoring (roc_curve / roc_auc_score)
# --------------------------------------------------------------------------- #
def _check_two_class(labels: Any) -> np.ndarray:
    y = np.asarray(labels, dtype=int)
    classes = set(np.unique(y).tolist())
    if classes - {0, 1}:
        raise EvalError(f"labels must be 0/1, found {sorted(classes)}")
    if classes != {0, 1}:
        raise EvalError("scoring needs BOTH classes present (single-class set rejected)")
    return y


def compute_eer(labels: Any, scores: Any) -> tuple[float, float]:
    """EER + its threshold via sklearn ``roc_curve`` and :func:`eer_from_roc`."""
    _load_sklearn()
    y = _check_two_class(labels)
    s = np.asarray(scores, dtype=float)
    if s.shape != y.shape:
        raise EvalError("compute_eer: labels and scores must be equal length")
    fpr, tpr, thr = _roc_curve(y, s)
    fnr = 1.0 - tpr
    eer, threshold = eer_from_roc(fpr, fnr, thr)
    # eer_from_roc suppresses the inf sentinel and any non-finite bound, so a
    # degenerate curve can leave threshold None; fall back to the score median so
    # a caller never gets a NaN operating point.
    if threshold is None or not math.isfinite(threshold):
        threshold = float(np.median(s))
    return eer, threshold


def compute_auc(labels: Any, scores: Any) -> float:
    """ROC AUC with spoof as the positive class."""
    _load_sklearn()
    y = _check_two_class(labels)
    return float(_roc_auc_score(y, np.asarray(scores, dtype=float)))


def operating_point(labels: Any, scores: Any, target_fpr: float) -> dict[str, float]:
    """Threshold achieving <= ``target_fpr``, with its realized FPR/FNR.

    Sweeps the ROC and picks the highest-recall threshold whose FPR does not
    exceed the target (a conservative choice: never overshoot the bona-fide
    false-positive budget). Reports the realized FPR and FNR at that threshold so
    quantile noise at a tight target is visible, not hidden.
    """
    _load_sklearn()
    y = _check_two_class(labels)
    s = np.asarray(scores, dtype=float)
    fpr, tpr, thr = _roc_curve(y, s)
    fnr = 1.0 - tpr
    eligible = np.where(fpr <= target_fpr + 1e-12)[0]
    # roc_curve is sorted by decreasing threshold, so FPR is non-decreasing; the
    # last eligible index is the highest FPR still within budget (max recall).
    idx = int(eligible[-1]) if eligible.size else 0
    threshold = float(thr[idx])
    # roc_curve prepends an inf threshold sentinel at the FPR=0 point; if a tight
    # budget admits only that point, emit a finite "reject everything" threshold
    # (above every score) so the operating point stays valid JSON.
    if not math.isfinite(threshold):
        threshold = float(np.max(s)) + 1.0
    return {
        "target_fpr": float(target_fpr),
        "threshold": threshold,
        "realized_fpr": float(fpr[idx]),
        "realized_fnr": float(fnr[idx]),
    }


def bootstrap_eer_ci(
    labels: Any,
    scores: Any,
    *,
    context: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
    alpha: float = BOOTSTRAP_ALPHA,
) -> dict[str, float]:
    """Seeded percentile bootstrap CI for EER (paired resample of clips).

    The RNG is seeded from SELECTION_SEED + ``context`` so a rerun reproduces the
    interval byte-for-byte. Resamples that happen to draw a single class are
    skipped (their EER is undefined), and the reported ``n_valid`` shows how many
    resamples contributed -- a small ``n_valid`` is itself a signal the corpus is
    too small for a stable interval.
    """
    y = _check_two_class(labels)
    s = np.asarray(scores, dtype=float)
    if s.shape != y.shape:
        raise EvalError("bootstrap_eer_ci: labels and scores must be equal length")
    n = y.size
    rng = np.random.default_rng(_seed_int("bootstrap", context))
    eers: list[float] = []
    for _ in range(resamples):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if set(np.unique(yb).tolist()) != {0, 1}:
            continue
        eer, _ = compute_eer(yb, s[idx])
        eers.append(eer)
    if not eers:
        raise EvalError(f"bootstrap_eer_ci[{context}]: no resample had both classes")
    arr = np.array(eers)
    # Pin the quantile method so the committed-fixture metrics stay byte-stable
    # across numpy versions (the default interpolation has changed before).
    return {
        "lo": float(np.quantile(arr, alpha / 2.0, method="linear")),
        "hi": float(np.quantile(arr, 1.0 - alpha / 2.0, method="linear")),
        "n_valid": len(eers),
    }


# --------------------------------------------------------------------------- #
# Journal + manifest join
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClipScore:
    """One journal result: a raw score XOR a skip reason, keyed by clip_id."""

    clip_id: str
    raw_score: float | None
    skip_reason: str | None
    n_windows: int
    window_scores: tuple[float, ...] | None = None


@dataclass(frozen=True)
class Journal:
    """A parsed inference journal: identity header + per-clip results."""

    header: dict[str, Any]
    results: tuple[ClipScore, ...]


def _validate_header(header: Any) -> dict[str, Any]:
    if not isinstance(header, dict):
        raise EvalError("journal header must be an object")
    if header.get("kind") != "synthdetect_journal":
        raise EvalError("first journal record must be the header (kind synthdetect_journal)")
    if header.get("schema_version") != JOURNAL_SCHEMA_VERSION:
        raise EvalError(f"journal schema_version must be {JOURNAL_SCHEMA_VERSION}")
    for key in ("inference_space", "model_id", "manifest_sha256"):
        if not isinstance(header.get(key), str) or not header[key]:
            raise EvalError(f"journal header.{key} must be a non-empty string")
    if not _is_sha256(header["manifest_sha256"]):
        raise EvalError("journal header.manifest_sha256 must be 64 lowercase hex chars")
    # Bind the header to the registry: the model must be known (a typo'd or
    # unregistered model can never produce official metrics) and its declared
    # inference space must match the registry's, so a weights/runtime identity can
    # never silently diverge from the model it claims to be.
    model = MODELS.get(header["model_id"])
    if model is None:
        raise EvalError(f"journal header.model_id {header['model_id']!r} is not a known model")
    if header["inference_space"] != model.inference_space:
        raise EvalError(
            f"journal header.inference_space {header['inference_space']!r} does not match the "
            f"registry inference space {model.inference_space!r} for model {header['model_id']!r}"
        )
    windowing = header.get("windowing")
    if not isinstance(windowing, dict) or "pooling" not in windowing:
        raise EvalError("journal header.windowing must be an object with a pooling policy")
    return header


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _validate_result(raw: Any, lineno: int) -> ClipScore:
    if not isinstance(raw, dict):
        raise EvalError(f"journal line {lineno}: result must be an object")
    clip_id = raw.get("clip_id")
    if not isinstance(clip_id, str) or not clip_id:
        raise EvalError(f"journal line {lineno}: clip_id must be a non-empty string")
    raw_score = raw.get("raw_score")
    skip_reason = raw.get("skip_reason")
    has_score = raw_score is not None
    has_skip = skip_reason is not None
    # The XOR invariant: a clip is scored OR skipped, never both, never neither.
    if has_score == has_skip:
        raise EvalError(
            f"journal line {lineno} ({clip_id}): exactly one of raw_score / skip_reason required"
        )
    if has_score:
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise EvalError(f"journal line {lineno} ({clip_id}): raw_score must be a number")
        if not math.isfinite(float(raw_score)):
            raise EvalError(f"journal line {lineno} ({clip_id}): raw_score must be finite")
    if has_skip and (not isinstance(skip_reason, str) or not skip_reason.strip()):
        raise EvalError(
            f"journal line {lineno} ({clip_id}): skip_reason must be a non-empty string"
        )
    n_windows = raw.get("n_windows", 0)
    if not isinstance(n_windows, int) or isinstance(n_windows, bool) or n_windows < 0:
        raise EvalError(f"journal line {lineno} ({clip_id}): n_windows must be an int >= 0")
    ws_raw = raw.get("window_scores")
    ws: tuple[float, ...] | None = None
    if ws_raw is not None:
        if not isinstance(ws_raw, list) or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))
            for v in ws_raw
        ):
            raise EvalError(
                f"journal line {lineno} ({clip_id}): "
                "window_scores must be a list of finite numbers"
            )
        ws = tuple(float(v) for v in ws_raw)
    return ClipScore(
        clip_id=clip_id,
        raw_score=float(raw_score) if raw_score is not None else None,
        skip_reason=skip_reason if skip_reason is not None else None,
        n_windows=n_windows,
        window_scores=ws,
    )


def parse_journal(text: str) -> Journal:
    """Parse a JSONL journal: first record is the header, the rest are results.

    Fail-closed on a missing/malformed header, a duplicate clip_id, or a result
    violating the score/skip XOR. Blank lines are ignored so a partially-flushed
    write-ahead journal still parses.
    """
    header: dict[str, Any] | None = None
    results: list[ClipScore] = []
    seen: set[str] = set()
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalError(f"journal line {lineno}: invalid JSON: {exc}") from exc
        if header is None:
            header = _validate_header(obj)
            continue
        result = _validate_result(obj, lineno)
        if result.clip_id in seen:
            raise EvalError(f"journal line {lineno}: duplicate clip_id {result.clip_id!r}")
        seen.add(result.clip_id)
        results.append(result)
    if header is None:
        raise EvalError("journal is empty (no header record)")
    if not results:
        raise EvalError("journal has a header but no clip results")
    return Journal(header=header, results=tuple(results))


@dataclass(frozen=True)
class ScoredClip:
    """A journal score joined to its manifest ground truth."""

    clip_id: str
    label: int  # 1=spoof, 0=bona_fide
    stratum: str
    raw_score: float
    partition_group_id: str | None = None


def join_scores(
    journal: Journal, manifest: Manifest, *, split: str | None
) -> tuple[list[ScoredClip], list[ClipScore]]:
    """Join scored journal clips to manifest ground truth for ONE split.

    Returns ``(scored, skipped)``. Every journal clip must exist in the manifest
    (a score for an unknown clip is a fail-closed error, not a silent drop).
    ``split`` restricts to clips whose manifest split matches (None = all splits).
    Skipped journal clips (``skip_reason`` set) are returned separately so the
    caller reports a skip rate -- a detector silently skipping short turns is a
    feature that does not work.
    """
    by_id: dict[str, ClipEntry] = {c.clip_id: c for c in manifest.clips}
    scored: list[ScoredClip] = []
    skipped: list[ClipScore] = []
    for result in journal.results:
        clip = by_id.get(result.clip_id)
        if clip is None:
            raise EvalError(
                f"journal clip {result.clip_id!r} is not in the manifest (cannot label it)"
            )
        if split is not None and clip.split != split:
            continue
        if result.skip_reason is not None:
            skipped.append(result)
            continue
        assert result.raw_score is not None  # XOR invariant guarantees this
        scored.append(
            ScoredClip(
                clip_id=clip.clip_id,
                label=1 if clip.label == POSITIVE_LABEL else 0,
                stratum=clip.stratum,
                raw_score=result.raw_score,
                partition_group_id=getattr(clip, "partition_group_id", None),
            )
        )
    return scored, skipped


# --------------------------------------------------------------------------- #
# `score` command
# --------------------------------------------------------------------------- #
def _score_set(clips: list[ScoredClip], *, context: str) -> dict[str, Any]:
    labels = [c.label for c in clips]
    scores = [c.raw_score for c in clips]
    eer, eer_threshold = compute_eer(labels, scores)
    return {
        "n": len(clips),
        "n_spoof": int(sum(labels)),
        "n_bona_fide": int(len(labels) - sum(labels)),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "auc": compute_auc(labels, scores),
        "eer_ci": bootstrap_eer_ci(labels, scores, context=context),
        "operating_points": [
            operating_point(labels, scores, PRIMARY_FPR),
            operating_point(labels, scores, DIAGNOSTIC_FPR),
        ],
    }


def score_report(
    journal: Journal, manifest: Manifest, *, split: str | None
) -> dict[str, Any]:
    """Build the full ``score`` metrics object (overall + per-stratum + skips)."""
    scored, skipped = join_scores(journal, manifest, split=split)
    if not scored:
        raise EvalError("no scored clips for this split (nothing to score)")
    total = len(scored) + len(skipped)
    report: dict[str, Any] = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "kind": "synthdetect_score_report",
        "inference_space": journal.header["inference_space"],
        "model_id": journal.header["model_id"],
        "split": split,
        "skip_rate": len(skipped) / total if total else 0.0,
        "n_skipped": len(skipped),
        "overall": _score_set(scored, context=f"overall:{split}"),
    }
    strata = sorted({c.stratum for c in scored})
    per_stratum: dict[str, Any] = {}
    for stratum in strata:
        subset = [c for c in scored if c.stratum == stratum]
        labels = {c.label for c in subset}
        # A stratum with only one class cannot yield an EER; report its makeup
        # rather than raising, so the overall number still renders.
        if labels != {0, 1}:
            per_stratum[stratum] = {
                "n": len(subset),
                "single_class": True,
                "n_spoof": int(sum(c.label for c in subset)),
            }
            continue
        per_stratum[stratum] = _score_set(subset, context=f"stratum:{stratum}:{split}")
    report["per_stratum"] = per_stratum
    return report


# --------------------------------------------------------------------------- #
# `compare` command (paired per-clip diff of two journals)
# --------------------------------------------------------------------------- #
def compare_journals(
    left: Journal, right: Journal, *, decision_threshold: float
) -> dict[str, Any]:
    """Paired per-clip diff over the clips SCORED in both journals.

    Requires a non-empty intersection of scored clip_ids and reports max-abs and
    mean-abs score delta, Spearman rank correlation, and decision agreement at
    ``decision_threshold`` (score >= threshold => spoof). This is the gate-2
    equivalence evidence: aggregate EER matching is deliberately NOT computed
    here, because two different functions can share an EER while disagreeing per
    clip.
    """
    # Parity evidence requires the two runs to share an identity: comparing
    # scores from different models, inference spaces, corpora, or windowing is
    # meaningless. Fail closed on any mismatch (a 3-of-3 review finding).
    for key in ("model_id", "inference_space", "manifest_sha256"):
        if left.header.get(key) != right.header.get(key):
            raise EvalError(
                f"compare: journals disagree on header.{key} "
                f"({left.header.get(key)!r} vs {right.header.get(key)!r})"
            )
    if left.header.get("windowing", {}).get("pooling") != right.header.get("windowing", {}).get(
        "pooling"
    ):
        raise EvalError("compare: journals disagree on windowing pooling policy")

    left_by = {r.clip_id: r.raw_score for r in left.results if r.raw_score is not None}
    right_by = {r.clip_id: r.raw_score for r in right.results if r.raw_score is not None}
    common = sorted(set(left_by) & set(right_by))
    if not common:
        raise EvalError("compare: the two journals share no scored clips")
    a = np.array([left_by[c] for c in common], dtype=float)
    b = np.array([right_by[c] for c in common], dtype=float)
    diff = np.abs(a - b)
    decisions_a = a >= decision_threshold
    decisions_b = b >= decision_threshold
    agree = int(np.sum(decisions_a == decisions_b))
    # Asymmetric coverage weakens the parity gate invisibly (a windowing change
    # that newly skips clips shrinks n_common with no other signal), so report
    # each side's scored-only ids rather than silently intersecting them away.
    left_only = sorted(set(left_by) - set(right_by))
    right_only = sorted(set(right_by) - set(left_by))
    return {
        "kind": "synthdetect_compare_report",
        "n_common": len(common),
        "max_abs_delta": float(np.max(diff)),
        "mean_abs_delta": float(np.mean(diff)),
        "spearman": _spearman(a, b),
        "decision_threshold": float(decision_threshold),
        "decision_agreement": agree / len(common),
        "n_disagree": len(common) - agree,
        "n_scored_left_only": len(left_only),
        "n_scored_right_only": len(right_only),
        "scored_left_only": left_only,
        "scored_right_only": right_only,
    }


def _spearman(a: Any, b: Any) -> float:
    """Spearman rank correlation (pure numpy; average-rank ties). 0.0 if constant."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = math.sqrt(float(np.sum(ra * ra)) * float(np.sum(rb * rb)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(ra * rb) / denom)


def _rankdata(x: Any) -> np.ndarray:
    """Average ranks (ties share the mean rank), matching scipy's default."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, x.size + 1, dtype=float)
    # Average tied ranks.
    sx = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            mean_rank = (i + 1 + j + 1) / 2.0
            ranks[order[i : j + 1]] = mean_rank
        i = j + 1
    return ranks


# --------------------------------------------------------------------------- #
# `verdict-windowing` command (upstream vs production FPR stability)
# --------------------------------------------------------------------------- #
def _fpr_at_thresholds(
    scores: np.ndarray, thresholds: np.ndarray,
) -> np.ndarray:
    """Bona fide FPR at each threshold: fraction scoring >= threshold."""
    n = len(scores)
    if n == 0:
        return np.zeros_like(thresholds, dtype=float)
    return np.array([float(np.sum(scores >= t)) / n for t in thresholds])


def _percentile_thresholds(all_scores: np.ndarray) -> np.ndarray:
    """Threshold sweep from the data: percentiles 50..99 plus 99.5 and 99.9."""
    pcts = [*list(range(50, 100, 5)), 99, 99.5, 99.9]
    return np.unique(np.percentile(all_scores, pcts))


def compare_windowing(
    upstream: Journal,
    production: Journal,
    *,
    manifest: Manifest | None = None,
) -> dict[str, Any]:
    """Windowing verdict: does production windowing raise bona fide FPR?

    The two journals must share model_id, inference_space, and manifest_sha256
    but differ in windowing mode. With bona fide-only data, FPR is the fraction
    of bona fide clips scoring above a threshold (higher = more synthetic).
    Per-window scores, if present, are summarized per clip.
    """
    for key in ("model_id", "inference_space", "manifest_sha256"):
        if upstream.header.get(key) != production.header.get(key):
            raise EvalError(
                f"verdict-windowing: journals disagree on header.{key} "
                f"({upstream.header.get(key)!r} vs {production.header.get(key)!r})"
            )
    u_wind = upstream.header.get("windowing", {})
    p_wind = production.header.get("windowing", {})
    u_mode = u_wind.get("mode", "unknown")
    p_mode = p_wind.get("mode", "unknown")
    if u_mode != "upstream" or p_mode != "production":
        raise EvalError(
            f"verdict-windowing: expected modes 'upstream' and 'production', "
            f"got {u_mode!r} and {p_mode!r}"
        )
    if u_wind.get("pooling") != p_wind.get("pooling"):
        raise EvalError(
            f"verdict-windowing: journals disagree on pooling policy "
            f"({u_wind.get('pooling')!r} vs {p_wind.get('pooling')!r})"
        )

    u_by = {r.clip_id: r for r in upstream.results if r.raw_score is not None}
    p_by = {r.clip_id: r for r in production.results if r.raw_score is not None}
    common = sorted(set(u_by) & set(p_by))
    if not common:
        raise EvalError("verdict-windowing: no clips scored in both journals")

    u_scores = np.array([u_by[c].raw_score for c in common], dtype=float)
    p_scores = np.array([p_by[c].raw_score for c in common], dtype=float)
    delta = p_scores - u_scores

    all_scores = np.concatenate([u_scores, p_scores])
    thresholds = _percentile_thresholds(all_scores)
    u_fpr = _fpr_at_thresholds(u_scores, thresholds)
    p_fpr = _fpr_at_thresholds(p_scores, thresholds)

    fpr_sweep = [
        {
            "threshold": float(t),
            "upstream_fpr": float(uf),
            "production_fpr": float(pf),
            "delta_fpr": float(pf - uf),
        }
        for t, uf, pf in zip(thresholds, u_fpr, p_fpr, strict=True)
    ]

    u_n_win = np.array([u_by[c].n_windows for c in common], dtype=int)
    p_n_win = np.array([p_by[c].n_windows for c in common], dtype=int)

    n_prod_higher = int(np.sum(delta > 0))
    n_prod_lower = int(np.sum(delta < 0))
    n_tied = int(np.sum(delta == 0))

    result: dict[str, Any] = {
        "kind": "synthdetect_windowing_verdict",
        "schema_version": METRICS_SCHEMA_VERSION,
        "model_id": upstream.header["model_id"],
        "inference_space": upstream.header["inference_space"],
        "manifest_sha256": upstream.header["manifest_sha256"],
        "upstream_mode": u_mode,
        "production_mode": p_mode,
        "n_common": len(common),
        "n_scored_upstream_only": len(set(u_by) - set(p_by)),
        "n_scored_production_only": len(set(p_by) - set(u_by)),
        "score_delta": {
            "mean": float(np.mean(delta)),
            "median": float(np.median(delta)),
            "std": float(np.std(delta)),
            "max_abs": float(np.max(np.abs(delta))),
            "min": float(np.min(delta)),
            "max": float(np.max(delta)),
        },
        "direction": {
            "n_production_higher": n_prod_higher,
            "n_production_lower": n_prod_lower,
            "n_tied": n_tied,
        },
        "upstream_summary": {
            "mean": float(np.mean(u_scores)),
            "std": float(np.std(u_scores)),
            "median": float(np.median(u_scores)),
            "p5": float(np.percentile(u_scores, 5)),
            "p95": float(np.percentile(u_scores, 95)),
        },
        "production_summary": {
            "mean": float(np.mean(p_scores)),
            "std": float(np.std(p_scores)),
            "median": float(np.median(p_scores)),
            "p5": float(np.percentile(p_scores, 5)),
            "p95": float(np.percentile(p_scores, 95)),
        },
        "windows": {
            "upstream_mean_n": float(np.mean(u_n_win)),
            "production_mean_n": float(np.mean(p_n_win)),
            "upstream_max_n": int(np.max(u_n_win)),
            "production_max_n": int(np.max(p_n_win)),
        },
        "fpr_sweep": fpr_sweep,
    }

    prod_spreads = []
    for cid in common:
        ws = p_by[cid].window_scores
        if ws is not None and len(ws) > 1:
            prod_spreads.append(max(ws) - min(ws))
    if prod_spreads:
        sa = np.array(prod_spreads, dtype=float)
        result["per_window_spread"] = {
            "n_multi_window_clips": len(prod_spreads),
            "mean_spread": float(np.mean(sa)),
            "median_spread": float(np.median(sa)),
            "max_spread": float(np.max(sa)),
        }

    if manifest is not None:
        by_id = {c.clip_id: c for c in manifest.clips}
        strata_map: dict[str, list[int]] = {}
        for i, cid in enumerate(common):
            entry = by_id.get(cid)
            if entry is None:
                raise EvalError(
                    f"verdict-windowing: common clip {cid!r} not in the supplied manifest"
                )
            strata_map.setdefault(entry.stratum, []).append(i)
        per_stratum: dict[str, Any] = {}
        for stratum in sorted(strata_map):
            idx = np.array(strata_map[stratum])
            sd = delta[idx]
            su = u_scores[idx]
            sp = p_scores[idx]
            per_stratum[stratum] = {
                "n": len(idx),
                "mean_delta": float(np.mean(sd)),
                "median_delta": float(np.median(sd)),
                "std_delta": float(np.std(sd)),
                "max_abs_delta": float(np.max(np.abs(sd))),
                "upstream_mean": float(np.mean(su)),
                "production_mean": float(np.mean(sp)),
                "n_production_higher": int(np.sum(sd > 0)),
                "n_production_lower": int(np.sum(sd < 0)),
            }
        result["per_stratum"] = per_stratum

    return result


# --------------------------------------------------------------------------- #
# `calibrate` command (Platt on raw logits over the calibration split)
# --------------------------------------------------------------------------- #
def calibrate_policy(
    journal: Journal,
    manifest: Manifest,
    *,
    policy_id: str,
    exclude_strata: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Fit the calibration policy on the CALIBRATION split and report its fit.

    Platt is fit on raw logits; the primary threshold is set at FPR 5% (the
    shipped operating point) and FPR 1% is reported as a diagnostic. The cohort
    hash binds the policy to exactly the clips it was fit on, so a later re-fit on
    a different cohort is a visibly different policy.

    ``exclude_strata`` removes clips whose stratum contains any of the given
    substrings (case-insensitive). Use this to fit on a generator subset when a
    generator is uncalibrable (e.g. near-chance EER) and would poison the Platt
    slope for generators the detector can actually discriminate.
    """
    all_scored, _ = join_scores(journal, manifest, split="calibration")
    if not all_scored:
        raise EvalError("calibrate: no clips in the calibration split")
    if exclude_strata:
        lowered = tuple(p.lower() for p in exclude_strata)
        scored = [c for c in all_scored if not any(p in c.stratum.lower() for p in lowered)]
        n_excluded = len(all_scored) - len(scored)
    else:
        scored = all_scored
        n_excluded = 0
    if not scored:
        raise EvalError("calibrate: all clips excluded after strata filter")
    # Partition-group weighting: one effective observation per group. Clips
    # sharing a partition_group_id (e.g. a bona fide parent and its TTS spoof)
    # each get weight 1/group_size so the group contributes equally to a
    # singleton. Clips with no partition_group_id are treated as singletons.
    group_counts: dict[str, int] = {}
    for c in scored:
        gid = c.partition_group_id if c.partition_group_id is not None else c.clip_id
        group_counts[gid] = group_counts.get(gid, 0) + 1
    weights = np.array([
        1.0 / group_counts[c.partition_group_id if c.partition_group_id is not None else c.clip_id]
        for c in scored
    ])
    n_groups = len(group_counts)
    max_group_size = max(group_counts.values())
    labels = [c.label for c in scored]
    scores = [c.raw_score for c in scored]
    a, b = fit_platt(scores, labels, sample_weights=weights)
    if a <= 0.0:
        raise EvalError(
            f"calibrate: fitted Platt slope A={a:.4g} is not positive; the raw scores do not "
            "increase with spoofiness (reversed polarity or no signal). Fix the runner's score "
            "polarity (higher = more synthetic) before calibrating."
        )
    probs = apply_platt(scores, a, b)
    cohort_hash = hashlib.sha256(
        "\x00".join(sorted(c.clip_id for c in scored)).encode()
    ).hexdigest()
    policy: dict[str, Any] = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "kind": "synthdetect_calibration_policy",
        "calibration_policy_id": policy_id,
        "inference_space": journal.header["inference_space"],
        "model_id": journal.header["model_id"],
        "method": "platt",
        "platt": {"A": a, "B": b},
        "primary_threshold": operating_point(labels, scores, PRIMARY_FPR),
        "diagnostic_threshold": operating_point(labels, scores, DIAGNOSTIC_FPR),
        "brier": brier_score(probs, labels, sample_weights=weights),
        "reliability": reliability_curve(probs, labels),
        "cohort_sha256": cohort_hash,
        "n_calibration": len(scored),
        "n_partition_groups": n_groups,
        "max_group_size": max_group_size,
        "group_weighting": "partition_group_inverse_size",
    }
    if exclude_strata:
        policy["excluded_strata"] = list(exclude_strata)
        policy["n_excluded"] = n_excluded
    return policy


# --------------------------------------------------------------------------- #
# `report` command (metrics JSON -> house-style Markdown)
# --------------------------------------------------------------------------- #
def _pct(fraction: float) -> str:
    return f"{fraction * 100:.2f}%"


def render_report(metrics: dict[str, Any], *, date: str) -> str:
    """Render a score-metrics object into house-style Markdown (no emdashes).

    The model's ``license_class`` is printed beside every number so a
    non-commercial or unlicensed result can never be mistaken for a shippable
    one. Pure function: metrics JSON in, Markdown out.
    """
    model_id = metrics.get("model_id", "unknown")
    entry = MODELS.get(model_id)
    license_class = entry.license_class if entry else "unknown"
    overall = metrics["overall"]
    ci = overall["eer_ci"]
    lines = [
        f"# Synthdetect eval report ({date})",
        "",
        f"Model: `{model_id}` (license class: **{license_class}**)  ",
        f"Inference space: `{metrics.get('inference_space', 'unknown')}`  ",
        f"Split: `{metrics.get('split')}`",
        "",
        "## Overall",
        "",
        f"- Clips scored: {overall['n']} "
        f"({overall['n_spoof']} spoof, {overall['n_bona_fide']} bona fide)",
        f"- Skip rate: {_pct(metrics['skip_rate'])} ({metrics['n_skipped']} skipped)",
        f"- EER: {_pct(overall['eer'])} "
        f"(95% CI {_pct(ci['lo'])} to {_pct(ci['hi'])}, {int(ci['n_valid'])} valid resamples)",
        f"- AUC: {overall['auc']:.4f}",
        "",
        "### Operating points",
        "",
        "| Target FPR | Threshold | Realized FPR | Realized FNR |",
        "|---|---|---|---|",
    ]
    for op in overall["operating_points"]:
        lines.append(
            f"| {_pct(op['target_fpr'])} | {op['threshold']:.4f} | "
            f"{_pct(op['realized_fpr'])} | {_pct(op['realized_fnr'])} |"
        )
    lines += ["", "## Per stratum", "", "| Stratum | Clips | EER |", "|---|---|---|"]
    for stratum, block in sorted(metrics.get("per_stratum", {}).items()):
        if block.get("single_class"):
            lines.append(f"| `{stratum}` | {block['n']} | single-class (not scored) |")
        else:
            lines.append(f"| `{stratum}` | {block['n']} | {_pct(block['eer'])} |")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _read_journal(path: str) -> Journal:
    return parse_journal(Path(path).read_text(encoding="utf-8"))


def _bind_manifest(journal: Journal, manifest_path: str) -> Manifest:
    """Load the manifest and verify it is the one the journal was produced against.

    The journal header carries the sha256 of the manifest bytes the runner
    scored; here it is recomputed from the file actually supplied and any
    mismatch is fatal. Without this the provenance rail is decorative: a journal
    could be joined to a different manifest sharing clip ids, silently swapping
    labels/strata/splits (a 3-of-3 review finding).
    """
    data = Path(manifest_path).read_bytes()
    manifest = load_manifest(json.loads(data.decode("utf-8")))
    actual = hashlib.sha256(data).hexdigest()
    if journal.header["manifest_sha256"] != actual:
        raise EvalError(
            f"journal was produced against a different manifest: header manifest_sha256 "
            f"{journal.header['manifest_sha256']} != sha256 of {manifest_path} ({actual})"
        )
    return manifest


def _write(out: str | None, payload: Any) -> None:
    text = (
        payload
        if isinstance(payload, str)
        # allow_nan=False fails closed on a non-finite metric rather than emitting
        # the JSON-invalid Infinity/NaN token (matches the repo's hashing idiom).
        else json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    if out:
        Path(out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def cmd_score(args: argparse.Namespace) -> int:
    journal = _read_journal(args.journal)
    manifest = _bind_manifest(journal, args.manifest)
    report = score_report(journal, manifest, split=args.split)
    _write(args.out, report)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    report = compare_journals(
        _read_journal(args.left), _read_journal(args.right),
        decision_threshold=args.decision_threshold,
    )
    _write(args.out, report)
    return 0


def cmd_verdict_windowing(args: argparse.Namespace) -> int:
    upstream = _read_journal(args.upstream)
    production = _read_journal(args.production)
    manifest = None
    if args.manifest:
        manifest = _bind_manifest(upstream, args.manifest)
    report = compare_windowing(upstream, production, manifest=manifest)
    _write(args.out, report)
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    journal = _read_journal(args.journal)
    manifest = _bind_manifest(journal, args.manifest)
    exclude = tuple(args.exclude_strata) if args.exclude_strata else ()
    policy = calibrate_policy(
        journal, manifest, policy_id=args.policy_id, exclude_strata=exclude,
    )
    _write(args.out, policy)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    _write(args.out, render_report(metrics, date=args.date))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="synthdetect host-side scorer (#144)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score", help="EER + CI + operating points for one split")
    p_score.add_argument("--journal", required=True)
    p_score.add_argument("--manifest", required=True)
    p_score.add_argument("--split", default=None, help="restrict to a manifest split")
    p_score.add_argument("--out", default=None)
    p_score.set_defaults(func=cmd_score)

    p_compare = sub.add_parser("compare", help="paired per-clip diff of two journals")
    p_compare.add_argument("--left", required=True)
    p_compare.add_argument("--right", required=True)
    p_compare.add_argument("--decision-threshold", type=float, default=0.0)
    p_compare.add_argument("--out", default=None)
    p_compare.set_defaults(func=cmd_compare)

    p_vw = sub.add_parser(
        "verdict-windowing",
        help="compare upstream vs production windowing FPR on bona fide clips",
    )
    p_vw.add_argument("--upstream", required=True, help="journal scored with --windowing upstream")
    p_vw.add_argument(
        "--production", required=True, help="journal scored with --windowing production",
    )
    p_vw.add_argument("--manifest", default=None, help="corpus manifest for per-stratum breakdown")
    p_vw.add_argument("--out", default=None)
    p_vw.set_defaults(func=cmd_verdict_windowing)

    p_cal = sub.add_parser("calibrate", help="fit a Platt policy on the calibration split")
    p_cal.add_argument("--journal", required=True)
    p_cal.add_argument("--manifest", required=True)
    p_cal.add_argument("--policy-id", required=True)
    p_cal.add_argument(
        "--exclude-strata", nargs="+", default=None,
        help="exclude clips whose stratum contains any of these substrings (case-insensitive)",
    )
    p_cal.add_argument("--out", default=None)
    p_cal.set_defaults(func=cmd_calibrate)

    p_report = sub.add_parser("report", help="render a score-metrics JSON to Markdown")
    p_report.add_argument("--metrics", required=True)
    p_report.add_argument("--date", required=True)
    p_report.add_argument("--out", default=None)
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (EvalError, OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
