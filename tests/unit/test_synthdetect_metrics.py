"""Metric-math tests for the synthdetect scorer (issue #144).

Covers the pure helpers directly on arrays (EER crossing interpolation, Platt
scaling, Brier, reliability, rank correlation) plus the sklearn-backed EER / AUC
/ operating-point / bootstrap path. The pure helpers need no extra; the
sklearn-backed tests import scikit-learn (present in CI via the
``synthdetect-eval`` extra) and skip cleanly if it is absent.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import synthdetect_corpus as sc  # noqa: E402
import synthdetect_eval as se  # noqa: E402

_sklearn = pytest.importorskip("sklearn", reason="synthdetect-eval extra not installed")


# --------------------------------------------------------------------------- #
# eer_from_roc (pure)
# --------------------------------------------------------------------------- #
def test_eer_crossing_interpolated() -> None:
    fpr = np.array([0.0, 0.0, 0.5, 1.0])
    fnr = np.array([1.0, 0.5, 0.0, 0.0])
    eer, _ = se.eer_from_roc(fpr, fnr)
    assert eer == pytest.approx(0.25)


def test_eer_threshold_interpolated() -> None:
    fpr = np.array([0.0, 0.5])
    fnr = np.array([0.5, 0.0])
    thr = np.array([1.0, 0.0])
    eer, threshold = se.eer_from_roc(fpr, fnr, thr)
    assert eer == pytest.approx(0.25)
    assert threshold == pytest.approx(0.5)


def test_eer_first_point_already_crosses() -> None:
    fpr = np.array([0.3, 0.6])
    fnr = np.array([0.2, 0.1])
    eer, _ = se.eer_from_roc(fpr, fnr)
    assert eer == pytest.approx(0.25)


def test_eer_never_crosses_falls_back() -> None:
    fpr = np.array([0.0, 0.1])
    fnr = np.array([0.9, 0.8])
    eer, _ = se.eer_from_roc(fpr, fnr)
    assert eer == pytest.approx(0.8)


def test_eer_mismatched_arrays_raise() -> None:
    with pytest.raises(se.EvalError, match="equal-length"):
        se.eer_from_roc(np.array([0.0, 1.0]), np.array([1.0]))


def test_eer_inf_sentinel_threshold_falls_back() -> None:
    # sklearn's roc_curve prepends an inf threshold at the FPR=0 point. When the
    # crossing borders it, the interpolated threshold must fall back to the finite
    # bound (never NaN/inf). This is the direct regression for the fixed bug.
    fpr = np.array([0.0, 1.0, 1.0])
    fnr = np.array([1.0, 1.0, 0.0])
    thr = np.array([np.inf, 1.0, 0.0])
    eer, threshold = se.eer_from_roc(fpr, fnr, thr)
    assert eer == pytest.approx(1.0)
    assert threshold == pytest.approx(1.0)
    assert math.isfinite(threshold)


def test_compute_eer_reversed_polarity_finite_threshold() -> None:
    # A reversed pair (spoof scored below bona fide) exercises the sentinel branch
    # through sklearn's real output; the threshold must stay finite.
    eer, threshold = se.compute_eer([0, 1], [1.0, 0.0])
    assert 0.0 <= eer <= 1.0
    assert math.isfinite(threshold)


# --------------------------------------------------------------------------- #
# compute_eer / compute_auc (sklearn)
# --------------------------------------------------------------------------- #
def test_perfect_separation_eer_zero() -> None:
    labels = [0, 0, 1, 1]
    scores = [-2.0, -1.0, 1.0, 2.0]
    eer, _ = se.compute_eer(labels, scores)
    assert eer == pytest.approx(0.0, abs=1e-9)


def test_perfect_separation_auc_one() -> None:
    assert se.compute_auc([0, 0, 1, 1], [-2.0, -1.0, 1.0, 2.0]) == pytest.approx(1.0)


def test_single_class_rejected() -> None:
    with pytest.raises(se.EvalError, match="BOTH classes"):
        se.compute_eer([1, 1, 1], [0.1, 0.2, 0.3])


def test_non_binary_labels_rejected() -> None:
    with pytest.raises(se.EvalError, match="0/1"):
        se.compute_eer([0, 2], [0.1, 0.2])


def test_tied_scores_do_not_crash() -> None:
    # Every clip gets the same score: the detector is useless (EER ~ 0.5) but the
    # scorer must still return a finite number in [0, 1], never raise.
    eer, _ = se.compute_eer([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
    assert 0.0 <= eer <= 1.0


# --------------------------------------------------------------------------- #
# operating_point
# --------------------------------------------------------------------------- #
def test_operating_point_respects_fpr_budget() -> None:
    rng = np.random.default_rng(0)
    bona = rng.normal(-1.0, 1.0, 200)
    spoof = rng.normal(2.0, 1.0, 200)
    labels = [0] * 200 + [1] * 200
    scores = list(bona) + list(spoof)
    op = se.operating_point(labels, scores, 0.05)
    assert op["realized_fpr"] <= 0.05 + 1e-9
    assert op["target_fpr"] == 0.05
    assert math.isfinite(op["threshold"])


def test_operating_point_tight_budget_finite_threshold() -> None:
    # When the top-scored clip is bona fide, NO finite threshold achieves FPR=0
    # (accepting anything accepts that bona fide clip), so only sklearn's inf
    # sentinel qualifies. The operating point must still be a finite
    # "reject everything" threshold so the metrics stay valid JSON.
    labels = [1, 1, 0]
    scores = [0.0, 1.0, 2.0]  # highest score belongs to a bona fide clip
    op = se.operating_point(labels, scores, 0.0)
    assert math.isfinite(op["threshold"])
    assert op["threshold"] > max(scores)  # reject-everything threshold
    assert op["realized_fpr"] == 0.0


# --------------------------------------------------------------------------- #
# bootstrap_eer_ci
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_is_seeded_and_reproducible() -> None:
    rng = np.random.default_rng(1)
    labels = [0] * 50 + [1] * 50
    scores = list(rng.normal(-1, 1, 50)) + list(rng.normal(1, 1, 50))
    ci1 = se.bootstrap_eer_ci(labels, scores, context="t", resamples=100)
    ci2 = se.bootstrap_eer_ci(labels, scores, context="t", resamples=100)
    assert ci1 == ci2
    assert ci1["lo"] <= ci1["hi"]
    assert ci1["n_valid"] > 0


def test_bootstrap_context_changes_interval() -> None:
    rng = np.random.default_rng(2)
    labels = [0] * 50 + [1] * 50
    scores = list(rng.normal(-1, 1, 50)) + list(rng.normal(1, 1, 50))
    ci_a = se.bootstrap_eer_ci(labels, scores, context="a", resamples=100)
    ci_b = se.bootstrap_eer_ci(labels, scores, context="b", resamples=100)
    # Different seed context => (almost surely) a different interval.
    assert (ci_a["lo"], ci_a["hi"]) != (ci_b["lo"], ci_b["hi"])


# --------------------------------------------------------------------------- #
# Platt scaling / probabilities
# --------------------------------------------------------------------------- #
def test_platt_is_monotonic_and_bounded() -> None:
    scores = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    a, b = se.fit_platt(scores, labels)
    probs = se.apply_platt(scores, a, b)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.all(np.diff(probs) > 0.0)  # higher raw score => higher spoof prob
    assert a > 0.0


def test_platt_needs_both_classes() -> None:
    with pytest.raises(se.EvalError, match="both bona_fide and spoof"):
        se.fit_platt(np.array([1.0, 2.0]), np.array([1, 1]))


def test_platt_separable_does_not_diverge() -> None:
    # Perfectly separable data would drive an unregularized fit to infinity;
    # target smoothing keeps A finite.
    scores = np.array([-10.0, -9.0, 9.0, 10.0])
    labels = np.array([0, 0, 1, 1])
    a, b = se.fit_platt(scores, labels)
    assert np.isfinite(a) and np.isfinite(b)


def test_brier_perfect_is_zero() -> None:
    assert se.brier_score([1.0, 0.0, 1.0], [1, 0, 1]) == pytest.approx(0.0)


def test_brier_mismatched_raises() -> None:
    with pytest.raises(se.EvalError):
        se.brier_score([0.5], [1, 0])


def test_reliability_curve_bins() -> None:
    probs = [0.05, 0.15, 0.95, 0.85]
    labels = [0, 0, 1, 1]
    curve = se.reliability_curve(probs, labels, n_bins=10)
    # Empty bins omitted; every kept bin reports a count and empirical rate.
    assert curve
    assert all(b["count"] >= 1 for b in curve)
    assert all(0.0 <= b["empirical_rate"] <= 1.0 for b in curve)


def test_reliability_curve_bad_bins() -> None:
    with pytest.raises(se.EvalError, match="n_bins"):
        se.reliability_curve([0.5], [1], n_bins=0)


# --------------------------------------------------------------------------- #
# Spearman / rankdata (pure)
# --------------------------------------------------------------------------- #
def test_spearman_perfect_monotonic() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([10.0, 20.0, 30.0, 40.0])
    assert se._spearman(a, b) == pytest.approx(1.0)


def test_spearman_perfect_inverse() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([40.0, 30.0, 20.0, 10.0])
    assert se._spearman(a, b) == pytest.approx(-1.0)


def test_spearman_constant_is_zero() -> None:
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([1.0, 2.0, 3.0])
    assert se._spearman(a, b) == 0.0


def test_rankdata_averages_ties() -> None:
    ranks = se._rankdata(np.array([10.0, 10.0, 30.0]))
    assert ranks[0] == pytest.approx(1.5)
    assert ranks[1] == pytest.approx(1.5)
    assert ranks[2] == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# calibrate_policy: exclude_strata
# --------------------------------------------------------------------------- #
def _make_calibration_fixtures(
    bf_scores: list[float],
    spoof_a_scores: list[float],
    spoof_b_scores: list[float],
    stratum_a: str = "spoof|tts|piper|meetingroom",
    stratum_b: str = "spoof|tts|chatterbox|meetingroom",
) -> tuple[se.Journal, sc.Manifest]:
    """Build minimal Journal + Manifest for calibrate_policy tests."""

    clips: list[sc.ClipEntry] = []
    results: list[se.ClipScore] = []

    for i, s in enumerate(bf_scores):
        cid = f"bf-{i:04d}"
        clips.append(sc.ClipEntry(
            clip_id=cid, rel_path=f"bf/{cid}.wav", sha256="a" * 64,
            duration_s=3.0, label="bona_fide", language="en",
            license_spdx="CC-BY-4.0", stratum="organic-bonafide",
            source="test", speaker_id=f"spk-{i}", split="calibration",
            generator=None, degradation=None, parent_clip_id=None, acquire=None,
        ))
        results.append(se.ClipScore(clip_id=cid, raw_score=s, skip_reason=None, n_windows=1))

    gen = sc.GeneratorProvenance(
        name="test", version="1.0", checkpoint_sha=None,
        voice="default", seed=None, text_source="test",
    )
    for i, s in enumerate(spoof_a_scores):
        cid = f"spoof-a-{i:04d}"
        clips.append(sc.ClipEntry(
            clip_id=cid, rel_path=f"spoof_a/{cid}.wav", sha256="b" * 64,
            duration_s=3.0, label="spoof", language="en",
            license_spdx="CC-BY-4.0", stratum=stratum_a,
            source="test", speaker_id=f"spk-{i}", split="calibration",
            generator=gen, degradation=None, parent_clip_id=None, acquire=None,
        ))
        results.append(se.ClipScore(clip_id=cid, raw_score=s, skip_reason=None, n_windows=1))

    for i, s in enumerate(spoof_b_scores):
        cid = f"spoof-b-{i:04d}"
        clips.append(sc.ClipEntry(
            clip_id=cid, rel_path=f"spoof_b/{cid}.wav", sha256="c" * 64,
            duration_s=3.0, label="spoof", language="en",
            license_spdx="CC-BY-4.0", stratum=stratum_b,
            source="test", speaker_id=f"spk-{i}", split="calibration",
            generator=gen, degradation=None, parent_clip_id=None, acquire=None,
        ))
        results.append(se.ClipScore(clip_id=cid, raw_score=s, skip_reason=None, n_windows=1))

    manifest = sc.Manifest(schema_version=3, clips=tuple(clips), corpus_kind="composite")
    header = {
        "kind": "synthdetect_journal", "schema_version": 1,
        "inference_space": "synthdetect-w2v2-aasist-v1", "model_id": "w2v2-aasist",
        "manifest_sha256": "a" * 64,
        "windowing": {"pooling": "mean"},
    }
    journal = se.Journal(header=header, results=tuple(results))
    return journal, manifest


def test_calibrate_exclude_strata_reduces_population() -> None:
    rng = np.random.default_rng(42)
    bf = rng.normal(0.0, 1.0, 50).tolist()
    piper = rng.normal(4.0, 1.0, 30).tolist()
    chatterbox = rng.normal(0.5, 1.0, 30).tolist()
    journal, manifest = _make_calibration_fixtures(bf, piper, chatterbox)

    full = se.calibrate_policy(journal, manifest, policy_id="test-full")
    assert full["n_calibration"] == 110
    assert "excluded_strata" not in full

    excl = se.calibrate_policy(
        journal, manifest, policy_id="test-excl",
        exclude_strata=("chatterbox",),
    )
    assert excl["n_calibration"] == 80
    assert excl["excluded_strata"] == ["chatterbox"]
    assert excl["n_excluded"] == 30
    assert excl["cohort_sha256"] != full["cohort_sha256"]


def test_calibrate_exclude_strata_steepens_platt() -> None:
    rng = np.random.default_rng(42)
    bf = rng.normal(0.0, 1.0, 100).tolist()
    piper = rng.normal(5.0, 1.0, 60).tolist()
    chatterbox = rng.normal(0.3, 1.0, 60).tolist()
    journal, manifest = _make_calibration_fixtures(bf, piper, chatterbox)

    full = se.calibrate_policy(journal, manifest, policy_id="full")
    excl = se.calibrate_policy(
        journal, manifest, policy_id="excl", exclude_strata=("chatterbox",),
    )
    assert excl["platt"]["A"] > full["platt"]["A"]
    assert excl["brier"] < full["brier"]


def test_calibrate_exclude_all_raises() -> None:
    rng = np.random.default_rng(42)
    bf = rng.normal(0.0, 1.0, 20).tolist()
    piper = rng.normal(4.0, 1.0, 20).tolist()
    journal, manifest = _make_calibration_fixtures(bf, piper, [], stratum_a="spoof|x")

    with pytest.raises(se.EvalError, match="all clips excluded"):
        se.calibrate_policy(
            journal, manifest, policy_id="bad",
            exclude_strata=("spoof", "bonafide"),
        )


# --------------------------------------------------------------------------- #
# calibrate_policy: partition-group weighting
# --------------------------------------------------------------------------- #
def _make_grouped_fixtures(
    n_bf: int,
    n_spoof_per_bf: int,
    bf_mean: float = 0.0,
    spoof_mean: float = 4.0,
    seed: int = 42,
) -> tuple[se.Journal, sc.Manifest]:
    """Build fixtures where each bf clip has n_spoof_per_bf paired spoofs."""
    rng = np.random.default_rng(seed)
    clips: list[sc.ClipEntry] = []
    results: list[se.ClipScore] = []
    gen = sc.GeneratorProvenance(
        name="test", version="1.0", checkpoint_sha=None,
        voice="default", seed=None, text_source="test",
    )
    for i in range(n_bf):
        bf_cid = f"bf-{i:04d}"
        clips.append(sc.ClipEntry(
            clip_id=bf_cid, rel_path=f"bf/{bf_cid}.wav", sha256="a" * 64,
            duration_s=3.0, label="bona_fide", language="en",
            license_spdx="CC-BY-4.0", stratum="organic-bonafide",
            source="test", speaker_id=f"spk-{i}", split="calibration",
            generator=None, degradation=None, parent_clip_id=None, acquire=None,
            partition_group_id=bf_cid,
        ))
        results.append(se.ClipScore(
            clip_id=bf_cid, raw_score=rng.normal(bf_mean, 1.0),
            skip_reason=None, n_windows=1,
        ))
        for j in range(n_spoof_per_bf):
            sp_cid = f"bf-{i:04d}--gen{j}"
            clips.append(sc.ClipEntry(
                clip_id=sp_cid, rel_path=f"spoof/{sp_cid}.wav", sha256="b" * 64,
                duration_s=3.0, label="spoof", language="en",
                license_spdx="CC-BY-4.0", stratum="spoof|tts|test|meetingroom",
                source="test", speaker_id=f"spk-{i}", split="calibration",
                generator=gen, degradation=None, parent_clip_id=None, acquire=None,
                partition_group_id=bf_cid,
            ))
            results.append(se.ClipScore(
                clip_id=sp_cid, raw_score=rng.normal(spoof_mean, 1.0),
                skip_reason=None, n_windows=1,
            ))

    manifest = sc.Manifest(schema_version=3, clips=tuple(clips), corpus_kind="composite")
    header = {
        "kind": "synthdetect_journal", "schema_version": 1,
        "inference_space": "synthdetect-w2v2-aasist-v1", "model_id": "w2v2-aasist",
        "manifest_sha256": "a" * 64,
        "windowing": {"pooling": "mean"},
    }
    return se.Journal(header=header, results=tuple(results)), manifest


def test_calibrate_emits_group_metadata() -> None:
    journal, manifest = _make_grouped_fixtures(30, 1)
    policy = se.calibrate_policy(journal, manifest, policy_id="grouped")
    assert policy["n_partition_groups"] == 30
    assert policy["max_group_size"] == 2
    assert policy["group_weighting"] == "partition_group_inverse_size"


def test_calibrate_singleton_groups_match_uniform() -> None:
    """When all groups are singletons, weighting is a no-op."""
    rng = np.random.default_rng(99)
    bf = rng.normal(0.0, 1.0, 40).tolist()
    spoof = rng.normal(4.0, 1.0, 40).tolist()
    journal, manifest = _make_calibration_fixtures(bf, spoof, [])
    policy = se.calibrate_policy(journal, manifest, policy_id="singleton")
    assert policy["max_group_size"] == 1
    a_singleton, b_singleton = policy["platt"]["A"], policy["platt"]["B"]
    a_direct, b_direct = se.fit_platt(
        bf + spoof, [0] * 40 + [1] * 40,
    )
    assert a_singleton == pytest.approx(a_direct, rel=1e-6)
    assert b_singleton == pytest.approx(b_direct, rel=1e-6)


def test_calibrate_grouped_changes_intercept() -> None:
    """Paired groups (bf + spoof) should produce a different B than singletons."""
    journal_grouped, manifest_grouped = _make_grouped_fixtures(50, 2, seed=7)
    policy_grouped = se.calibrate_policy(
        journal_grouped, manifest_grouped, policy_id="grouped",
    )
    assert policy_grouped["max_group_size"] == 3

    journal_singleton, manifest_singleton = _make_grouped_fixtures(50, 2, seed=7)
    clips_no_group = tuple(
        sc.ClipEntry(
            clip_id=c.clip_id, rel_path=c.rel_path, sha256=c.sha256,
            duration_s=c.duration_s, label=c.label, language=c.language,
            license_spdx=c.license_spdx, stratum=c.stratum, source=c.source,
            speaker_id=c.speaker_id, split=c.split, generator=c.generator,
            degradation=c.degradation, parent_clip_id=c.parent_clip_id,
            acquire=c.acquire, partition_group_id=None,
        )
        for c in manifest_singleton.clips
    )
    manifest_nogroup = sc.Manifest(
        schema_version=3, clips=clips_no_group, corpus_kind="composite",
    )
    policy_singleton = se.calibrate_policy(
        journal_singleton, manifest_nogroup, policy_id="singleton",
    )
    assert policy_singleton["max_group_size"] == 1
    assert policy_grouped["platt"]["B"] != pytest.approx(
        policy_singleton["platt"]["B"], rel=0.01,
    )


def test_fit_platt_uniform_weights_match_no_weights() -> None:
    scores = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    a1, b1 = se.fit_platt(scores, labels)
    a2, b2 = se.fit_platt(scores, labels, sample_weights=np.ones(6))
    assert a1 == pytest.approx(a2, rel=1e-9)
    assert b1 == pytest.approx(b2, rel=1e-9)


def test_fit_platt_weights_bad_length_raises() -> None:
    with pytest.raises(se.EvalError, match="sample_weights"):
        se.fit_platt(np.array([1.0, 2.0]), np.array([0, 1]), sample_weights=np.array([1.0]))


def test_brier_weighted_vs_uniform() -> None:
    probs = [0.9, 0.1, 0.8]
    labels = [1, 0, 1]
    unweighted = se.brier_score(probs, labels)
    weighted = se.brier_score(probs, labels, sample_weights=[1.0, 1.0, 1.0])
    assert unweighted == pytest.approx(weighted)
