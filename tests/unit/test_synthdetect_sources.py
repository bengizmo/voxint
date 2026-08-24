"""Registry-integrity tests for the synthdetect sources pins (issue #144).

Freezes the two rails the registry exists to hold: an ``unlicensed`` model
(Nes2Net, no license file) is never runnable, and CANDIDATE shas mean no model
is claimed serve-ready in S1. Also asserts the structural invariants the import
time validator enforces, so a bad pin fails a test rather than silently shipping.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import synthdetect_sources as src  # noqa: E402


def test_registry_imports_clean() -> None:
    # _validate_registry() runs at import; reaching here means it passed.
    assert set(src.MODELS) >= {"w2v2-aasist", "antideepfake-xlsr-2b", "audioseal", "nes2net"}


def test_exactly_one_default_and_it_is_shippable() -> None:
    default = src.default_model()
    assert default.model_id == "w2v2-aasist"
    assert default.default is True
    assert default.license_class == "shippable"


def test_unlicensed_model_refuses_to_run() -> None:
    nes2net = src.get_model("nes2net")
    assert nes2net.license_class == "unlicensed"
    assert nes2net.enabled() is False
    assert src.runnable(nes2net) is False
    with pytest.raises(src.SourcesError, match="refuses to run"):
        src.assert_runnable(nes2net)


def test_licensed_models_are_runnable() -> None:
    for model_id in ("w2v2-aasist", "antideepfake-xlsr-2b", "audioseal"):
        model = src.get_model(model_id)
        assert src.runnable(model) is True
        src.assert_runnable(model)  # does not raise


def test_shas_are_candidate_in_s1() -> None:
    # No model is weights-pinned yet; every weight sha is CANDIDATE (None).
    for model in src.MODELS.values():
        assert model.weights_pinned() is False
        assert all(w.sha256 is None for w in model.weights)
        assert model.commit is None


def test_noncommercial_model_flagged() -> None:
    nii = src.get_model("antideepfake-xlsr-2b")
    assert nii.license_class == "noncommercial"
    assert nii.weights_license_spdx == "CC-BY-NC-SA-4.0"


def test_audioseal_is_harness_only() -> None:
    audioseal = src.get_model("audioseal")
    assert audioseal.harness_only is True
    assert audioseal.license_class == "shippable"


def test_reproduction_targets_reference_known_benchmarks() -> None:
    for model in src.MODELS.values():
        for target in model.reproduction_targets:
            assert target.benchmark in src.BENCHMARKS
            assert target.tolerance_status in ("provisional", "ratified")
            assert target.tolerance_pp > 0


def test_get_model_unknown_raises() -> None:
    with pytest.raises(src.SourcesError, match="unknown model"):
        src.get_model("does-not-exist")


def test_license_classes_all_valid() -> None:
    for model in src.MODELS.values():
        assert model.license_class in src.LICENSE_CLASSES


def test_benchmark_keys_match_ids() -> None:
    for key, bench in src.BENCHMARKS.items():
        assert key == bench.dataset_id
        assert bench.license_status in ("verified", "unverified")


def test_windowing_pooling_is_logit_mean() -> None:
    # The pooling rule is pinned inside the inference space; raw-mean vs
    # logit-mean vs max change the decision surface, so it must not drift.
    for model in src.MODELS.values():
        assert model.windowing.pooling == "logit-mean"
        assert model.windowing.sample_rate_hz == 16000
