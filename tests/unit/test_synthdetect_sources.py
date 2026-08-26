"""Registry-integrity tests for the synthdetect sources pins (issue #144).

Freezes the two rails the registry exists to hold: an ``unlicensed`` model
(Nes2Net, no license file) is never runnable, and CANDIDATE shas mean no model
is claimed serve-ready in S1. Also asserts the structural invariants the import
time validator enforces, so a bad pin fails a test rather than silently shipping.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import synthdetect_sources as src  # noqa: E402


def _registry_with(model_id: str, model: src.ModelEntry) -> dict[str, src.ModelEntry]:
    """A copy of the real registry with one model swapped, for validator tests."""
    swapped = dict(src.MODELS)
    swapped[model_id] = model
    return swapped


def test_registry_imports_clean() -> None:
    # _validate_registry() runs at import; reaching here means it passed.
    assert set(src.MODELS) >= {
        "w2v2-aasist",
        "w2v2-aasist-df",
        "antideepfake-xlsr-2b",
        "audioseal",
        "nes2net",
    }


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
    for model_id in ("w2v2-aasist", "w2v2-aasist-df", "antideepfake-xlsr-2b", "audioseal"):
        model = src.get_model(model_id)
        assert src.runnable(model) is True
        src.assert_runnable(model)  # does not raise


# The models whose weights are frozen from real bytes. S2b froze the default
# w2v2-aasist; S3 (2026-08-25) froze the DF-tuned sibling w2v2-aasist-df's sha
# (PINNED_UNQUALIFIED). Every other model stays CANDIDATE until its own freeze.
_PINNED_MODEL_IDS = {"w2v2-aasist", "w2v2-aasist-df"}


def test_pinned_models_frozen_others_candidate() -> None:
    default = src.default_model()
    assert default.model_id == "w2v2-aasist"
    for model_id in _PINNED_MODEL_IDS:
        model = src.get_model(model_id)
        assert model.weights_pinned() is True, f"{model_id} must be pinned"
        assert all(w.sha256 is not None and w.size_bytes is not None for w in model.weights)
        assert model.commit is not None
    for model in src.MODELS.values():
        if model.model_id in _PINNED_MODEL_IDS:
            continue
        assert model.weights_pinned() is False
        assert all(w.sha256 is None for w in model.weights)
        assert model.commit is None


def test_df_sibling_shares_runtime_but_is_a_distinct_inference_space() -> None:
    # w2v2-aasist-df carries the hard 2.85% DF stop-gate on the DF-tuned
    # checkpoint; the default w2v2-aasist does NOT (its ASVspoof2019-LA checkpoint
    # is not the source of that number). Same code/commit and XLS-R base, a
    # different aasist checkpoint, and therefore a different inference space.
    default = src.get_model("w2v2-aasist")
    df = src.get_model("w2v2-aasist-df")
    assert df.commit == default.commit  # same vendored model.py
    assert df.inference_space != default.inference_space
    df_ckpt = {w.filename for w in df.weights if w.role == "aasist_checkpoint"}
    la_ckpt = {w.filename for w in default.weights if w.role == "aasist_checkpoint"}
    assert df_ckpt == {"Best_LA_model_for_DF.pth"}
    assert la_ckpt == {"LA_model.pth"}
    # The XLS-R base is byte-identical across the two.
    df_base = next(w for w in df.weights if w.role == "xlsr_ssl_base")
    la_base = next(w for w in default.weights if w.role == "xlsr_ssl_base")
    assert df_base.sha256 == la_base.sha256
    # The DF checkpoint was saved from a DataParallel model: it declares the
    # "module." key-prefix strip as data. The default loads verbatim (None), so
    # its already-qualified load path is untouched.
    df_ckpt_w = next(w for w in df.weights if w.role == "aasist_checkpoint")
    la_ckpt_w = next(w for w in default.weights if w.role == "aasist_checkpoint")
    assert df_ckpt_w.state_dict_key_prefix == "module."
    assert la_ckpt_w.state_dict_key_prefix is None
    assert la_base.state_dict_key_prefix is None


def test_df_anchor_is_the_only_asvspoof_df_stop_gate() -> None:
    # The 2.85% ASVspoof 2021 DF anchor is a stop-gate on exactly one model, and
    # it is the DF-tuned checkpoint. No other entry may claim a DF EER stop-gate.
    df_stop_gates = [
        m.model_id
        for m in src.MODELS.values()
        for t in m.reproduction_targets
        if t.benchmark == "asvspoof2021_df"
        and t.metric == "eer"
        and t.gate_role == "stop_gate"
        and src.runnable(m)
    ]
    assert df_stop_gates == ["w2v2-aasist-df"]


def test_default_model_has_no_stop_gate_it_cannot_meet() -> None:
    # The production default's only reproduction number is a DIAGNOSTIC ITW
    # observation, never a stop-gate: it must not carry a hard bar tied to a
    # checkpoint it is not.
    default = src.default_model()
    assert [t.gate_role for t in default.reproduction_targets] == ["diagnostic"]
    assert all(t.benchmark == "itw" for t in default.reproduction_targets)


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
            assert target.gate_role in src.GATE_ROLES


def test_inference_spaces_are_unique() -> None:
    # The parity gate governs the inference-space id; two models sharing one would
    # let a numerically different runtime claim another's qualification.
    spaces = [m.inference_space for m in src.MODELS.values()]
    assert len(spaces) == len(set(spaces))


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


def test_sources_version_pinned() -> None:
    # A registry SHAPE change (a new field/model) must move the version so a
    # reshuffle or pin refresh is a visible, deliberate event in artifacts.
    assert src.SOURCES_VERSION == "synthdetect-sources-v4"


def test_pinned_weight_shas_and_commits_have_real_shape() -> None:
    # A frozen pin is a full sha, not merely non-None: the import rail rejects a
    # truncated or placeholder digest. Assert the real data satisfies it.
    for model_id in _PINNED_MODEL_IDS:
        model = src.get_model(model_id)
        assert model.commit is not None and src._COMMIT_RE.match(model.commit)
        for weight in model.weights:
            assert weight.sha256 is not None and src._SHA256_RE.match(weight.sha256)
            assert weight.size_bytes is not None and weight.size_bytes > 0


def test_weights_pinned_requires_commit() -> None:
    # A model with real shas but a CANDIDATE commit is not serve-ready.
    default = src.get_model("w2v2-aasist")
    assert default.weights_pinned() is True
    no_commit = replace(default, commit=None)
    assert no_commit.weights_pinned() is False


def test_validator_rejects_truncated_weight_sha() -> None:
    default = src.get_model("w2v2-aasist")
    bad_weight = replace(default.weights[0], sha256="deadbeef")
    bad = replace(default, weights=(bad_weight, *default.weights[1:]))
    with pytest.raises(src.SourcesError, match="64-char hex"):
        src._validate_registry(_registry_with("w2v2-aasist", bad))


def test_validator_rejects_pinned_weight_without_size() -> None:
    default = src.get_model("w2v2-aasist")
    bad_weight = replace(default.weights[0], size_bytes=0)
    bad = replace(default, weights=(bad_weight, *default.weights[1:]))
    with pytest.raises(src.SourcesError, match="positive size_bytes"):
        src._validate_registry(_registry_with("w2v2-aasist", bad))


def test_validator_rejects_unknown_state_dict_key_prefix() -> None:
    # The prefix vocabulary is closed: only None (load verbatim) or "module." (the
    # DataParallel unwrap). A new transform must be a deliberate registry change.
    default = src.get_model("w2v2-aasist")
    bad_weight = replace(default.weights[0], state_dict_key_prefix="mod.")
    bad = replace(default, weights=(bad_weight, *default.weights[1:]))
    with pytest.raises(src.SourcesError, match="unknown state_dict_key_prefix"):
        src._validate_registry(_registry_with("w2v2-aasist", bad))
    assert set(src.STATE_DICT_KEY_PREFIXES) == {None, "module."}


def test_validator_rejects_malformed_commit() -> None:
    default = src.get_model("w2v2-aasist")
    bad = replace(default, commit="not-a-sha")
    with pytest.raises(src.SourcesError, match="40-char hex"):
        src._validate_registry(_registry_with("w2v2-aasist", bad))


def test_validator_rejects_unknown_metric_and_tolerance_status() -> None:
    default = src.get_model("w2v2-aasist")
    bad_metric = replace(default, reproduction_targets=(
        replace(default.reproduction_targets[0], metric="bogus"),
    ))
    with pytest.raises(src.SourcesError, match="unknown metric"):
        src._validate_registry(_registry_with("w2v2-aasist", bad_metric))
    bad_tol = replace(default, reproduction_targets=(
        replace(default.reproduction_targets[0], tolerance_status="maybe"),
    ))
    with pytest.raises(src.SourcesError, match="unknown tolerance_status"):
        src._validate_registry(_registry_with("w2v2-aasist", bad_tol))


def test_validator_rejects_default_with_stop_gate() -> None:
    # The default's only target is diagnostic; make it a stop-gate and the rail
    # must fire (a default must never gate on a checkpoint it is not).
    default = src.get_model("w2v2-aasist")
    bad = replace(default, reproduction_targets=(
        replace(default.reproduction_targets[0], gate_role="stop_gate"),
    ))
    with pytest.raises(src.SourcesError, match="must not carry a stop_gate"):
        src._validate_registry(_registry_with("w2v2-aasist", bad))


def test_validator_rejects_a_second_runnable_df_stop_gate() -> None:
    # Nes2Net declares a DF stop-gate but is unlicensed (excluded). License it and
    # a second runnable DF anchor exists -> the rail forces a deliberate fix.
    nes2net = src.get_model("nes2net")
    licensed = replace(nes2net, license_class="shippable")
    with pytest.raises(src.SourcesError, match="more than one runnable model"):
        src._validate_registry(_registry_with("nes2net", licensed))
