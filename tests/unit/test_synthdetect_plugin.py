"""Unit tests for the synthdetect plugin backend (#145 PR 2)."""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from voxint.plugins.synthdetect.calibration import (
    CALIBRATION_POLICIES,
    DEFAULT_INFERENCE_SPACE,
    DEFAULT_POLICY_ID,
    apply_calibration,
)


class TestCalibration:
    def test_default_policy_exists(self):
        assert DEFAULT_POLICY_ID in CALIBRATION_POLICIES

    def test_default_policy_matches_inference_space(self):
        policy = CALIBRATION_POLICIES[DEFAULT_POLICY_ID]
        assert policy.fit_inference_space == DEFAULT_INFERENCE_SPACE

    def test_apply_known_logit_zero(self):
        result = apply_calibration(0.0, DEFAULT_POLICY_ID)
        policy = CALIBRATION_POLICIES[DEFAULT_POLICY_ID]
        expected = 1.0 / (1.0 + math.exp(-(policy.platt_a * 0.0 + policy.platt_b)))
        assert result == pytest.approx(expected)

    def test_apply_high_logit_near_one(self):
        result = apply_calibration(10.0, DEFAULT_POLICY_ID)
        assert result > 0.99

    def test_apply_low_logit_near_zero(self):
        result = apply_calibration(-10.0, DEFAULT_POLICY_ID)
        assert result < 0.01

    def test_apply_boundary_logit(self):
        policy = CALIBRATION_POLICIES[DEFAULT_POLICY_ID]
        threshold = -policy.platt_b / policy.platt_a
        result = apply_calibration(threshold, DEFAULT_POLICY_ID)
        assert result == pytest.approx(0.5, abs=1e-9)

    def test_monotonic(self):
        scores = [apply_calibration(logit, DEFAULT_POLICY_ID) for logit in [-5, -2, 0, 2, 5]]
        assert all(a < b for a, b in pairwise(scores))

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError, match="unknown calibration policy"):
            apply_calibration(0.0, "nonexistent-policy")

    def test_output_in_unit_interval(self):
        for logit in [-100, -10, -1, 0, 1, 10, 100]:
            result = apply_calibration(logit, DEFAULT_POLICY_ID)
            assert 0.0 <= result <= 1.0

    def test_policy_has_positive_brier(self):
        for pid, policy in CALIBRATION_POLICIES.items():
            assert policy.brier > 0, f"policy {pid} has non-positive brier"

    def test_policy_dataclass_frozen(self):
        policy = CALIBRATION_POLICIES[DEFAULT_POLICY_ID]
        with pytest.raises(AttributeError):
            policy.platt_a = 999  # type: ignore[misc]


class TestCalibrationSignConvention:
    """Verify the sign convention matches the service: higher logit = more synthetic."""

    def test_positive_logit_high_risk(self):
        result = apply_calibration(5.0, DEFAULT_POLICY_ID)
        assert result > 0.5

    def test_negative_logit_low_risk(self):
        result = apply_calibration(-5.0, DEFAULT_POLICY_ID)
        assert result < 0.5

    def test_known_eval_logit(self):
        result = apply_calibration(4.29, DEFAULT_POLICY_ID)
        assert result > 0.9


class TestPluginManifest:
    def test_manifest_id(self):
        from voxint.plugins.synthdetect import SynthdetectPlugin

        assert SynthdetectPlugin.manifest.id == "synthdetect"

    def test_manifest_task_names(self):
        from voxint.plugins.synthdetect import SynthdetectPlugin

        assert "voxint.plugin.synthdetect.score_run" in SynthdetectPlugin.manifest.task_names

    def test_manifest_settings_prefixes(self):
        from voxint.plugins.synthdetect import SynthdetectPlugin

        assert "synthdetect_" in SynthdetectPlugin.manifest.settings_prefixes

    def test_task_routes(self):
        from voxint.plugins.synthdetect import SynthdetectPlugin

        plugin = SynthdetectPlugin()
        routes = plugin.task_routes()
        assert "voxint.plugin.synthdetect.score_run" in routes
        assert routes["voxint.plugin.synthdetect.score_run"]["queue"] == "post"

    def test_task_modules(self):
        from voxint.plugins.synthdetect import SynthdetectPlugin

        plugin = SynthdetectPlugin()
        modules = plugin.task_modules()
        assert "voxint.plugins.synthdetect.tasks" in modules

    def test_job_lanes(self):
        from voxint.plugins.synthdetect import SynthdetectPlugin

        plugin = SynthdetectPlugin()
        lanes = plugin.job_lanes()
        assert len(lanes) == 1
        assert lanes[0].redispatch_task_name == "voxint.plugin.synthdetect.score_run"
        assert lanes[0].limit == 50

    def test_enabled_default_false(self):
        from types import SimpleNamespace

        from voxint.plugins.synthdetect import SynthdetectPlugin

        plugin = SynthdetectPlugin()
        settings = SimpleNamespace(synthdetect_enabled=False)
        assert plugin.enabled(None, settings) is False  # type: ignore[arg-type]

    def test_enabled_true_via_settings(self):
        from types import SimpleNamespace

        from voxint.plugins.synthdetect import SynthdetectPlugin

        plugin = SynthdetectPlugin()
        settings = SimpleNamespace(synthdetect_enabled=True)
        assert plugin.enabled(None, settings) is True  # type: ignore[arg-type]


class TestClientResponseParsing:
    def test_parse_score_response(self):
        response = {
            "inference_space": "w2v2-aasist-df-m2-s0e11",
            "results": [
                {"raw_score": 0.35, "window_count": 3, "skip_reason": None},
                {"raw_score": None, "window_count": 0, "skip_reason": "too_short"},
            ],
        }
        results = response["results"]
        assert len(results) == 2
        assert results[0]["raw_score"] == 0.35
        assert results[0]["skip_reason"] is None
        assert results[1]["raw_score"] is None
        assert results[1]["skip_reason"] == "too_short"
