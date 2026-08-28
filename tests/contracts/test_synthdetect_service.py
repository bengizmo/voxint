"""Contract tests for the synthdetect service schemas and provenance.

Torch-free by design: imports only the pydantic schemas from the service, so
these run in CI without GPU dependencies. The schemas are the stable contract
between the service and the plugin client. The provenance tests verify that
weight shas are consistent across scoring.py, Dockerfile, and provenance.json.
"""

import importlib
import json
import re
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

SERVICE_ROOT = Path(__file__).resolve().parents[2] / "services" / "synthdetect"

# Import service schemas via importlib so CI does not need the service's
# full dependency tree (torch, fairseq). Only pydantic + resource_models
# are needed, and those are self-contained.


@pytest.fixture(scope="module")
def schemas():
    app_path = str(SERVICE_ROOT)
    if app_path not in sys.path:
        sys.path.insert(0, app_path)
    try:
        # Invalidate any stale cache.
        for mod_name in list(sys.modules):
            if mod_name.startswith("app."):
                del sys.modules[mod_name]
        return importlib.import_module("app.schemas")
    finally:
        if app_path in sys.path:
            sys.path.remove(app_path)


class TestConstants:
    def test_inference_space_nonempty(self, schemas):
        assert isinstance(schemas.INFERENCE_SPACE, str)
        assert len(schemas.INFERENCE_SPACE) > 0

    def test_model_window_samples(self, schemas):
        assert schemas.MODEL_WINDOW_SAMPLES == 64_600

    def test_sample_rate(self, schemas):
        assert schemas.SAMPLE_RATE == 16_000

    def test_min_scorable_samples(self, schemas):
        assert schemas.MIN_SCORABLE_SAMPLES == 8_000

    def test_window_seconds_matches(self, schemas):
        expected = schemas.MODEL_WINDOW_SAMPLES / schemas.SAMPLE_RATE
        assert abs(expected - 4.0375) < 1e-9


class TestScoreRequest:
    def test_valid_request(self, schemas):
        req = schemas.ScoreRequest(
            path="items/abc/audio.wav",
            intervals=[schemas.Interval(start_seconds=0.0, end_seconds=4.0)],
        )
        assert req.path == "items/abc/audio.wav"
        assert len(req.intervals) == 1

    def test_rejects_empty_intervals(self, schemas):
        with pytest.raises((ValueError, ValidationError)):
            schemas.ScoreRequest(path="x.wav", intervals=[])

    def test_rejects_extra_fields(self, schemas):
        with pytest.raises((ValueError, ValidationError)):
            schemas.ScoreRequest(path="x.wav", intervals=[], extra="bad")

    def test_rejects_nan_bounds(self, schemas):
        with pytest.raises((ValueError, ValidationError)):
            schemas.Interval(start_seconds=float("nan"), end_seconds=1.0)

    def test_rejects_inf_bounds(self, schemas):
        with pytest.raises((ValueError, ValidationError)):
            schemas.Interval(start_seconds=0.0, end_seconds=float("inf"))

    def test_rejects_inverted_bounds(self, schemas):
        with pytest.raises((ValueError, ValidationError)):
            schemas.Interval(start_seconds=5.0, end_seconds=2.0)

    def test_rejects_zero_duration(self, schemas):
        with pytest.raises((ValueError, ValidationError)):
            schemas.Interval(start_seconds=1.0, end_seconds=1.0)

    def test_rejects_negative_start(self, schemas):
        with pytest.raises((ValueError, ValidationError)):
            schemas.Interval(start_seconds=-1.0, end_seconds=1.0)


class TestIntervalResult:
    def test_scored_result(self, schemas):
        r = schemas.IntervalResult(raw_score=2.5, window_count=3, skip_reason=None)
        assert r.raw_score == 2.5
        assert r.window_count == 3
        assert r.skip_reason is None

    def test_skipped_result(self, schemas):
        r = schemas.IntervalResult(raw_score=None, window_count=0, skip_reason="too_short")
        assert r.raw_score is None
        assert r.skip_reason == "too_short"

    def test_rejects_both_score_and_skip(self, schemas):
        with pytest.raises((ValueError, ValidationError)):
            schemas.IntervalResult(raw_score=1.0, window_count=1, skip_reason="too_short")

    def test_rejects_neither_score_nor_skip(self, schemas):
        with pytest.raises((ValueError, ValidationError)):
            schemas.IntervalResult(raw_score=None, window_count=0, skip_reason=None)

    def test_rejects_nan_score(self, schemas):
        with pytest.raises((ValueError, ValidationError)):
            schemas.IntervalResult(raw_score=float("nan"), window_count=1)

    def test_rejects_inf_score(self, schemas):
        with pytest.raises((ValueError, ValidationError)):
            schemas.IntervalResult(raw_score=float("inf"), window_count=1)

    def test_rejects_zero_windows_with_score(self, schemas):
        with pytest.raises((ValueError, ValidationError)):
            schemas.IntervalResult(raw_score=1.0, window_count=0)

    def test_rejects_nonzero_windows_with_skip(self, schemas):
        with pytest.raises((ValueError, ValidationError)):
            schemas.IntervalResult(raw_score=None, window_count=1, skip_reason="too_short")

    def test_rejects_unknown_skip_reason(self, schemas):
        with pytest.raises((ValueError, ValidationError)):
            schemas.IntervalResult(raw_score=None, window_count=0, skip_reason="unknown_reason")


class TestScoreResponse:
    def test_response_carries_inference_space(self, schemas):
        r = schemas.ScoreResponse(
            results=[
                schemas.IntervalResult(raw_score=1.0, window_count=1),
            ]
        )
        assert r.inference_space == schemas.INFERENCE_SPACE


class TestHealthResponse:
    def test_degraded_state(self, schemas):
        h = schemas.HealthResponse(
            status="degraded",
            version="1.0.0",
            model=None,
            device="cuda:0",
            engine="fairseq-aasist",
            model_loaded=False,
        )
        assert h.status == "degraded"
        assert h.model is None
        assert h.model_window_samples == 64_600

    def test_ok_state(self, schemas):
        h = schemas.HealthResponse(
            status="ok",
            version="1.0.0",
            model="w2v2-aasist",
            device="cuda:0",
            engine="fairseq-aasist",
            engine_version="fairseq-0.12.2",
            model_loaded=True,
        )
        assert h.status == "ok"
        assert h.inference_space == schemas.INFERENCE_SPACE


class TestProvenanceParity:
    """Verify weight shas are consistent across scoring.py, Dockerfile, and provenance.json."""

    @pytest.fixture(scope="class")
    @classmethod
    def scoring_shas(cls):
        scoring_path = SERVICE_ROOT / "app" / "scoring.py"
        text = scoring_path.read_text()
        aasist = re.search(
            r'SYNTHDETECT_AASIST_SHA.*?["\']([0-9a-f]{64})["\']', text, re.DOTALL
        )
        xlsr = re.search(
            r'SYNTHDETECT_XLSR_SHA.*?["\']([0-9a-f]{64})["\']', text, re.DOTALL
        )
        return {
            "aasist": aasist.group(1) if aasist else None,
            "xlsr": xlsr.group(1) if xlsr else None,
        }

    @pytest.fixture(scope="class")
    @classmethod
    def dockerfile_shas(cls):
        dockerfile_path = SERVICE_ROOT / "Dockerfile"
        text = dockerfile_path.read_text()
        aasist = re.search(r'ARG\s+AASIST_SHA="([0-9a-f]{64})"', text)
        xlsr = re.search(r'ARG\s+XLSR_SHA="([0-9a-f]{64})"', text)
        return {
            "aasist": aasist.group(1) if aasist else None,
            "xlsr": xlsr.group(1) if xlsr else None,
        }

    @pytest.fixture(scope="class")
    @classmethod
    def provenance(cls):
        prov_path = SERVICE_ROOT / "provenance.json"
        return json.loads(prov_path.read_text())

    def test_scoring_shas_nonempty(self, scoring_shas):
        assert scoring_shas["aasist"], "AASIST sha missing from scoring.py"
        assert scoring_shas["xlsr"], "XLS-R sha missing from scoring.py"

    def test_dockerfile_shas_nonempty(self, dockerfile_shas):
        assert dockerfile_shas["aasist"], "AASIST sha missing from Dockerfile"
        assert dockerfile_shas["xlsr"], "XLS-R sha missing from Dockerfile"

    def test_aasist_sha_consistent(self, scoring_shas, dockerfile_shas, provenance):
        prov_sha = provenance["weights"]["aasist_checkpoint"]["sha256"]
        assert scoring_shas["aasist"] == prov_sha, (
            f"AASIST sha mismatch: scoring.py={scoring_shas['aasist'][:16]}... "
            f"vs provenance={prov_sha[:16]}..."
        )
        assert dockerfile_shas["aasist"] == prov_sha, (
            f"AASIST sha mismatch: Dockerfile={dockerfile_shas['aasist'][:16]}... "
            f"vs provenance={prov_sha[:16]}..."
        )

    def test_xlsr_sha_consistent(self, scoring_shas, dockerfile_shas, provenance):
        prov_sha = provenance["weights"]["xlsr_ssl_base"]["sha256"]
        assert scoring_shas["xlsr"] == prov_sha, (
            f"XLS-R sha mismatch: scoring.py={scoring_shas['xlsr'][:16]}... "
            f"vs provenance={prov_sha[:16]}..."
        )
        assert dockerfile_shas["xlsr"] == prov_sha, (
            f"XLS-R sha mismatch: Dockerfile={dockerfile_shas['xlsr'][:16]}... "
            f"vs provenance={prov_sha[:16]}..."
        )

    def test_provenance_inference_space_matches_schema(self, provenance):
        app_path = str(SERVICE_ROOT)
        if app_path not in sys.path:
            sys.path.insert(0, app_path)
        try:
            for mod_name in list(sys.modules):
                if mod_name.startswith("app."):
                    del sys.modules[mod_name]
            schemas = importlib.import_module("app.schemas")
            assert provenance["inference_space"] == schemas.INFERENCE_SPACE
        finally:
            if app_path in sys.path:
                sys.path.remove(app_path)

    def test_provenance_windowing_matches_schema(self, provenance):
        w = provenance["windowing"]
        assert w["model_window_samples"] == 64_600
        assert w["sample_rate"] == 16_000
        assert w["min_scorable_samples"] == 8_000

    def test_vendored_model_sha_present(self, provenance):
        vm = provenance["vendored_model"]
        assert len(vm["sha256"]) == 64
        assert vm["upstream_commit"] == "4acaa61dcef5f7610f43aa4d0b29c4559b970cd2"

    def test_aasist_checkpoint_has_reconstitution_info(self, provenance):
        ckpt = provenance["weights"]["aasist_checkpoint"]
        assert ckpt["total_keys"] == 674
        assert ckpt["ssl_keys"] == 429
        assert ckpt["backend_keys"] == 245
        assert ckpt["state_dict_key_prefix"] is None
