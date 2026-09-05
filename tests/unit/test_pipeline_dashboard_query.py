"""Pure-function tests for the pipeline dashboard read model (#423)."""

from voxint.api.pipeline_dashboard_query import (
    _CPU_TIER_FACTOR,
    _HEURISTIC_GPU_SECONDS,
    _MIN_HISTORY_SAMPLES,
    _heuristic_seconds,
    compute_stage_eta,
)


class TestComputeStageEta:
    def test_active_with_elapsed(self) -> None:
        assert compute_stage_eta(300.0, 200.0, is_active=True) == 100.0

    def test_active_elapsed_exceeds_avg(self) -> None:
        assert compute_stage_eta(300.0, 500.0, is_active=True) == 0.0

    def test_active_no_elapsed(self) -> None:
        assert compute_stage_eta(300.0, None, is_active=True) == 300.0

    def test_queued(self) -> None:
        assert compute_stage_eta(300.0, None, is_active=False) == 300.0

    def test_no_avg(self) -> None:
        assert compute_stage_eta(None, None, is_active=True) is None
        assert compute_stage_eta(None, None, is_active=False) is None


class TestHeuristicSeconds:
    def test_gpu_tier(self) -> None:
        assert _heuristic_seconds("transcribe", "gpu") == 600.0
        assert _heuristic_seconds("acquire", "gpu") == 45.0

    def test_cpu_tier_scales_compute_bound(self) -> None:
        assert _heuristic_seconds("transcribe", "cpu") == 600.0 * _CPU_TIER_FACTOR
        assert _heuristic_seconds("diarize_embed", "cpu") == 450.0 * _CPU_TIER_FACTOR

    def test_cpu_tier_does_not_scale_io_bound(self) -> None:
        assert _heuristic_seconds("acquire", "cpu") == 45.0
        assert _heuristic_seconds("enhance_match", "cpu") == 120.0
        assert _heuristic_seconds("finalize", "cpu") == 30.0

    def test_rocm_metal_use_gpu_defaults(self) -> None:
        for tier in ("rocm", "metal"):
            assert _heuristic_seconds("transcribe", tier) == 600.0

    def test_unknown_stage_returns_default(self) -> None:
        assert _heuristic_seconds("unknown_stage", "gpu") == 120.0


class TestConstants:
    def test_all_stages_have_heuristics(self) -> None:
        from voxint.db.models import STAGE_ORDER

        for stage in STAGE_ORDER:
            assert stage.value in _HEURISTIC_GPU_SECONDS

    def test_min_history_samples_positive(self) -> None:
        assert _MIN_HISTORY_SAMPLES >= 1
