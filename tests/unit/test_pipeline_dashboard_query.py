"""Pure-function tests for the pipeline dashboard read model (#423)."""

from datetime import UTC, datetime, timedelta

import pytest

from voxint.api import health_probe
from voxint.api.pipeline_dashboard_query import (
    _CPU_TIER_FACTOR,
    _HEURISTIC_GPU_SECONDS,
    _MIN_HISTORY_SAMPLES,
    _SERVICE_STAGE,
    RunStageProgress,
    StageProgress,
    _estimate_drain,
    _heuristic_seconds,
    compute_stage_eta,
    degraded_stages,
    estimate_run_stage_progress,
)


def _stage(
    stage: str = "transcribe",
    queued: int = 0,
    active: int = 0,
    avg_seconds: float | None = 100.0,
    eta_seconds: float | None = None,
) -> StageProgress:
    return StageProgress(
        stage=stage,
        queued=queued,
        active=active,
        active_started_at=None,
        avg_seconds=avg_seconds,
        eta_seconds=eta_seconds,
        using_heuristic=False,
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


class TestEstimateRunStageProgress:
    started_at = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)

    def test_start_and_clock_skew_are_zero_percent(self) -> None:
        expected = RunStageProgress(stage="transcribe", percent=0, overrun=False)
        assert (
            estimate_run_stage_progress(
                "transcribe", self.started_at, 600.0, self.started_at
            )
            == expected
        )
        assert (
            estimate_run_stage_progress(
                "transcribe",
                self.started_at,
                600.0,
                self.started_at - timedelta(seconds=5),
            )
            == expected
        )

    def test_halfway_is_fifty_percent(self) -> None:
        assert estimate_run_stage_progress(
            "transcribe",
            self.started_at,
            600.0,
            self.started_at + timedelta(seconds=300),
        ) == RunStageProgress(stage="transcribe", percent=50, overrun=False)

    def test_just_below_average_floors_to_ninety_nine(self) -> None:
        assert estimate_run_stage_progress(
            "transcribe",
            self.started_at,
            600.0,
            self.started_at + timedelta(seconds=599.9),
        ) == RunStageProgress(stage="transcribe", percent=99, overrun=False)

    @pytest.mark.parametrize("elapsed_seconds", [600, 900])
    def test_at_or_beyond_average_is_overrun(self, elapsed_seconds: int) -> None:
        assert estimate_run_stage_progress(
            "transcribe",
            self.started_at,
            600.0,
            self.started_at + timedelta(seconds=elapsed_seconds),
        ) == RunStageProgress(stage="transcribe", percent=None, overrun=True)

    @pytest.mark.parametrize(
        ("stage", "started_at", "avg_seconds"),
        [
            (None, started_at, 600.0),
            ("transcribe", None, 600.0),
            ("transcribe", started_at, None),
            ("transcribe", started_at, 0.0),
        ],
    )
    def test_missing_or_nonpositive_inputs_return_none(
        self,
        stage: str | None,
        started_at: datetime | None,
        avg_seconds: float | None,
    ) -> None:
        assert (
            estimate_run_stage_progress(
                stage, started_at, avg_seconds, self.started_at
            )
            is None
        )


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


class TestDegradedStages:
    def test_transcription_down(self) -> None:
        assert degraded_stages([("transcription", False)], llm_enabled=True) == {
            "transcribe": "transcriber is down"
        }

    def test_diarization_and_embedding_down(self) -> None:
        assert degraded_stages(
            [("diarization", False), ("speaker embedding", False)],
            llm_enabled=True,
        ) == {"diarize_embed": "voice separation is down and speaker matching is down"}

    def test_only_embedding_down(self) -> None:
        assert degraded_stages([("speaker embedding", False)], llm_enabled=True) == {
            "diarize_embed": "speaker matching is down"
        }

    def test_llm_disabled_with_services_up(self) -> None:
        services = [(name, True) for name in _SERVICE_STAGE]
        assert degraded_stages(services, llm_enabled=False) == {
            "enhance_match": "local AI model is off"
        }

    def test_missing_telemetry_does_not_degrade(self) -> None:
        assert degraded_stages([], llm_enabled=None) == {}

    def test_unknown_service_ignored(self) -> None:
        assert degraded_stages([("future service", False)], llm_enabled=True) == {}

    def test_service_mapping_matches_probe_names(self) -> None:
        assert set(_SERVICE_STAGE) <= {name for name, _ in health_probe._SERVICES}


class TestEstimateDrain:
    def test_zero_outstanding_returns_none(self) -> None:
        assert _estimate_drain([_stage(active=1, eta_seconds=50.0)], 0) is None

    def test_active_only_returns_eta_plus_downstream(self) -> None:
        stages = [
            _stage(stage="a", active=1, eta_seconds=50.0, avg_seconds=100.0),
            _stage(stage="b", avg_seconds=200.0),
        ]
        # active at stage a: 50s remaining + 200s downstream
        assert _estimate_drain(stages, 1) == 250.0

    def test_queued_charged_suffix_sum(self) -> None:
        stages = [
            _stage(stage="a", queued=1, avg_seconds=100.0),
            _stage(stage="b", avg_seconds=200.0),
        ]
        # queued at stage a: 1 * (100 + 200) = 300
        assert _estimate_drain(stages, 1) == 300.0

    def test_all_etas_zero_returns_zero_not_none(self) -> None:
        """Overrun state: all active stages exceeded their average."""
        stages = [
            _stage(stage="a", active=1, eta_seconds=0.0, avg_seconds=100.0),
        ]
        result = _estimate_drain(stages, 1)
        assert result is not None
        assert result == 0.0

    def test_no_estimates_available(self) -> None:
        stages = [_stage(avg_seconds=None, eta_seconds=None)]
        assert _estimate_drain(stages, 1) is None


class TestConstants:
    def test_all_stages_have_heuristics(self) -> None:
        from voxint.db.models import STAGE_ORDER

        for stage in STAGE_ORDER:
            assert stage.value in _HEURISTIC_GPU_SECONDS

    def test_min_history_samples_positive(self) -> None:
        assert _MIN_HISTORY_SAMPLES >= 1
