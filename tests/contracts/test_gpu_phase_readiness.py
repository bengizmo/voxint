"""The language-model lane requires a recent answer, independently of phase age."""

from datetime import UTC, datetime, timedelta

from tests.unit.test_gpu_phase import phase_settings
from voxint.db.models import POST_SEGMENT
from voxint.gpu_phase.state import (
    GpuPhase,
    GpuPhaseSnapshot,
    lane_open_for_segment,
    llm_freshness_seconds,
    post_lane_open,
)


def test_readiness_freshness_contract() -> None:
    now = datetime.now(UTC)
    settings = phase_settings(gpu_phase_tick_seconds=40)
    assert llm_freshness_seconds(5) == 90
    assert llm_freshness_seconds(40) == 120
    snapshot = GpuPhaseSnapshot(
        1,
        GpuPhase.LLM,
        now,
        None,
        None,
        None,
        0,
        None,
        None,
        now,
        True,
        now - timedelta(seconds=120),
    )
    assert post_lane_open(snapshot, settings, now=now)
    assert lane_open_for_segment(snapshot, settings, POST_SEGMENT, now=now)
    later = now + timedelta(microseconds=1)
    assert not post_lane_open(snapshot, settings, now=later)
    assert not lane_open_for_segment(snapshot, settings, POST_SEGMENT, now=later)
