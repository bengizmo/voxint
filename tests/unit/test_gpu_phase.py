"""Phase predicates fail closed during transitions, with a query-free off switch."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from voxint.config import Settings
from voxint.db.models import GPU_SEGMENT, POST_SEGMENT
from voxint.gpu_phase.state import (
    GpuPhase,
    GpuPhaseSnapshot,
    gpu_lane_open,
    lane_open_for_segment,
    post_lane_open,
)


def phase_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "gpu_phase_enabled": True,
        "gpu_lease_acquire_url": "http://localhost/acquire",
        "gpu_lease_release_url": "https://localhost/release",
        "gpu_lease_status_url": "http://localhost/status",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


@pytest.mark.parametrize("ready", [False, True])
@pytest.mark.parametrize("age", [None, 0, 90, 91])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("phase", [None, *GpuPhase])
def test_all_phase_predicates(
    phase: GpuPhase | None, enabled: bool, ready: bool, age: int | None
) -> None:
    now = datetime.now(UTC)
    snapshot = (
        GpuPhaseSnapshot(
            1,
            phase,
            now,
            None,
            None,
            None,
            0,
            None,
            None,
            now,
            ready,
            now - timedelta(seconds=age) if age is not None else None,
        )
        if phase is not None
        else None
    )
    settings = phase_settings(gpu_phase_enabled=enabled)
    gpu = not enabled or phase == GpuPhase.AUDIO
    # A missing row (None) closes both lanes while enabled.
    post = not enabled or (phase == GpuPhase.LLM and ready and age is not None and age <= 90)
    assert gpu_lane_open(snapshot, settings) is gpu
    assert post_lane_open(snapshot, settings, now=now) is post
    assert lane_open_for_segment(snapshot, settings, GPU_SEGMENT) is gpu
    assert lane_open_for_segment(snapshot, settings, POST_SEGMENT, now=now) is post


def test_unknown_segment_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown pipeline segment"):
        lane_open_for_segment(None, phase_settings(), frozenset())


@pytest.mark.parametrize(
    "field", ["gpu_lease_acquire_url", "gpu_lease_release_url", "gpu_lease_status_url"]
)
@pytest.mark.parametrize("url", ["", "ftp://localhost/path", "http://", "http://[bad"])
def test_enabled_requires_http_endpoints(field: str, url: str) -> None:
    with pytest.raises(ValidationError, match=field):
        phase_settings(**{field: url})


def test_config_bounds_and_secrets() -> None:
    assert Settings(_env_file=None).gpu_phase_enabled is False
    with pytest.raises(ValidationError, match="gpu_lease_acquire_url"):
        Settings(_env_file=None, gpu_phase_enabled=True)
    for field, value in [
        ("gpu_phase_tick_seconds", 4),
        ("gpu_phase_min_dwell_seconds", -1),
        ("gpu_phase_max_audio_seconds", 599),
    ]:
        with pytest.raises(ValidationError, match=field):
            phase_settings(**{field: value})
    # The timing relation only binds when the feature is on.
    assert (
        Settings(
            _env_file=None, gpu_phase_min_dwell_seconds=8000, gpu_phase_max_audio_seconds=100
        ).gpu_phase_enabled
        is False
    )
    settings = phase_settings(
        gpu_phase_min_dwell_seconds=0,
        gpu_phase_max_audio_seconds=0,
        gpu_phase_tick_seconds=5,
        gpu_lease_token="private-token",
    )
    assert "private-token" not in repr(settings)


def test_disabled_admission_issues_no_query() -> None:
    from unittest.mock import MagicMock

    from voxint.gpu_phase.state import admit_lane

    session = MagicMock()
    assert admit_lane(session, phase_settings(gpu_phase_enabled=False), POST_SEGMENT)
    session.execute.assert_not_called()
