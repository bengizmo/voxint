"""Phase coordination has a dedicated queue and opt-in, expiring beat ticks."""

from tests.unit.test_gpu_phase import phase_settings
from voxint.worker.app import GPU_PHASE_QUEUE, app, build_beat_schedule


def test_tick_queue_and_schedule():
    assert app.conf.task_routes["voxint.gpu_phase_tick"] == {"queue": GPU_PHASE_QUEUE}
    assert "gpu-phase-tick" not in build_beat_schedule(phase_settings(gpu_phase_enabled=False))
    assert build_beat_schedule(phase_settings(gpu_phase_tick_seconds=45))["gpu-phase-tick"] == {
        "task": "voxint.gpu_phase_tick",
        "schedule": 45,
        "options": {"expires": 45},
    }


def test_tick_is_fire_and_forget_without_autoretry():
    from voxint.worker.tasks import gpu_phase_tick

    assert gpu_phase_tick.ignore_result is True
    assert getattr(gpu_phase_tick, "autoretry_for", ()) == ()
