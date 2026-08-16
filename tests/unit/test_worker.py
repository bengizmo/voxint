from voxint.clients.errors import ProtocolError, ServiceError
from voxint.config import Settings
from voxint.db.models import Stage
from voxint.pipeline.engine import StageFailedError
from voxint.worker.app import app, build_beat_schedule
from voxint.worker.tasks import backoff_seconds, gc_sweep, notify_sweep, retryable_cause


def test_worker_reliability_settings() -> None:
    assert app.conf.task_acks_late is True
    assert app.conf.worker_prefetch_multiplier == 1
    assert "voxint.worker.tasks" in app.conf.include
    assert app.conf.broker_transport_options["visibility_timeout"] >= 21600
    assert "recovery-sweep" in app.conf.beat_schedule


def test_gc_sweep_beat_entry_is_opt_in() -> None:
    # OFF by default: no gc-sweep entry unless the operator enables retention.
    assert "gc-sweep" not in build_beat_schedule(Settings(_env_file=None))
    enabled = build_beat_schedule(
        Settings(_env_file=None, media_retention_enabled=True, gc_sweep_seconds=1234)
    )
    assert enabled["gc-sweep"] == {"task": "voxint.gc_sweep", "schedule": 1234}
    assert "recovery-sweep" in enabled  # the recovery sweep is unconditional


def test_gc_sweep_task_noops_when_disabled(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # The task re-checks the gate itself: disabled → it returns an all-zero
    # summary and never touches the DB runtime (a stale beat entry can't act).
    import voxint.worker.tasks as tasks_mod

    monkeypatch.setattr(tasks_mod, "get_settings", lambda: Settings(_env_file=None))

    def _boom() -> None:  # pragma: no cover - must not be called
        raise AssertionError("_runtime() must not run when retention is disabled")

    monkeypatch.setattr(tasks_mod, "_runtime", _boom)
    assert gc_sweep() == {
        "selected": 0,
        "reclaimed": 0,
        "missing": 0,
        "failed": 0,
        "bytes": 0,
    }


def test_notify_sweep_beat_entry_is_opt_in() -> None:
    # OFF by default: no notify-sweep entry unless the operator enables webhooks.
    assert "notify-sweep" not in build_beat_schedule(Settings(_env_file=None))
    enabled = build_beat_schedule(
        Settings(
            _env_file=None,
            notify_enabled=True,
            notify_webhook_url="https://hooks.example.com/x",
            notify_webhook_secret="a-sufficiently-long-secret",
            notify_sweep_seconds=42,
        )
    )
    assert enabled["notify-sweep"] == {"task": "voxint.notify_sweep", "schedule": 42}
    assert "recovery-sweep" in enabled  # the recovery sweep is unconditional


def test_notify_sweep_task_noops_when_disabled(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # The task re-checks the gate itself: disabled → all-zero summary, no runtime.
    import voxint.worker.tasks as tasks_mod

    monkeypatch.setattr(tasks_mod, "get_settings", lambda: Settings(_env_file=None))

    def _boom() -> None:  # pragma: no cover - must not be called
        raise AssertionError("_runtime() must not run when notify is disabled")

    monkeypatch.setattr(tasks_mod, "_runtime", _boom)
    assert notify_sweep() == {
        "claimed": 0,
        "delivered": 0,
        "suppressed": 0,
        "retried": 0,
        "dead": 0,
        "purged": 0,
    }


def wrap(cause: Exception) -> StageFailedError:
    return StageFailedError(Stage.TRANSCRIBE, cause)


def test_retryable_cause_follows_service_verdict() -> None:
    assert retryable_cause(wrap(ServiceError("saturated", "busy", retryable=True)))
    assert not retryable_cause(wrap(ServiceError("inference_failed", "boom", retryable=False)))
    assert not retryable_cause(wrap(ProtocolError("bad shape")))
    assert not retryable_cause(wrap(ValueError("some stage bug")))


def test_backoff_is_exponential_and_capped() -> None:
    assert backoff_seconds(1, 30.0, 1800.0) == 30.0
    assert backoff_seconds(2, 30.0, 1800.0) == 60.0
    assert backoff_seconds(5, 30.0, 1800.0) == 480.0
    assert backoff_seconds(50, 30.0, 1800.0) == 1800.0
    assert backoff_seconds(0, 30.0, 1800.0) == 30.0  # defensive floor
