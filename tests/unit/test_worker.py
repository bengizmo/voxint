from voxint.clients.errors import ProtocolError, ServiceError
from voxint.db.models import Stage
from voxint.pipeline.engine import StageFailedError
from voxint.worker.app import app
from voxint.worker.tasks import backoff_seconds, retryable_cause


def test_worker_reliability_settings() -> None:
    assert app.conf.task_acks_late is True
    assert app.conf.worker_prefetch_multiplier == 1
    assert "voxint.worker.tasks" in app.conf.include
    assert app.conf.broker_transport_options["visibility_timeout"] >= 21600
    assert "recovery-sweep" in app.conf.beat_schedule


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
