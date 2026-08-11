from voxint.worker.app import app


def test_worker_reliability_settings() -> None:
    assert app.conf.task_acks_late is True
    assert app.conf.worker_prefetch_multiplier == 1
