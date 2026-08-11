"""Celery application. Stage tasks are thin wrappers over pure pipeline functions."""

from celery import Celery

from voxint.config import get_settings

settings = get_settings()

app = Celery(
    "voxint",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["voxint.worker.tasks"],
)
app.conf.task_acks_late = True
app.conf.worker_prefetch_multiplier = 1
# Acks-late + Redis: an unacked task is redelivered after this horizon, so it
# must exceed the longest run_pipeline execution. Stage claims make an early
# redelivery harmless (the duplicate sees an active claim and returns), but a
# too-small value still churns the queue.
app.conf.broker_transport_options = {
    "visibility_timeout": settings.celery_visibility_timeout_seconds
}
app.conf.beat_schedule = {
    "recovery-sweep": {
        "task": "voxint.recovery_sweep",
        "schedule": settings.recovery_sweep_seconds,
    }
}
