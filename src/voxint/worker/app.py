"""Celery application. Stage tasks are thin wrappers over pure pipeline functions."""

from typing import Any

from celery import Celery

from voxint.config import Settings, get_settings

settings = get_settings()


def build_beat_schedule(settings: Settings) -> dict[str, dict[str, Any]]:
    """Beat entries for this configuration.

    The recovery sweep always runs; the media-retention GC sweep (issue #15) is
    opt-in — only scheduled when the operator has enabled it (the task re-checks
    the gate as a backstop, so a stale entry can never act).
    """
    schedule: dict[str, dict[str, Any]] = {
        "recovery-sweep": {
            "task": "voxint.recovery_sweep",
            "schedule": settings.recovery_sweep_seconds,
        }
    }
    if settings.media_retention_enabled:
        schedule["gc-sweep"] = {
            "task": "voxint.gc_sweep",
            "schedule": settings.gc_sweep_seconds,
        }
    return schedule


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
app.conf.beat_schedule = build_beat_schedule(settings)
