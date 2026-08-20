"""Celery application. Stage tasks are thin wrappers over pure pipeline functions."""

from typing import Any

from celery import Celery
from kombu import Queue  # type: ignore[import-untyped]

from voxint.config import Settings, get_settings

settings = get_settings()

POST_QUEUE = "post"


def build_beat_schedule(settings: Settings) -> dict[str, dict[str, Any]]:
    """Beat entries for this configuration.

    The recovery sweep always runs; the media-retention GC sweep (issue #15) and
    the run-webhook delivery sweep (issue #12) are opt-in — only scheduled when
    the operator has enabled each (both tasks re-check their gate as a backstop,
    so a stale entry can never act).

    The watch-folder sweep (issue #60) is registered UNCONDITIONALLY, because its
    enable gate is a runtime DB override (app_settings.watch_folder_enabled) that
    the startup env config cannot see; the task re-checks the effective gate and
    no-ops (one DB read, no walk) when disabled, so a UI toggle applies with no
    restart. Its cadence stays an env setting.
    """
    schedule: dict[str, dict[str, Any]] = {
        "recovery-sweep": {
            "task": "voxint.recovery_sweep",
            "schedule": settings.recovery_sweep_seconds,
        },
        "watch-sweep": {
            "task": "voxint.watch_sweep",
            "schedule": settings.watch_folder_sweep_seconds,
        },
    }
    if settings.media_retention_enabled:
        schedule["gc-sweep"] = {
            "task": "voxint.gc_sweep",
            "schedule": settings.gc_sweep_seconds,
        }
    if settings.notify_enabled:
        schedule["notify-sweep"] = {
            "task": "voxint.notify_sweep",
            "schedule": settings.notify_sweep_seconds,
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
# Declare both lanes explicitly while retaining Celery's conventional default
# queue. A worker started without ``-Q`` therefore consumes BOTH queues, keeping
# the base Compose command and native launcher unchanged; GPU deployments alone
# split the lanes into separate workers with overlay-level ``-Q`` flags.
app.conf.task_queues = (Queue("celery"), Queue(POST_QUEUE))
app.conf.task_default_queue = "celery"
# Run assets and speaker research are LLM-bound too: they must not serialize
# behind GPU work on a concurrency-1 GPU-lane worker. The beat sweeps route to
# the post lane for the same reason, and one more: the recovery sweep is the
# fallback that republishes a handed-off run whose finish publication was lost,
# so on a split deployment it must never sit queued behind a multi-hour GPU
# segment (observed on maintainer hardware: a sweep parked behind hundreds of
# backlogged GPU-lane messages). Sweeps are DB/broker-bound, never GPU-bound.
# Flagless single-worker deployments are unaffected: one worker drains both
# queues either way.
app.conf.task_routes = {
    "voxint.finish_pipeline": {"queue": POST_QUEUE},
    "voxint.generate_run_asset": {"queue": POST_QUEUE},
    "voxint.research_speaker": {"queue": POST_QUEUE},
    "voxint.recovery_sweep": {"queue": POST_QUEUE},
    "voxint.gc_sweep": {"queue": POST_QUEUE},
    "voxint.notify_sweep": {"queue": POST_QUEUE},
    "voxint.watch_sweep": {"queue": POST_QUEUE},
}
# Acks-late + Redis: an unacked task is redelivered after this horizon, so it
# must exceed the longest single lane-task execution (run_pipeline or
# finish_pipeline). Stage claims make an early redelivery harmless (the
# duplicate sees an active claim and returns), but a too-small value still
# churns the queue.
app.conf.broker_transport_options = {
    "visibility_timeout": settings.celery_visibility_timeout_seconds
}
app.conf.beat_schedule = build_beat_schedule(settings)
