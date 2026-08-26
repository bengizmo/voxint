"""Celery application. Stage tasks are thin wrappers over pure pipeline functions."""

from typing import Any

from celery import Celery
from kombu import Queue  # type: ignore[import-untyped]

from voxint.config import Settings, get_settings
from voxint.plugins import load_plugins
from voxint.plugins.base import PluginError
from voxint.plugins.boot import validate_boot

settings = get_settings()

# Plugin framework (issue #138). Build and validate the registry at worker import,
# BEFORE constructing Celery, so a malformed builtin or an env-sourced plugin
# invariant violation stops the worker, beat, and task inspection loudly rather
# than half-starting one broken feature (mirrors the api's create_app fail-loud).
# Dormant while BUILTIN is empty: no plugins, no contributions, identical config.
_registry = load_plugins(settings)
validate_boot(_registry, settings=settings)

# The core Celery task names, pinned by tests/contracts/fixtures/task_inventory.json.
# The registry only rejects a plugin task name that collides with ANOTHER plugin;
# a plugin shadowing a CORE task name is caught here, at the one worker-assembly
# seam, WITHOUT importing the task modules (that would perturb registration order
# and risk a circular import — worker.tasks imports this module).
_CORE_TASK_NAMES: frozenset[str] = frozenset(
    {
        "voxint.activity_prune",
        "voxint.finish_pipeline",
        "voxint.gc_sweep",
        "voxint.generate_run_asset",
        "voxint.generate_segment_embeddings",
        "voxint.notify_sweep",
        "voxint.recovery_sweep",
        "voxint.research_speaker",
        "voxint.run_pipeline",
        "voxint.translate_run",
        "voxint.watch_sweep",
    }
)
_core_task_collisions = sorted(_CORE_TASK_NAMES.intersection(_registry.task_names()))
if _core_task_collisions:
    raise PluginError(
        "plugin Celery task name(s) collide with core tasks: "
        + ", ".join(_core_task_collisions)
    )
# Every active plugin's task modules, appended to the core include. Empty ⇒ the
# include list carries only the core module, unchanged.
_plugin_task_modules = [
    module for plugin in _registry.plugins for module in plugin.task_modules()
]

POST_QUEUE = "post"

# Fixed cadence for the activity-outbox retention prune (issue #162): hourly is
# ample for a 500-row cap, and this earns no env knob (anti-bloat).
_ACTIVITY_PRUNE_SECONDS = 3600


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
    if settings.console_activity_enabled:
        # Retention for the activity outbox (issue #162). Hourly is ample for a
        # 500-row cap that grows one row per completed run; the task re-checks the
        # gate, so a stale entry never acts. Fixed cadence (no tuning knob).
        schedule["activity-prune"] = {
            "task": "voxint.activity_prune",
            "schedule": _ACTIVITY_PRUNE_SECONDS,
        }
    return schedule


app = Celery(
    "voxint",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["voxint.worker.tasks", *_plugin_task_modules],
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
    "voxint.activity_prune": {"queue": POST_QUEUE},
    # Active plugins' post-queue routes (registry validates a plugin only routes a
    # task it declares). Empty ⇒ no extra keys, an equal dict.
    **_registry.task_routes(),
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
