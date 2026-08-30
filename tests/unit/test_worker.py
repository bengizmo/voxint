import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from voxint.clients.errors import ProtocolError, ServiceError
from voxint.config import Settings
from voxint.db.models import GPU_SEGMENT, POST_SEGMENT, RunStatus, Stage
from voxint.pipeline.engine import StageFailedError
from voxint.worker.app import POST_QUEUE, app, build_beat_schedule
from voxint.worker.tasks import (
    activity_prune,
    backoff_seconds,
    finish_pipeline,
    gc_sweep,
    notify_sweep,
    pipeline_task_for_stage,
    retryable_cause,
    run_pipeline,
)


def test_worker_reliability_settings() -> None:
    assert app.conf.task_acks_late is True
    assert app.conf.worker_prefetch_multiplier == 1
    assert "voxint.worker.tasks" in app.conf.include
    assert app.conf.broker_transport_options["visibility_timeout"] >= 21600
    assert "recovery-sweep" in app.conf.beat_schedule
    assert app.conf.task_default_queue == "celery"
    assert {queue.name for queue in app.conf.task_queues} == {"celery", POST_QUEUE}
    assert app.conf.task_routes == {
        "voxint.finish_pipeline": {"queue": POST_QUEUE},
        "voxint.generate_run_asset": {"queue": POST_QUEUE},
        "voxint.research_speaker": {"queue": POST_QUEUE},
        # The sweeps must never queue behind a GPU segment on a split
        # deployment — the recovery sweep is the lost-handoff fallback.
        "voxint.recovery_sweep": {"queue": POST_QUEUE},
        "voxint.gc_sweep": {"queue": POST_QUEUE},
        "voxint.notify_sweep": {"queue": POST_QUEUE},
        "voxint.watch_sweep": {"queue": POST_QUEUE},
        "voxint.activity_prune": {"queue": POST_QUEUE},
        "voxint.compute_speaker_insights": {"queue": POST_QUEUE},
        "voxint.compute_term_stats": {"queue": POST_QUEUE},
        "voxint.media_reconcile": {"queue": POST_QUEUE},
        "voxint.plugin.synthdetect.score_run": {"queue": POST_QUEUE},
    }


def test_pipeline_task_for_stage_routes_both_segments() -> None:
    assert pipeline_task_for_stage(None) is run_pipeline
    assert all(pipeline_task_for_stage(stage) is run_pipeline for stage in GPU_SEGMENT)
    assert all(pipeline_task_for_stage(stage) is finish_pipeline for stage in POST_SEGMENT)


def test_publish_finish_defers_only_broker_errors(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from celery.exceptions import OperationalError

    import voxint.worker.tasks as tasks_mod

    def broker_down(args, **kwargs):  # type: ignore[no-untyped-def]
        raise OperationalError("broker down")

    monkeypatch.setattr(tasks_mod.finish_pipeline, "apply_async", broker_down)
    assert tasks_mod._publish_finish_or_defer(uuid.uuid4()) is False

    def bug(args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("publisher bug")

    monkeypatch.setattr(tasks_mod.finish_pipeline, "apply_async", bug)
    with pytest.raises(RuntimeError, match="publisher bug"):
        tasks_mod._publish_finish_or_defer(uuid.uuid4())


def _stub_segment_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, SimpleNamespace, uuid.UUID]:
    """Remove DB/context construction so lane-result behavior is isolated."""
    import voxint.worker.tasks as tasks_mod

    factory = MagicMock()
    factory.return_value.__enter__.return_value.get.return_value = None
    ctx = SimpleNamespace(llm=None)
    monkeypatch.setattr(tasks_mod, "_runtime", lambda: (factory, ctx))
    monkeypatch.setattr(tasks_mod, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(tasks_mod.app_settings, "get_app_settings", lambda session: None)
    monkeypatch.setattr(
        tasks_mod.app_settings,
        "resolve_effective_llm_api_key",
        lambda row, settings: "",
    )
    monkeypatch.setattr(
        tasks_mod.app_settings, "llm_bundled_active", lambda row, settings: False
    )
    monkeypatch.setattr(tasks_mod, "resolve_run_preferences", lambda row, settings: object())
    monkeypatch.setattr(tasks_mod, "domain_pack_from_snapshot", lambda snapshot, settings: object())
    monkeypatch.setattr(tasks_mod, "apply_run_preferences", lambda *args, **kwargs: ctx)
    monkeypatch.setattr(tasks_mod, "build_stage_fns", lambda applied: {})
    return factory, ctx, uuid.uuid4()


def test_gpu_segment_publishes_only_a_post_lane_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import voxint.worker.tasks as tasks_mod

    _factory, _ctx, run_id = _stub_segment_driver(monkeypatch)
    results = iter(
        (
            SimpleNamespace(
                status=RunStatus.QUEUED, current_stage=Stage.ENHANCE_MATCH
            ),
            SimpleNamespace(status=RunStatus.QUEUED, current_stage=Stage.TRANSCRIBE),
        )
    )
    monkeypatch.setattr(tasks_mod, "execute_run", lambda *args, **kwargs: next(results))
    published: list[tuple[tuple[str, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        tasks_mod.finish_pipeline,
        "apply_async",
        lambda args, **kwargs: published.append((args, kwargs)),
    )

    assert tasks_mod.run_pipeline.run(str(run_id)) == RunStatus.QUEUED.value
    assert published == [((str(run_id),), {"ignore_result": True})]

    assert tasks_mod.run_pipeline.run(str(run_id)) == RunStatus.QUEUED.value
    assert published == [((str(run_id),), {"ignore_result": True})]


def test_only_post_segment_runs_completion_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import voxint.worker.tasks as tasks_mod

    factory, _ctx, run_id = _stub_segment_driver(monkeypatch)
    monkeypatch.setattr(
        tasks_mod,
        "execute_run",
        lambda *args, **kwargs: SimpleNamespace(status=RunStatus.COMPLETED),
    )
    generated: list[tuple[object, uuid.UUID, Settings]] = []
    monkeypatch.setattr(
        tasks_mod,
        "_autogenerate_run_assets",
        lambda received_factory, received_id, settings: generated.append(
            (received_factory, received_id, settings)
        ),
    )

    assert tasks_mod.finish_pipeline.run(str(run_id)) == RunStatus.COMPLETED.value
    assert len(generated) == 1
    assert generated[0][0] is factory
    assert generated[0][1] == run_id

    assert tasks_mod.run_pipeline.run(str(run_id)) == RunStatus.COMPLETED.value
    assert len(generated) == 1


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


def test_watch_sweep_task_delegates_to_sweep(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # The task is a thin wrapper: it hands the process session factory to
    # sweep_watch_folders and returns its summary dict.
    import voxint.worker.tasks as tasks_mod
    from voxint.ingest.watch import WatchSweepSummary

    monkeypatch.setattr(tasks_mod, "get_settings", lambda: Settings(_env_file=None))
    sentinel_factory = object()
    monkeypatch.setattr(tasks_mod, "_runtime", lambda: (sentinel_factory, None))
    captured: dict[str, object] = {}

    def fake_sweep(factory, settings):  # type: ignore[no-untyped-def]
        captured["factory"] = factory
        return WatchSweepSummary(picked_up=2, already_known=1)

    monkeypatch.setattr(tasks_mod, "sweep_watch_folders", fake_sweep)

    result = tasks_mod.watch_sweep()
    assert result == WatchSweepSummary(picked_up=2, already_known=1).as_dict()
    assert captured["factory"] is sentinel_factory


def test_watch_sweep_beat_entry_is_unconditional() -> None:
    # Unlike gc/notify, the watch sweep is ALWAYS registered (its enable gate is a
    # runtime DB override the startup config can't see); the task re-checks the
    # effective gate and no-ops when off. Cadence stays an env setting.
    disabled = build_beat_schedule(Settings(_env_file=None))
    assert disabled["watch-sweep"] == {"task": "voxint.watch_sweep", "schedule": 300}
    tuned = build_beat_schedule(Settings(_env_file=None, watch_folder_sweep_seconds=45))
    assert tuned["watch-sweep"] == {"task": "voxint.watch_sweep", "schedule": 45}


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


def test_activity_prune_beat_entry_is_opt_in() -> None:
    # OFF by default: no activity-prune entry unless the activity feed is on.
    assert "activity-prune" not in build_beat_schedule(Settings(_env_file=None))
    enabled = build_beat_schedule(Settings(_env_file=None, console_activity_enabled=True))
    assert enabled["activity-prune"]["task"] == "voxint.activity_prune"
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


def test_activity_prune_task_noops_when_disabled(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # The task re-checks the gate: disabled → no runtime, no DB.
    import voxint.worker.tasks as tasks_mod

    monkeypatch.setattr(tasks_mod, "get_settings", lambda: Settings(_env_file=None))

    def _boom() -> None:  # pragma: no cover - must not be called
        raise AssertionError("_runtime() must not run when activity is disabled")

    monkeypatch.setattr(tasks_mod, "_runtime", _boom)
    assert activity_prune() == {"pruned": 0}


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
