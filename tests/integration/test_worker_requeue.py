"""Retry bookkeeping against real Postgres: the ledger-bounded requeue path."""

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy.orm import Session, sessionmaker

from voxint.clients.errors import ServiceError
from voxint.db.models import GPU_SEGMENT, MediaItem, PipelineRun, RunStatus, Stage
from voxint.domain_packs.base import load_default
from voxint.pipeline.engine import (
    StageFailedError,
    StageFn,
    execute_run,
    recover_interrupted_runs,
    submit,
)
from voxint.pipeline.transitions import RunSnapshot, snapshot
from voxint.worker.tasks import requeue_failed_stage, stage_attempts


def failing_stage_fns(failures_left: dict[str, int]) -> dict[Stage, StageFn]:
    """PREPARE fails with a retryable ServiceError until the budget runs out."""

    def prepare(session: Session, run_id: uuid.UUID) -> None:
        if failures_left["n"] > 0:
            failures_left["n"] -= 1
            raise ServiceError("saturated", "busy", retryable=True)

    def noop(session: Session, run_id: uuid.UUID) -> None:
        pass

    return {stage: (prepare if stage is Stage.PREPARE else noop) for stage in Stage}


def submit_run(session_factory: sessionmaker[Session]) -> uuid.UUID:
    with session_factory() as session:
        media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
        session.add(media)
        session.flush()
        run_id = submit(session, media.id, domain_pack=load_default().to_mapping()).id
        session.commit()
    return run_id


def fail_once(
    session_factory: sessionmaker[Session],
    run_id: uuid.UUID,
    fns: dict[Stage, StageFn],
) -> RunSnapshot:
    with pytest.raises(StageFailedError) as exc_info:
        execute_run(session_factory, run_id, fns)
    failed = exc_info.value.failed_snapshot
    assert failed is not None
    return failed


def test_requeue_retry_loop_completes_and_counts_attempts(
    session_factory: sessionmaker[Session],
) -> None:
    run_id = submit_run(session_factory)
    fns = failing_stage_fns({"n": 2})

    for expected_attempts in (1, 2):
        failed = fail_once(session_factory, run_id, fns)
        with session_factory() as session:
            assert stage_attempts(session, run_id, Stage.PREPARE) == expected_attempts
            run = session.get(PipelineRun, run_id)
            assert run is not None and run.status == RunStatus.FAILED.value
        assert requeue_failed_stage(session_factory, failed) is True

    final = execute_run(session_factory, run_id, fns)
    assert final.status is RunStatus.COMPLETED
    with session_factory() as session:
        # The budget counts transient failures only — the completed attempt
        # and any interruptions never eat into it.
        assert stage_attempts(session, run_id, Stage.PREPARE) == 2


def test_stale_requeue_callback_declines_aba(
    session_factory: sessionmaker[Session],
) -> None:
    """An old failure's callback must not requeue a NEWER failure at the same
    stage — same (FAILED, stage) shape, different revision."""
    run_id = submit_run(session_factory)
    fns = failing_stage_fns({"n": 5})

    first_failed = fail_once(session_factory, run_id, fns)
    # Someone else (operator, another sweep) requeues and it fails again.
    assert requeue_failed_stage(session_factory, first_failed) is True
    second_failed = fail_once(session_factory, run_id, fns)
    assert second_failed.revision > first_failed.revision

    # The stale callback fires now: same stage, still FAILED — must decline.
    assert requeue_failed_stage(session_factory, first_failed) is False
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None and run.status == RunStatus.FAILED.value

    # The rightful owner still works.
    assert requeue_failed_stage(session_factory, second_failed) is True


def test_interrupted_attempts_do_not_eat_transient_budget(
    session_factory: sessionmaker[Session],
) -> None:
    """Lease-expiry interruptions (worker death, OOM) are not service failures."""
    run_id = submit_run(session_factory)
    failed = fail_once(session_factory, run_id, failing_stage_fns({"n": 1}))
    with session_factory() as session:
        assert stage_attempts(session, run_id, Stage.PREPARE) == 1

    # An interruption: requeue, force RUNNING with a dead claim, sweep it.
    assert requeue_failed_stage(session_factory, failed) is True
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        run.status = RunStatus.RUNNING.value
        session.commit()
    with session_factory() as session:
        assert recover_interrupted_runs(session) == [run_id]
        session.commit()

    with session_factory() as session:
        # Still 1: the interrupted attempt carries the interrupted: marker.
        assert stage_attempts(session, run_id, Stage.PREPARE) == 1


def test_duplicate_dispatch_entry_race_is_benign(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two run_pipeline tasks legitimately racing on one QUEUED run (pending
    retry + recovery sweep) must both survive the QUEUED→RUNNING entry CAS —
    one wins, the other yields; neither crashes."""
    import threading

    import voxint.pipeline.engine as engine_mod

    run_id = submit_run(session_factory)
    fns = failing_stage_fns({"n": 0})  # every stage succeeds

    barrier = threading.Barrier(2, timeout=10)
    local = threading.local()
    real_snapshot = engine_mod.snapshot

    def racing_snapshot(run: PipelineRun) -> RunSnapshot:
        snap = real_snapshot(run)
        # Hold both workers until each has read the run as QUEUED, so both
        # attempt the entry CAS against the same revision.
        if snap.status is RunStatus.QUEUED and not getattr(local, "synced", False):
            local.synced = True
            barrier.wait()
        return snap

    monkeypatch.setattr(engine_mod, "snapshot", racing_snapshot)

    errors: list[Exception] = []

    def dispatch() -> None:
        try:
            execute_run(session_factory, run_id, fns)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=dispatch) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert errors == []
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.COMPLETED.value


def test_requeue_rejects_non_failed_snapshot(
    session_factory: sessionmaker[Session],
) -> None:
    run_id = submit_run(session_factory)
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        held = snapshot(run)
    assert held.status is RunStatus.QUEUED
    assert requeue_failed_stage(session_factory, held) is False


def test_recovery_respects_attempt_budget(
    session_factory: sessionmaker[Session],
) -> None:
    """A run whose stage burned its attempts is parked FAILED by the sweep,
    not granted unlimited retries via lease expiry."""
    run_id = submit_run(session_factory)
    fns = failing_stage_fns({"n": 10})

    # Burn three attempts (fail → requeue → fail → ...).
    failed = fail_once(session_factory, run_id, fns)
    for _ in range(2):
        assert requeue_failed_stage(session_factory, failed) is True
        failed = fail_once(session_factory, run_id, fns)

    # Simulate a worker death: force RUNNING with an expired/no claim.
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        run.status = RunStatus.RUNNING.value
        session.commit()

    with session_factory() as session:
        recovered = recover_interrupted_runs(session, max_attempts=3)
        session.commit()
    assert recovered == []
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.FAILED.value

    # Without a budget the same shape is recovered.
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        run.status = RunStatus.RUNNING.value
        session.commit()
    with session_factory() as session:
        recovered = recover_interrupted_runs(session)
        session.commit()
    assert recovered == [run_id]


def test_recovery_sweep_routes_stale_runs_by_current_stage(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recovery must publish each durable QUEUED row to its owning lane."""
    from voxint.config import Settings
    from voxint.worker import tasks

    gpu_run_id = submit_run(session_factory)
    post_run_id = submit_run(session_factory)
    stale_at = datetime.now(UTC) - timedelta(hours=2)
    with session_factory() as session:
        gpu_run = session.get(PipelineRun, gpu_run_id)
        post_run = session.get(PipelineRun, post_run_id)
        assert gpu_run is not None and post_run is not None
        gpu_run.current_stage = Stage.TRANSCRIBE.value
        gpu_run.updated_at = stale_at
        post_run.current_stage = Stage.ENHANCE_MATCH.value
        post_run.updated_at = stale_at
        session.commit()

    monkeypatch.setattr(
        tasks, "get_settings", lambda: Settings(_env_file=None, queued_run_stale_seconds=60)
    )
    monkeypatch.setattr(tasks, "_runtime", lambda: (session_factory, None))
    published: list[tuple[str, str]] = []
    monkeypatch.setattr(
        tasks.run_pipeline,
        "apply_async",
        lambda args, **kwargs: published.append(("gpu", args[0])),
    )
    monkeypatch.setattr(
        tasks.finish_pipeline,
        "apply_async",
        lambda args, **kwargs: published.append(("post", args[0])),
    )

    result = tasks.recovery_sweep()

    assert result["stale_queued"] == 2
    assert set(published) == {
        ("gpu", str(gpu_run_id)),
        ("post", str(post_run_id)),
    }


def test_recovery_sweep_continues_after_one_publish_outage(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broker failure for one durable row must not skip another lane's row."""
    from celery.exceptions import OperationalError

    from voxint.config import Settings
    from voxint.worker import tasks

    gpu_run_id = submit_run(session_factory)
    post_run_id = submit_run(session_factory)
    stale_at = datetime.now(UTC) - timedelta(hours=2)
    with session_factory() as session:
        gpu_run = session.get(PipelineRun, gpu_run_id)
        post_run = session.get(PipelineRun, post_run_id)
        assert gpu_run is not None and post_run is not None
        gpu_run.current_stage = Stage.TRANSCRIBE.value
        gpu_run.updated_at = stale_at
        post_run.current_stage = Stage.ENHANCE_MATCH.value
        post_run.updated_at = stale_at
        session.commit()

    monkeypatch.setattr(
        tasks, "get_settings", lambda: Settings(_env_file=None, queued_run_stale_seconds=60)
    )
    monkeypatch.setattr(tasks, "_runtime", lambda: (session_factory, None))

    def broker_down(args: tuple[str], **kwargs: object) -> None:
        raise OperationalError("broker down")

    published: list[str] = []
    monkeypatch.setattr(tasks.run_pipeline, "apply_async", broker_down)
    monkeypatch.setattr(
        tasks.finish_pipeline,
        "apply_async",
        lambda args, **kwargs: published.append(args[0]),
    )

    result = tasks.recovery_sweep()

    assert result["stale_queued"] == 2
    assert published == [str(post_run_id)]


def _seed_stale_embedding_job(
    session_factory: sessionmaker[Session], *, age: timedelta
) -> uuid.UUID:
    """A COMPLETED run holding one QUEUED embedding job created ``age`` ago.

    The run is COMPLETED so the sweep's run pass ignores it — only the stranded
    embedding job (issue #130) is under test."""
    import hashlib

    from voxint.db.models import EmbeddingJob, EmbeddingJobStatus
    from voxint.embeddings.onnx_embedder import EMBEDDING_SPACE

    run_id = submit_run(session_factory)
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        run.status = RunStatus.COMPLETED.value
        job = EmbeddingJob(
            pipeline_run_id=run_id,
            embedding_space=EMBEDDING_SPACE,
            status=EmbeddingJobStatus.QUEUED.value,
            cancel_requested=False,
            source_content_hash=hashlib.sha256(str(run_id).encode()).hexdigest(),
            created_at=datetime.now(UTC) - age,
        )
        session.add(job)
        session.commit()
        return job.id


def test_recovery_sweep_redispatches_stale_queued_embedding_job(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A QUEUED embedding job stranded past the grace is re-dispatched by id, and
    the row is left untouched — re-dispatch, not reclaim (issue #130)."""
    from voxint.config import Settings
    from voxint.db.models import EmbeddingJob, EmbeddingJobStatus
    from voxint.worker import tasks

    stale_id = _seed_stale_embedding_job(session_factory, age=timedelta(hours=2))
    fresh_id = _seed_stale_embedding_job(session_factory, age=timedelta(seconds=5))

    monkeypatch.setattr(
        tasks, "get_settings", lambda: Settings(_env_file=None, queued_run_stale_seconds=60)
    )
    monkeypatch.setattr(tasks, "_runtime", lambda: (session_factory, None))
    dispatched: list[str] = []
    monkeypatch.setattr(
        tasks.generate_segment_embeddings,
        "apply_async",
        lambda args, **kwargs: dispatched.append(args[0]),
    )

    result = tasks.recovery_sweep()

    assert result["stale_embedding_jobs"] == 1
    # With an empty plugin registry the result is the exact pre-#138 four-key dict:
    # the `plugin_lanes` key appears only when a plugin declares a job lane, so the
    # dormant seam is byte-identical for result consumers (issue #138).
    assert set(result) == {
        "recovered",
        "stale_queued",
        "cancelled_claims_closed",
        "stale_embedding_jobs",
    }
    assert dispatched == [str(stale_id)]  # the fresh job is spared
    with session_factory() as session:
        row = session.get(EmbeddingJob, stale_id)
        assert row is not None
        assert row.status == EmbeddingJobStatus.QUEUED.value  # unmutated
        assert row.started_at is None
        # The fresh job stays QUEUED and untouched (not swept, not dispatched).
        fresh = session.get(EmbeddingJob, fresh_id)
        assert fresh is not None
        assert fresh.status == EmbeddingJobStatus.QUEUED.value


def test_recovery_sweep_continues_after_one_embedding_redispatch_outage(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broker failure re-dispatching one stale embedding job must not skip the
    others (issue #130); the count still reflects everything selected."""
    from celery.exceptions import OperationalError

    from voxint.config import Settings
    from voxint.worker import tasks

    first_id = _seed_stale_embedding_job(session_factory, age=timedelta(hours=2))
    second_id = _seed_stale_embedding_job(session_factory, age=timedelta(hours=3))

    monkeypatch.setattr(
        tasks, "get_settings", lambda: Settings(_env_file=None, queued_run_stale_seconds=60)
    )
    monkeypatch.setattr(tasks, "_runtime", lambda: (session_factory, None))
    dispatched: list[str] = []

    def flaky(args: tuple[str], **kwargs: object) -> None:
        if args[0] == str(first_id):
            raise OperationalError("broker down")
        dispatched.append(args[0])

    monkeypatch.setattr(tasks.generate_segment_embeddings, "apply_async", flaky)

    result = tasks.recovery_sweep()

    assert result["stale_embedding_jobs"] == 2
    assert dispatched == [str(second_id)]  # the outage on first didn't abort


def test_finish_pipeline_retries_its_own_transient_failure(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retry from ENHANCE_MATCH stays attached to the routed post task."""
    from celery.exceptions import Retry

    from voxint.config import Settings
    from voxint.worker import tasks

    run_id = submit_run(session_factory)
    noops = {stage: (lambda session, rid: None) for stage in Stage}
    handed_off = execute_run(session_factory, run_id, noops, stages=GPU_SEGMENT)
    assert handed_off.current_stage is Stage.ENHANCE_MATCH

    def fail_enhance(session: Session, rid: uuid.UUID) -> None:
        raise ServiceError("saturated", "busy", retryable=True)

    post_fns = dict(noops)
    post_fns[Stage.ENHANCE_MATCH] = fail_enhance
    settings = Settings(_env_file=None)
    ctx = SimpleNamespace(llm=None)
    monkeypatch.setattr(tasks, "get_settings", lambda: settings)
    monkeypatch.setattr(tasks, "_runtime", lambda: (session_factory, ctx))
    monkeypatch.setattr(tasks.app_settings, "get_app_settings", lambda session: None)
    monkeypatch.setattr(tasks, "resolve_run_preferences", lambda row, resolved: object())
    monkeypatch.setattr(
        tasks.app_settings, "resolve_effective_llm_api_key", lambda row, resolved: ""
    )
    monkeypatch.setattr(tasks.app_settings, "llm_bundled_active", lambda row, resolved: False)
    monkeypatch.setattr(tasks, "apply_run_preferences", lambda *args, **kwargs: ctx)
    monkeypatch.setattr(tasks, "build_stage_fns", lambda applied: post_fns)
    retry_calls: list[dict[str, object]] = []

    def retry(**kwargs: object) -> None:
        retry_calls.append(kwargs)
        raise Retry()

    monkeypatch.setattr(tasks.finish_pipeline, "retry", retry)
    monkeypatch.setattr(
        tasks.run_pipeline,
        "retry",
        lambda **kwargs: pytest.fail("GPU task must not own a post-stage retry"),
    )

    with pytest.raises(Retry):
        tasks.finish_pipeline.run(str(run_id))
    assert len(retry_calls) == 1
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.QUEUED.value
        assert run.current_stage == Stage.ENHANCE_MATCH.value
