"""Retry bookkeeping against real Postgres: the ledger-bounded requeue path."""

import uuid

import pytest
from sqlalchemy.orm import Session, sessionmaker

from voxint.clients.errors import ServiceError
from voxint.db.models import MediaItem, PipelineRun, RunStatus, Stage
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
