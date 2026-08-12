"""CAS conflict, stage-claim, and crash/restart semantics against real Postgres."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from voxint.db.models import (
    STAGE_ORDER,
    MediaItem,
    PipelineRun,
    RunStatus,
    Stage,
    StageRun,
    StageStatus,
)
from voxint.pipeline.engine import (
    StageFailedError,
    StageFn,
    execute_run,
    recover_interrupted_runs,
    submit,
)
from voxint.pipeline.transitions import (
    StaleRevisionError,
    cas_update_run,
    snapshot,
)


def make_run(session_factory: sessionmaker[Session], path: str) -> uuid.UUID:
    with session_factory() as session:
        media = MediaItem(source_path=path)
        session.add(media)
        session.flush()
        run = submit(session, media.id)
        run_id = run.id
        session.commit()
    return run_id


def advance_to(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, target: Stage
) -> None:
    """Walk a QUEUED run to RUNNING at ``target`` through legal transitions."""
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        held = snapshot(run)
        # A fresh QUEUED run must enter at STAGE_ORDER[0] (ACQUIRE); walk from
        # there through the canonical order to the requested target stage.
        held = cas_update_run(session, held, status=RunStatus.RUNNING, current_stage=STAGE_ORDER[0])
        stage = STAGE_ORDER[0]
        while stage is not target:
            following = list(Stage)[list(Stage).index(stage) + 1]
            held = cas_update_run(
                session, held, status=RunStatus.RUNNING, current_stage=following
            )
            stage = following
        session.commit()


NOOP_FNS: dict[Stage, StageFn] = {stage: (lambda s, r: None) for stage in Stage}


def test_cas_conflict_raises_stale_revision(
    session_factory: sessionmaker[Session],
) -> None:
    run_id = make_run(session_factory, "/data/media/cas.wav")
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        held = snapshot(run)
        # First writer wins... (a fresh run enters at STAGE_ORDER[0] = ACQUIRE)
        cas_update_run(session, held, status=RunStatus.RUNNING, current_stage=STAGE_ORDER[0])
        session.commit()
        # ...second writer holding the stale snapshot must NOT clobber.
        with pytest.raises(StaleRevisionError):
            cas_update_run(session, held, status=RunStatus.CANCELLED, current_stage=None)
        session.rollback()

    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.RUNNING.value
        assert run.revision == 1


def test_active_claim_blocks_second_worker(
    session_factory: sessionmaker[Session],
) -> None:
    run_id = make_run(session_factory, "/data/media/claimed.wav")
    advance_to(session_factory, run_id, Stage.TRANSCRIBE)
    # worker A holds an unexpired claim on the current stage
    with session_factory() as session:
        session.add(
            StageRun(
                pipeline_run_id=run_id,
                stage=Stage.TRANSCRIBE.value,
                status=StageStatus.RUNNING.value,
                attempt=1,
                worker_id="worker-a",
                lease_expires_at=datetime.now(tz=UTC) + timedelta(hours=1),
            )
        )
        session.commit()

    executed: list[Stage] = []

    def tracking(stage: Stage) -> StageFn:
        def fn(session: Session, rid: uuid.UUID) -> None:
            executed.append(stage)

        return fn

    # worker B must yield without executing anything
    result = execute_run(session_factory, run_id, {s: tracking(s) for s in Stage})
    assert executed == []
    assert result.status is RunStatus.RUNNING
    assert result.current_stage is Stage.TRANSCRIBE


def test_stage_failure_marks_run_failed_and_requeue_retries(
    session_factory: sessionmaker[Session],
) -> None:
    run_id = make_run(session_factory, "/data/media/flaky.wav")
    calls = {"transcribe": 0}

    def flaky_transcribe(session: Session, rid: uuid.UUID) -> None:
        calls["transcribe"] += 1
        if calls["transcribe"] == 1:
            raise RuntimeError("ASR service unavailable")

    fns = dict(NOOP_FNS)
    fns[Stage.TRANSCRIBE] = flaky_transcribe

    with pytest.raises(StageFailedError):
        execute_run(session_factory, run_id, fns)

    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.FAILED.value
        assert run.current_stage == Stage.TRANSCRIBE.value
        assert run.error is not None and "ASR service unavailable" in run.error
        # explicit requeue (FAILED -> QUEUED keeps current_stage for retry)
        held = snapshot(run)
        cas_update_run(session, held, status=RunStatus.QUEUED, current_stage=Stage.TRANSCRIBE)
        session.commit()

    final = execute_run(session_factory, run_id, fns)
    assert final.status is RunStatus.COMPLETED
    assert calls["transcribe"] == 2

    with session_factory() as session:
        attempts = (
            session.execute(
                select(StageRun).where(
                    StageRun.pipeline_run_id == run_id,
                    StageRun.stage == Stage.TRANSCRIBE.value,
                )
            )
            .scalars()
            .all()
        )
        by_attempt = {a.attempt: a.status for a in attempts}
        assert by_attempt == {1: StageStatus.FAILED.value, 2: StageStatus.COMPLETED.value}


def test_recovery_skips_active_lease_and_reclaims_expired(
    session_factory: sessionmaker[Session],
) -> None:
    active_id = make_run(session_factory, "/data/media/active.wav")
    dead_id = make_run(session_factory, "/data/media/dead.wav")
    advance_to(session_factory, active_id, Stage.TRANSCRIBE)
    advance_to(session_factory, dead_id, Stage.DIARIZE_EMBED)
    now = datetime.now(tz=UTC)
    with session_factory() as session:
        session.add(
            StageRun(
                pipeline_run_id=active_id,
                stage=Stage.TRANSCRIBE.value,
                status=StageStatus.RUNNING.value,
                attempt=1,
                worker_id="healthy-worker",
                lease_expires_at=now + timedelta(hours=1),
            )
        )
        session.add(
            StageRun(
                pipeline_run_id=dead_id,
                stage=Stage.DIARIZE_EMBED.value,
                status=StageStatus.RUNNING.value,
                attempt=1,
                worker_id="dead-worker",
                lease_expires_at=now - timedelta(minutes=5),
            )
        )
        session.commit()

    with session_factory() as session:
        recovered = recover_interrupted_runs(session)
        session.commit()
    assert recovered == [dead_id]  # healthy worker's run untouched

    with session_factory() as session:
        active = session.get(PipelineRun, active_id)
        dead = session.get(PipelineRun, dead_id)
        assert active is not None and dead is not None
        assert active.status == RunStatus.RUNNING.value
        assert dead.status == RunStatus.QUEUED.value
        assert dead.current_stage == Stage.DIARIZE_EMBED.value
        # the interrupted attempt is recorded, not erased
        claim = session.execute(
            select(StageRun).where(StageRun.pipeline_run_id == dead_id)
        ).scalar_one()
        assert claim.status == StageStatus.FAILED.value
        assert claim.error is not None and "lease expired" in claim.error

    executed: list[Stage] = []

    def tracking(stage: Stage) -> StageFn:
        def fn(session: Session, rid: uuid.UUID) -> None:
            executed.append(stage)

        return fn

    final = execute_run(session_factory, dead_id, {s: tracking(s) for s in Stage})
    assert final.status is RunStatus.COMPLETED
    # resumed AT the interrupted stage — earlier stages not re-run, none skipped
    assert executed == [Stage.DIARIZE_EMBED, Stage.ENHANCE_MATCH, Stage.FINALIZE]
