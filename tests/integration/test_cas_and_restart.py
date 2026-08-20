"""CAS conflict, stage-claim, and crash/restart semantics against real Postgres."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from voxint.db.models import (
    GPU_SEGMENT,
    POST_SEGMENT,
    STAGE_ORDER,
    MediaItem,
    PipelineRun,
    RunStatus,
    Stage,
    StageRun,
    StageStatus,
)
from voxint.domain_packs.base import load_default
from voxint.ingest import cancel_run
from voxint.pipeline.engine import (
    StageFailedError,
    StageFn,
    close_cancelled_run_claims,
    execute_run,
    recover_interrupted_runs,
    submit,
)
from voxint.pipeline.transitions import (
    StaleRevisionError,
    cas_update_run,
    next_stage,
    snapshot,
)


def make_run(session_factory: sessionmaker[Session], path: str) -> uuid.UUID:
    with session_factory() as session:
        media = MediaItem(source_path=path)
        session.add(media)
        session.flush()
        run = submit(session, media.id, domain_pack=load_default().to_mapping())
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


def test_two_segments_handoff_and_resume_without_wrong_lane_claims(
    session_factory: sessionmaker[Session],
) -> None:
    run_id = make_run(session_factory, "/data/media/two-segments.wav")

    handed_off = execute_run(session_factory, run_id, NOOP_FNS, stages=GPU_SEGMENT)
    assert handed_off.status is RunStatus.QUEUED
    assert handed_off.current_stage is Stage.ENHANCE_MATCH

    with session_factory() as session:
        claims = (
            session.execute(
                select(StageRun)
                .where(StageRun.pipeline_run_id == run_id)
                .order_by(StageRun.started_at, StageRun.id)
            )
            .scalars()
            .all()
        )
        assert [Stage(claim.stage) for claim in claims] == list(STAGE_ORDER[:4])
        assert all(claim.status == StageStatus.COMPLETED.value for claim in claims)

    wrong_lane = execute_run(session_factory, run_id, NOOP_FNS, stages=GPU_SEGMENT)
    assert wrong_lane == handed_off
    with session_factory() as session:
        claims = (
            session.execute(
                select(StageRun).where(StageRun.pipeline_run_id == run_id)
            )
            .scalars()
            .all()
        )
        assert len(claims) == 4

    completed = execute_run(session_factory, run_id, NOOP_FNS, stages=POST_SEGMENT)
    assert completed.status is RunStatus.COMPLETED
    assert completed.current_stage is None

    with session_factory() as session:
        claims = (
            session.execute(
                select(StageRun).where(StageRun.pipeline_run_id == run_id)
            )
            .scalars()
            .all()
        )
        assert len(claims) == len(STAGE_ORDER)
        assert all(claim.status == StageStatus.COMPLETED.value for claim in claims)


def test_cancel_between_segments_prevents_post_lane_resurrection(
    session_factory: sessionmaker[Session],
) -> None:
    run_id = make_run(session_factory, "/data/media/cancel-between-segments.wav")
    handed_off = execute_run(session_factory, run_id, NOOP_FNS, stages=GPU_SEGMENT)
    assert handed_off.status is RunStatus.QUEUED
    assert handed_off.current_stage is Stage.ENHANCE_MATCH

    with session_factory() as session:
        cancel_run(session, run_id)
        session.commit()

    result = execute_run(session_factory, run_id, NOOP_FNS, stages=POST_SEGMENT)
    assert result.status is RunStatus.CANCELLED
    assert result.current_stage is Stage.ENHANCE_MATCH

    with session_factory() as session:
        claims = (
            session.execute(
                select(StageRun).where(StageRun.pipeline_run_id == run_id)
            )
            .scalars()
            .all()
        )
        assert len(claims) == len(GPU_SEGMENT)


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


# --- cooperative cancellation mid-stage (issue #5) ----------------------------


def test_cancel_during_stage_body_stops_cleanly(
    session_factory: sessionmaker[Session],
) -> None:
    """An operator cancels while a stage body runs: the post-stage advance CAS
    loses, but execute_run stops cleanly (no StaleRevisionError escapes), the run
    stays CANCELLED at its stage, later stages never run, and the abandoned claim
    is closed SKIPPED (not left RUNNING forever)."""
    run_id = make_run(session_factory, "/data/media/cancel-mid.wav")
    executed: list[Stage] = []

    def cancel_then_return(stage: Stage) -> StageFn:
        def fn(session: Session, rid: uuid.UUID) -> None:
            executed.append(stage)
            if stage is Stage.TRANSCRIBE:
                # Simulate a concurrent operator cancel landing mid-stage, on a
                # SEPARATE session/transaction (as the API would).
                with session_factory() as other:
                    cancel_run(other, rid)
                    other.commit()

        return fn

    result = execute_run(
        session_factory, run_id, {s: cancel_then_return(s) for s in Stage}
    )
    assert result.status is RunStatus.CANCELLED
    assert result.current_stage is Stage.TRANSCRIBE
    # ACQUIRE, PREPARE ran, then TRANSCRIBE — nothing past the cancel point.
    assert executed == [Stage.ACQUIRE, Stage.PREPARE, Stage.TRANSCRIBE]

    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.CANCELLED.value
        assert run.current_stage == Stage.TRANSCRIBE.value
        claim = session.execute(
            select(StageRun).where(
                StageRun.pipeline_run_id == run_id,
                StageRun.stage == Stage.TRANSCRIBE.value,
            )
        ).scalar_one()
        # Not left RUNNING — the abandoned attempt is honestly closed.
        assert claim.status == StageStatus.SKIPPED.value
        assert claim.error == "cancelled before commit"


def test_cancel_during_failing_stage_body_stops_cleanly(
    session_factory: sessionmaker[Session],
) -> None:
    """A stage fails AND the run is cancelled concurrently: the FAILED CAS loses,
    and cancellation wins — execute_run returns CANCELLED rather than raising
    StageFailedError, and the claim is closed SKIPPED."""
    run_id = make_run(session_factory, "/data/media/cancel-fail.wav")

    def cancel_then_raise(session: Session, rid: uuid.UUID) -> None:
        with session_factory() as other:
            cancel_run(other, rid)
            other.commit()
        raise RuntimeError("stage blew up as it was cancelled")

    fns = dict(NOOP_FNS)
    fns[Stage.TRANSCRIBE] = cancel_then_raise

    result = execute_run(session_factory, run_id, fns)
    assert result.status is RunStatus.CANCELLED

    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.CANCELLED.value
        claim = session.execute(
            select(StageRun).where(
                StageRun.pipeline_run_id == run_id,
                StageRun.stage == Stage.TRANSCRIBE.value,
            )
        ).scalar_one()
        assert claim.status == StageStatus.SKIPPED.value
        assert claim.error == "cancelled before commit"


def test_recovery_closes_orphaned_claim_on_cancelled_run(
    session_factory: sessionmaker[Session],
) -> None:
    """Crash-window backstop: a worker died AFTER a cancel committed but BEFORE
    it closed its own claim, leaving a RUNNING claim on a CANCELLED run.
    recover_interrupted_runs only scans RUNNING *runs* so it never touches it;
    close_cancelled_run_claims (run from the beat) closes it SKIPPED and never
    requeues the terminal run."""
    run_id = make_run(session_factory, "/data/media/orphan-claim.wav")
    advance_to(session_factory, run_id, Stage.TRANSCRIBE)
    # A live worker's RUNNING claim, with an unexpired lease (so recovery would
    # never steal it even if the run were RUNNING).
    with session_factory() as session:
        session.add(
            StageRun(
                pipeline_run_id=run_id,
                stage=Stage.TRANSCRIBE.value,
                attempt=1,
                status=StageStatus.RUNNING.value,
                worker_id="dead-worker",
                lease_expires_at=datetime.now(UTC) + timedelta(seconds=300),
            )
        )
        session.commit()
    # Operator cancels; the worker never runs its cleanup (simulated crash).
    with session_factory() as session:
        cancel_run(session, run_id)
        session.commit()

    # recover_interrupted_runs ignores the cancelled run entirely...
    with session_factory() as session:
        assert recover_interrupted_runs(session) == []
        claim = session.execute(
            select(StageRun).where(StageRun.pipeline_run_id == run_id)
        ).scalar_one()
        assert claim.status == StageStatus.RUNNING.value  # still orphaned

    # ...the cancel sweep closes it.
    with session_factory() as session:
        closed = close_cancelled_run_claims(session)
        session.commit()
        assert run_id in closed

    with session_factory() as session:
        claim = session.execute(
            select(StageRun).where(StageRun.pipeline_run_id == run_id)
        ).scalar_one()
        assert claim.status == StageStatus.SKIPPED.value
        assert claim.error == "cancelled before commit"
        assert claim.finished_at is not None
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.CANCELLED.value  # not requeued


def test_lost_advance_cas_from_non_cancel_still_raises(
    session_factory: sessionmaker[Session],
) -> None:
    """The cancel handling is NARROW: a post-stage CAS lost to something other
    than a cancel (here a duplicate worker advancing the run) must still raise
    StaleRevisionError — we never silently swallow a genuine race."""
    run_id = make_run(session_factory, "/data/media/dup-worker.wav")

    def advance_elsewhere(session: Session, rid: uuid.UUID) -> None:
        # Simulate a stray duplicate worker completing this stage first: advance
        # the run to the next stage (revision bumps, status stays RUNNING). The
        # worker's own advance CAS then loses — and must NOT be swallowed.
        with session_factory() as other:
            held = snapshot(other.get(PipelineRun, rid))  # type: ignore[arg-type]
            nxt = next_stage(held.current_stage)
            assert nxt is not None
            cas_update_run(other, held, status=RunStatus.RUNNING, current_stage=nxt)
            other.commit()

    fns = dict(NOOP_FNS)
    fns[Stage.TRANSCRIBE] = advance_elsewhere

    with pytest.raises(StaleRevisionError):
        execute_run(session_factory, run_id, fns)

    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.RUNNING.value  # not cancelled, not failed
