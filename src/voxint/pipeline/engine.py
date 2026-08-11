"""Minimal stage engine: drives a pipeline run through the canonical stages.

P1 scope — the orchestration contract, not the science. Stage bodies are pure
functions injected by the caller; the engine owns state transitions, stage_run
bookkeeping, CAS discipline, and crash recovery. P3 swaps the trivial stage
bodies for the real preprocess/transcribe/diarize implementations without
touching this file.

Concurrency contract:

- Before a stage body runs, the worker **claims** the stage by committing a
  RUNNING ``stage_runs`` row carrying its worker id and a lease. The
  ``(pipeline_run_id, stage, attempt)`` unique constraint arbitrates ties, so
  two workers can never both believe they own an attempt.
- Success and failure update that exact claim row and CAS-advance the run in
  one transaction with the stage's own DB effects.
- Recovery touches only runs whose claim lease has expired — a healthy worker
  mid-transcription is never swept. Stage bodies remain at-least-once for
  non-transactional effects (filesystem, GPU calls) and must be idempotent.
"""

import os
import socket
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from voxint.config import get_settings
from voxint.db.models import PipelineRun, RunStatus, Stage, StageRun, StageStatus
from voxint.pipeline.transitions import RunSnapshot, cas_update_run, next_stage, snapshot

# A stage body receives the session and the run id; whatever it persists is its own business.
StageFn = Callable[[Session, uuid.UUID], None]


class StageFailedError(Exception):
    def __init__(self, stage: Stage, cause: Exception) -> None:
        super().__init__(f"stage {stage.value} failed: {cause}")
        self.stage = stage
        self.cause = cause


def default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def submit(session: Session, media_item_id: uuid.UUID) -> PipelineRun:
    run = PipelineRun(media_item_id=media_item_id, status=RunStatus.QUEUED.value)
    session.add(run)
    session.flush()
    return run


def _latest_claim(session: Session, run_id: uuid.UUID, stage: Stage) -> StageRun | None:
    return session.execute(
        select(StageRun)
        .where(StageRun.pipeline_run_id == run_id, StageRun.stage == stage.value)
        .order_by(StageRun.attempt.desc())
        .limit(1)
    ).scalar_one_or_none()


def _claim_stage(
    session_factory: sessionmaker[Session],
    run_id: uuid.UUID,
    stage: Stage,
    worker_id: str,
    lease_seconds: int,
) -> uuid.UUID | None:
    """Commit a RUNNING stage_run claim; None if another worker holds an active one."""
    with session_factory() as session:
        latest = _latest_claim(session, run_id, stage)
        now = datetime.now(tz=UTC)
        if (
            latest is not None
            and latest.status == StageStatus.RUNNING.value
            and latest.lease_expires_at is not None
            and latest.lease_expires_at > now
        ):
            return None  # actively owned by someone else
        claim = StageRun(
            pipeline_run_id=run_id,
            stage=stage.value,
            status=StageStatus.RUNNING.value,
            attempt=(latest.attempt if latest else 0) + 1,
            worker_id=worker_id,
            lease_expires_at=now + timedelta(seconds=lease_seconds),
        )
        session.add(claim)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            return None  # lost the claim race to a concurrent worker
        return claim.id


def _finish_claim(
    session: Session,
    claim_id: uuid.UUID,
    status: StageStatus,
    error: str | None = None,
) -> None:
    claim = session.get(StageRun, claim_id)
    assert claim is not None
    claim.status = status.value
    claim.finished_at = datetime.now(tz=UTC)
    claim.error = error


def execute_run(
    session_factory: sessionmaker[Session],
    run_id: uuid.UUID,
    stage_fns: dict[Stage, StageFn],
    *,
    worker_id: str | None = None,
    lease_seconds: int | None = None,
) -> RunSnapshot:
    """Advance a QUEUED (or requeued) run through all stages to COMPLETED.

    Returns the run snapshot as this worker last saw it; if another worker holds
    an active claim, returns immediately without executing anything. Each stage
    executes in its own transaction: stage effects, the claim update, and the
    CAS advance commit atomically, so a crash between stages leaves a
    consistent, resumable run.
    """
    worker = worker_id or default_worker_id()
    lease = lease_seconds if lease_seconds is not None else get_settings().stage_lease_seconds

    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        if run is None:
            raise LookupError(f"no pipeline run {run_id}")
        held = snapshot(run)
        if held.status is RunStatus.QUEUED:
            # A requeued run retries its interrupted stage; a fresh run starts at the top.
            stage = held.current_stage or next_stage(None)
            held = cas_update_run(session, held, status=RunStatus.RUNNING, current_stage=stage)
            session.commit()

    while True:
        with session_factory() as session:
            run = session.get(PipelineRun, run_id)
            assert run is not None
            held = snapshot(run)
            if held.status is not RunStatus.RUNNING or held.current_stage is None:
                return held
            stage = held.current_stage

        claim_id = _claim_stage(session_factory, run_id, stage, worker, lease)
        if claim_id is None:
            return held  # another worker owns this stage right now

        with session_factory() as session:
            try:
                stage_fns[stage](session, run_id)
            except Exception as exc:
                session.rollback()
                _finish_claim(session, claim_id, StageStatus.FAILED, str(exc))
                cas_update_run(
                    session, held, status=RunStatus.FAILED, current_stage=stage, error=str(exc)
                )
                session.commit()
                raise StageFailedError(stage, exc) from exc
            _finish_claim(session, claim_id, StageStatus.COMPLETED)
            upcoming = next_stage(stage)
            if upcoming is None:
                held = cas_update_run(
                    session, held, status=RunStatus.COMPLETED, current_stage=None
                )
                session.commit()
                return held
            cas_update_run(session, held, status=RunStatus.RUNNING, current_stage=upcoming)
            session.commit()


def recover_interrupted_runs(session: Session) -> list[uuid.UUID]:
    """Requeue RUNNING runs whose stage claim lease has expired (or never existed).

    Runs with an unexpired RUNNING claim belong to a live worker and are left
    alone. The interrupted attempt is marked failed so the ledger records it.
    """
    now = datetime.now(tz=UTC)
    recovered: list[uuid.UUID] = []
    runs = session.execute(
        select(PipelineRun).where(PipelineRun.status == RunStatus.RUNNING.value)
    ).scalars()
    for run in runs:
        held = snapshot(run)
        if held.current_stage is None:
            continue
        claim = _latest_claim(session, held.id, held.current_stage)
        if (
            claim is not None
            and claim.status == StageStatus.RUNNING.value
            and claim.lease_expires_at is not None
            and claim.lease_expires_at > now
        ):
            continue  # live worker — do not steal
        if claim is not None and claim.status == StageStatus.RUNNING.value:
            _finish_claim(session, claim.id, StageStatus.FAILED, "interrupted: lease expired")
        # RUNNING -> FAILED -> QUEUED keeps the transition map honest about what happened.
        held = cas_update_run(
            session,
            held,
            status=RunStatus.FAILED,
            current_stage=held.current_stage,
            error="interrupted: worker died mid-stage",
        )
        cas_update_run(session, held, status=RunStatus.QUEUED, current_stage=held.current_stage)
        recovered.append(held.id)
    return recovered
