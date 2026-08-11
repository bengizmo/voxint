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
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from voxint.config import get_settings
from voxint.db.models import PipelineRun, RunStatus, Stage, StageRun, StageStatus
from voxint.pipeline.transitions import (
    RunSnapshot,
    StaleRevisionError,
    cas_update_run,
    next_stage,
    snapshot,
)

# A stage body receives the session and the run id; whatever it persists is its own business.
StageFn = Callable[[Session, uuid.UUID], None]

# Error prefix marking attempts killed by lease expiry rather than the stage
# itself failing — the retry budget filters these out by prefix.
INTERRUPTED_PREFIX = "interrupted:"


class StageFailedError(Exception):
    def __init__(
        self, stage: Stage, cause: Exception, failed_snapshot: "RunSnapshot | None" = None
    ) -> None:
        super().__init__(f"stage {stage.value} failed: {cause}")
        self.stage = stage
        self.cause = cause
        # The run's state as this failure left it. A retry handler must CAS
        # against exactly this revision — matching on (FAILED, stage) alone is
        # an ABA bug: a *newer* failure at the same stage looks identical.
        self.failed_snapshot = failed_snapshot


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
    expected: RunSnapshot,
    stage: Stage,
    worker_id: str,
    lease_seconds: int,
) -> uuid.UUID | None:
    """Commit a RUNNING stage_run claim; None if another worker holds an active
    one or the run has moved past the snapshot this worker is executing for.

    The run row is locked while the claim is inserted: between a worker's CAS
    advance and its claim there is an unclaimed instant that recovery may
    legitimately requeue — verifying (status, stage, revision) under the lock
    ensures a worker never starts expensive work for an obsolete revision.
    """
    run_id = expected.id
    with session_factory() as session:
        run = session.execute(
            select(PipelineRun).where(PipelineRun.id == run_id).with_for_update()
        ).scalar_one_or_none()
        if (
            run is None
            or run.status != RunStatus.RUNNING.value
            or run.current_stage != stage.value
            or run.revision != expected.revision
        ):
            return None  # the run moved on without us
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


def default_stage_leases() -> dict[Stage, int]:
    """Per-stage lease budgets from settings.

    diarize_embed gets its own: one diarization call plus every embedding
    batch must fit inside the lease, or a healthy worker gets robbed mid-stage.
    """
    settings = get_settings()
    leases = {stage: settings.stage_lease_seconds for stage in Stage}
    leases[Stage.DIARIZE_EMBED] = settings.diarize_embed_lease_seconds
    return leases


def execute_run(
    session_factory: sessionmaker[Session],
    run_id: uuid.UUID,
    stage_fns: dict[Stage, StageFn],
    *,
    worker_id: str | None = None,
    lease_seconds: int | Mapping[Stage, int] | None = None,
) -> RunSnapshot:
    """Advance a QUEUED (or requeued) run through all stages to COMPLETED.

    Returns the run snapshot as this worker last saw it; if another worker holds
    an active claim, returns immediately without executing anything. Each stage
    executes in its own transaction: stage effects, the claim update, and the
    CAS advance commit atomically, so a crash between stages leaves a
    consistent, resumable run.
    """
    worker = worker_id or default_worker_id()
    if lease_seconds is None:
        leases: Mapping[Stage, int] | int = default_stage_leases()
    else:
        leases = lease_seconds

    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        if run is None:
            raise LookupError(f"no pipeline run {run_id}")
        held = snapshot(run)
        if held.status is RunStatus.QUEUED:
            # A requeued run retries its interrupted stage; a fresh run starts at the top.
            stage = held.current_stage or next_stage(None)
            try:
                held = cas_update_run(
                    session, held, status=RunStatus.RUNNING, current_stage=stage
                )
                session.commit()
            except StaleRevisionError:
                # Duplicate dispatch of the same QUEUED run (sweep + pending
                # retry can race here legitimately): the other invocation won
                # the entry CAS. Fall through and re-read — the claim step
                # arbitrates from there.
                session.rollback()

    while True:
        with session_factory() as session:
            run = session.get(PipelineRun, run_id)
            assert run is not None
            held = snapshot(run)
            if held.status is not RunStatus.RUNNING or held.current_stage is None:
                return held
            stage = held.current_stage

        lease = leases[stage] if isinstance(leases, Mapping) else leases
        claim_id = _claim_stage(session_factory, held, stage, worker, lease)
        if claim_id is None:
            return held  # another worker owns this stage right now

        with session_factory() as session:
            try:
                stage_fns[stage](session, run_id)
                # Surface deferred constraint violations HERE, not during the
                # claim/CAS bookkeeping below — an autoflush failure outside
                # this block would leave a RUNNING claim with no failure record.
                session.flush()
            except Exception as exc:
                session.rollback()
                _finish_claim(session, claim_id, StageStatus.FAILED, str(exc))
                failed = cas_update_run(
                    session, held, status=RunStatus.FAILED, current_stage=stage, error=str(exc)
                )
                session.commit()
                raise StageFailedError(stage, exc, failed) from exc
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


def recover_interrupted_runs(
    session: Session, *, max_attempts: int | None = None
) -> list[uuid.UUID]:
    """Requeue RUNNING runs whose stage claim lease has expired (or never existed).

    Runs with an unexpired RUNNING claim belong to a live worker and are left
    alone. The interrupted attempt is marked failed so the ledger records it.
    With ``max_attempts``, a run whose current stage already burned that many
    attempts is parked FAILED instead of requeued — otherwise repeated worker
    deaths grant unlimited retries that the retry path's budget never sees.
    A run that another worker moves mid-sweep is skipped, not fatal.
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
            _finish_claim(
                session, claim.id, StageStatus.FAILED, f"{INTERRUPTED_PREFIX} lease expired"
            )
        try:
            # RUNNING -> FAILED -> QUEUED keeps the transition map honest.
            held = cas_update_run(
                session,
                held,
                status=RunStatus.FAILED,
                current_stage=held.current_stage,
                error=f"{INTERRUPTED_PREFIX} worker died mid-stage",
            )
            if max_attempts is not None and claim is not None and claim.attempt >= max_attempts:
                continue  # budget exhausted — parked FAILED for the failure lane
            cas_update_run(
                session, held, status=RunStatus.QUEUED, current_stage=held.current_stage
            )
        except StaleRevisionError:
            continue  # someone else moved it mid-sweep; their view wins
        recovered.append(held.id)
    return recovered
