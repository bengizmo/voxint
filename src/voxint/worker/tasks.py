"""Celery tasks: thin wrappers over the stage engine.

Orchestration philosophy (design-gated with codex, P3):

- One ``run_pipeline`` task drives a run through ALL stages via
  ``execute_run``. Task-per-stage would open an unclaimed window between
  stage handoffs that the recovery sweep misreads as a crash; the P1 engine
  already resumes an interrupted run at its current stage, so the finer
  granularity buys nothing on a single box.
- Transient failures (a saturated service, a dead socket) leave an honest
  FAILED stage attempt in the ledger; the task then CAS-requeues the run and
  retries itself with backoff. The attempt budget is counted from the
  persisted ``stage_runs`` ledger, so broker loss or worker restarts can
  never reset it. Deterministic failures stay FAILED for the failure lane.
- A beat sweep requeues runs whose stage lease expired (dead workers) and
  re-enqueues stale QUEUED runs whose task evaporated with the broker.
"""

import random
import uuid
from datetime import UTC, datetime, timedelta
from functools import lru_cache

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, sessionmaker

from voxint.clients.errors import ServiceError
from voxint.config import get_settings
from voxint.db.models import PipelineRun, RunStatus, Stage, StageRun, StageStatus
from voxint.db.session import build_engine, build_session_factory
from voxint.pipeline.engine import (
    INTERRUPTED_PREFIX,
    StageFailedError,
    StageFn,
    execute_run,
    recover_interrupted_runs,
)
from voxint.pipeline.stages.context import build_stage_context, build_stage_fns
from voxint.pipeline.transitions import (
    InvalidTransitionError,
    RunSnapshot,
    StaleRevisionError,
    cas_update_run,
)
from voxint.worker.app import app


@lru_cache(maxsize=1)
def _runtime() -> tuple[sessionmaker[Session], dict[Stage, StageFn]]:
    """Per-process singletons: engine pool, HTTP clients, stage map."""
    settings = get_settings()
    factory = build_session_factory(build_engine(settings.database_url))
    return factory, build_stage_fns(build_stage_context(settings))


def retryable_cause(exc: StageFailedError) -> bool:
    return isinstance(exc.cause, ServiceError) and exc.cause.retryable


def backoff_seconds(attempts: int, base: float, cap: float) -> float:
    """Exponential in completed attempts, capped; jitter is the caller's."""
    return float(min(base * 2 ** max(attempts - 1, 0), cap))


def stage_attempts(session: Session, run_id: uuid.UUID, stage: Stage) -> int:
    """Transient-failure attempts recorded in the ledger — the restart-proof
    retry budget.

    Interruption attempts (lease expiry: worker death, OOM, redeploy) are
    excluded — infra churn must not eat the budget for *service* failures.
    Crash loops are bounded separately by the recovery sweep's own ceiling.
    """
    return int(
        session.execute(
            select(func.count())
            .select_from(StageRun)
            .where(
                StageRun.pipeline_run_id == run_id,
                StageRun.stage == stage.value,
                StageRun.status == StageStatus.FAILED.value,
                or_(
                    StageRun.error.is_(None),
                    ~StageRun.error.like(f"{INTERRUPTED_PREFIX}%"),
                ),
            )
        ).scalar_one()
    )


def requeue_failed_stage(factory: sessionmaker[Session], failed: RunSnapshot) -> bool:
    """CAS FAILED → QUEUED against the exact revision our failure produced.

    Matching on (FAILED, stage) alone would be an ABA bug: if anything moved
    the run since — an operator requeue, a newer attempt failing at the same
    stage — the revision differs, the CAS misses, and this callback declines.
    """
    if failed.status is not RunStatus.FAILED or failed.current_stage is None:
        return False
    with factory() as session:
        try:
            cas_update_run(
                session, failed, status=RunStatus.QUEUED, current_stage=failed.current_stage
            )
        except (StaleRevisionError, InvalidTransitionError):
            return False
        session.commit()
        return True


# ignore_result=True: nothing consumes run_pipeline's return value, so storing
# it wastes a Redis write. It also matters for degraded publishing — with a Redis
# result backend, enqueuing onto a DOWN broker otherwise raises a vague
# RuntimeError from the result consumer's reconnect loop instead of the broker's
# own kombu OperationalError, which is what the API's _publish_or_defer catches.
# Setting it on the task (not just per apply_async) makes self.retry(), the CLI's
# .delay(), and the recovery sweep's re-publish all inherit the policy — a
# per-call flag is not propagated by Celery's Task.retry().
@app.task(bind=True, name="voxint.run_pipeline", max_retries=None, ignore_result=True)  # type: ignore[misc, untyped-decorator, unused-ignore]
def run_pipeline(self: object, run_id_str: str) -> str:
    """Advance one run to COMPLETED (or park it FAILED / adjudication-paused)."""
    factory, stage_fns = _runtime()
    run_id = uuid.UUID(run_id_str)
    try:
        final = execute_run(factory, run_id, stage_fns)
    except StageFailedError as exc:
        settings = get_settings()
        if not retryable_cause(exc) or exc.failed_snapshot is None:
            raise  # deterministic — the failure lane owns it now
        with factory() as session:
            attempts = stage_attempts(session, run_id, exc.stage)
        if attempts >= settings.stage_max_attempts:
            raise  # transient budget exhausted; stays FAILED, honestly
        if not requeue_failed_stage(factory, exc.failed_snapshot):
            return "lost-requeue-race"
        delay = backoff_seconds(
            attempts,
            settings.retry_backoff_base_seconds,
            settings.retry_backoff_max_seconds,
        )
        raise self.retry(  # type: ignore[attr-defined]  # noqa: B904 — celery Retry carries exc=
            exc=exc, countdown=delay + random.uniform(0, delay * 0.1)
        )
    return final.status.value


@app.task(name="voxint.recovery_sweep")  # type: ignore[misc, untyped-decorator, unused-ignore]
def recovery_sweep() -> dict[str, int]:
    """Reclaim expired-lease runs and re-enqueue stale QUEUED runs.

    Duplicate enqueues are safe — stage claims and CAS arbitrate — so this errs
    toward re-publishing. The staleness grace keeps it from stepping on
    freshly-submitted runs and pending retry countdowns.
    """
    settings = get_settings()
    factory, _ = _runtime()
    with factory() as session:
        recovered = recover_interrupted_runs(session, max_attempts=settings.stage_max_attempts)
        session.commit()
    cutoff = datetime.now(tz=UTC) - timedelta(seconds=settings.queued_run_stale_seconds)
    with factory() as session:
        stale_queued = (
            session.execute(
                select(PipelineRun.id).where(
                    PipelineRun.status == RunStatus.QUEUED.value,
                    PipelineRun.updated_at < cutoff,
                )
            )
            .scalars()
            .all()
        )
    for rid in {*recovered, *stale_queued}:
        run_pipeline.delay(str(rid))
    return {"recovered": len(recovered), "stale_queued": len(stale_queued)}
