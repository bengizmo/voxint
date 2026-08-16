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

import logging
import random
import uuid
from datetime import UTC, datetime, timedelta
from functools import lru_cache

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, sessionmaker

from voxint import app_settings
from voxint.clients.errors import ServiceError
from voxint.clients.llm import HttpLLMClient
from voxint.config import Settings, get_settings
from voxint.db.models import PipelineRun, RunStatus, Stage, StageRun, StageStatus
from voxint.db.session import build_engine, build_session_factory
from voxint.enrichment import asset_jobs
from voxint.enrichment.research_jobs import execute_job
from voxint.media.reclaim import (
    ReclaimSummary,
    configured_tutorial_run_id,
    reclaim_expired_intermediates,
)
from voxint.notify.delivery import DeliverySummary, deliver_due, purge_expired_deliveries
from voxint.pipeline.engine import (
    INTERRUPTED_PREFIX,
    StageFailedError,
    close_cancelled_run_claims,
    execute_run,
    recover_interrupted_runs,
)
from voxint.pipeline.stages.context import (
    StageContext,
    apply_run_preferences,
    build_stage_context,
    build_stage_fns,
    resolve_run_preferences,
)
from voxint.pipeline.transitions import (
    InvalidTransitionError,
    RunSnapshot,
    StaleRevisionError,
    cas_update_run,
)
from voxint.worker.app import app

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _runtime() -> tuple[sessionmaker[Session], StageContext]:
    """Per-process singletons: engine pool + the base StageContext (transport
    clients + domain pack). ``run_pipeline`` layers each run's app_settings
    preferences onto this base, so wizard edits take effect with no restart."""
    settings = get_settings()
    factory = build_session_factory(build_engine(settings.database_url))
    return factory, build_stage_context(settings)


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
    factory, base_ctx = _runtime()
    settings = get_settings()
    run_id = uuid.UUID(run_id_str)
    # Snapshot the wizard's preferences once per invocation and layer them onto the
    # process-cached base context, so a settings edit lands on the next run with no
    # worker restart. A stage retry inside this execute_run reuses this snapshot;
    # the next run_pipeline invocation re-reads the row.
    with factory() as session:
        row = app_settings.get_app_settings(session)
        prefs = resolve_run_preferences(row, settings)
        # Resolve the effective key (a UI-stored row value wins over env) inside the
        # session, so it reaches the per-run HttpLLMClient the same no-restart way as
        # base_url/model. Kept off RunPreferences (which has a repr); passed as a str.
        llm_api_key = app_settings.resolve_effective_llm_api_key(row, settings)
    ctx = apply_run_preferences(base_ctx, settings, prefs, llm_api_key=llm_api_key)
    stage_fns = build_stage_fns(ctx)
    try:
        final = execute_run(factory, run_id, stage_fns, settings=settings)
    except StageFailedError as exc:
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
    else:
        if final.status is RunStatus.COMPLETED:
            _autogenerate_run_assets(factory, run_id, settings)
        return final.status.value
    finally:
        # apply_run_preferences may build a per-run HttpLLMClient that owns its
        # httpx.Client; close it on every exit path (success, retry, hard fail) so
        # a long-lived worker doesn't leak a connection pool per run. A stage retry
        # re-enters run_pipeline and builds a fresh one.
        if isinstance(ctx.llm, HttpLLMClient):
            ctx.llm.close()


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
        recovered = recover_interrupted_runs(
            session, max_attempts=settings.stage_max_attempts, settings=settings
        )
        session.commit()
    # Backstop for cooperative cancellation (issue #5): if a worker died between
    # a cancel commit and its own claim cleanup, the stage claim is orphaned
    # RUNNING on a CANCELLED (terminal) run that recover_interrupted_runs never
    # scans. Close those claims SKIPPED; never requeue a cancelled run.
    with factory() as session:
        cancelled_claims = close_cancelled_run_claims(session)
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
    return {
        "recovered": len(recovered),
        "stale_queued": len(stale_queued),
        "cancelled_claims_closed": len(cancelled_claims),
    }


@app.task(name="voxint.gc_sweep")  # type: ignore[misc, untyped-decorator, unused-ignore]
def gc_sweep() -> dict[str, int]:
    """Reclaim expired normalized-audio intermediates for old terminal runs.

    File reclamation only (issue #15): unlink the WAV, stamp the artifact row,
    keep everything else. OFF unless ``media_retention_enabled`` — the gate is
    re-checked here (not just at beat registration) so a stale schedule entry
    can never act. Safe under overlapping sweeps (FOR UPDATE SKIP LOCKED).
    """
    settings = get_settings()
    empty = ReclaimSummary()
    if not settings.media_retention_enabled:
        return empty.as_dict()
    factory, _ = _runtime()
    cutoff = datetime.now(tz=UTC) - timedelta(seconds=settings.media_retention_seconds)
    with factory() as session:
        tutorial_run_id = configured_tutorial_run_id(session)
        summary = reclaim_expired_intermediates(
            session,
            media_root=settings.media_root,
            cutoff=cutoff,
            batch_limit=settings.gc_batch_limit,
            tutorial_run_id=tutorial_run_id,
        )
    logger.info("gc_sweep %s", summary.as_dict())
    return summary.as_dict()


@app.task(name="voxint.notify_sweep")  # type: ignore[misc, untyped-decorator, unused-ignore]
def notify_sweep() -> dict[str, int]:
    """Deliver due run-webhook rows (issue #12).

    Claims a batch of pending/lapsed ``notification_deliveries`` rows and POSTs
    each as a signed webhook outside any DB transaction. OFF unless
    ``notify_enabled`` — the gate is re-checked here (not just at beat
    registration) so a stale schedule entry can never send. Safe under
    overlapping sweeps (FOR UPDATE SKIP LOCKED + a per-claim lease).
    """
    settings = get_settings()
    empty = DeliverySummary()
    if not settings.notify_enabled:
        return {**empty.as_dict(), "purged": 0}
    factory, _ = _runtime()
    summary = deliver_due(factory, settings)
    purged = purge_expired_deliveries(factory, settings)
    result = {**summary.as_dict(), "purged": purged}
    logger.info("notify_sweep %s", result)
    return result


@app.task(name="voxint.generate_run_asset", ignore_result=True)  # type: ignore[misc, untyped-decorator, unused-ignore]
def generate_run_asset(job_id_str: str) -> None:
    """Run one queued run-asset generation job (issue #41).

    No Celery retries on purpose (the research_speaker precedent): failures
    land as honest, bounded error text on the job row for the operator to
    see. A duplicate delivery no-ops on the guarded queued→running claim.
    """
    factory, _ = _runtime()
    asset_jobs.execute_job(factory, uuid.UUID(job_id_str), settings=get_settings())


def _autogenerate_run_assets(
    factory: sessionmaker[Session], run_id: uuid.UUID, settings: Settings
) -> None:
    """Opt-in post-finalize step: enqueue asset jobs for kinds that are
    missing or stale. Best-effort by contract — a completed run is COMPLETED
    whatever happens here, so every failure is logged and swallowed."""
    if not (
        settings.enrichment_run_assets_autogenerate
        and asset_jobs.run_asset_gates_open(settings)
    ):
        return
    try:
        with factory() as session:
            needed = asset_jobs.kinds_needing_generation(session, run_id)
            if not needed:
                return
            created, _ = asset_jobs.create_jobs(
                session, pipeline_run_id=run_id, kinds=needed, settings=settings
            )
            session.commit()
            job_ids = [str(job.id) for job in created]
        for job_id in job_ids:
            generate_run_asset.delay(job_id)
    except Exception:
        logger.exception("post-finalize run-asset enqueue failed for run %s", run_id)


@app.task(name="voxint.research_speaker", ignore_result=True)  # type: ignore[misc, untyped-decorator, unused-ignore]
def research_speaker(job_id_str: str) -> None:
    """Run one queued web-research job (issue #40).

    No Celery retries on purpose: the loop is non-deterministic and its
    failures land as honest, bounded error text on the job row for the
    operator to see — hidden re-execution would be worse than a visible
    failure. A duplicate delivery no-ops on the guarded queued→running claim.
    """
    factory, _ = _runtime()
    execute_job(factory, uuid.UUID(job_id_str), settings=get_settings())
