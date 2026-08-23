"""Celery tasks: thin wrappers over the stage engine.

Orchestration philosophy (two execution lanes over the P1 stage engine):

- ``run_pipeline`` drives the GPU segment through DIARIZE_EMBED;
  ``finish_pipeline`` drives ENHANCE_MATCH + FINALIZE. This one boundary lets
  remote LLM work overlap the next run's GPU work without turning every stage
  into a broker-level task.
- The boundary is durable DB state: completion of DIARIZE_EMBED and parking at
  QUEUED/ENHANCE_MATCH commit together. That is exactly the state the existing
  stale-QUEUED sweep already republishes, so a crash after commit cannot create
  the unclaimed RUNNING window that originally ruled out task-per-stage.
- Task-per-stage remains rejected: five additional handoffs add recovery and
  routing ceremony without a demonstrated scheduling benefit on one box.
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
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, sessionmaker

from voxint import app_settings
from voxint.clients.errors import ServiceError
from voxint.clients.llm import HttpLLMClient
from voxint.config import Settings, get_settings
from voxint.db.models import (
    GPU_SEGMENT,
    POST_SEGMENT,
    PipelineRun,
    RunStatus,
    Stage,
    StageRun,
    StageStatus,
)
from voxint.db.session import build_engine, build_session_factory
from voxint.domain_packs.registry import domain_pack_from_snapshot
from voxint.embeddings.onnx_embedder import minilm_artifacts_available
from voxint.enrichment import asset_jobs, embedding_jobs
from voxint.enrichment.research_jobs import execute_job
from voxint.ingest.watch import sweep_watch_folders
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
def _drive_segment(task: object, run_id_str: str, segment: frozenset[Stage]) -> str:
    """Drive exactly one execution lane with the shared retry/settings policy."""
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
        # Whether the scoped bundled local model is the active enhancement endpoint
        # (issue #67): resolved in-session like the key so it lands on the next run
        # with no restart. When active, enhancement routes to the keyless bundled
        # endpoint and its name_hints are dropped.
        bundled = app_settings.llm_bundled_active(row, settings)
        # The run's frozen domain-pack snapshot (issue #11); NULL for a legacy run.
        # Copy the run's per-run scalars out INSIDE the session, the same way as the
        # pack snapshot: reading them off run_row after the session closes would hit
        # a detached instance (and, if anything in this block ever commits, an
        # expired attribute) rather than the loaded value.
        run_row = session.get(PipelineRun, run_id)
        pack_snapshot = run_row.domain_pack if run_row is not None else None
        max_speakers_hint = (
            run_row.diarization_max_speakers if run_row is not None else None
        )
        num_speakers_hint = (
            run_row.diarization_num_speakers if run_row is not None else None
        )
    pack = domain_pack_from_snapshot(pack_snapshot, settings)
    ctx = apply_run_preferences(
        base_ctx, settings, prefs, pack, llm_api_key=llm_api_key, bundled=bundled
    )
    # Per-run diarization speaker-count hint (issue #128), frozen on the run at
    # submit. A stored max overrides the install-wide ceiling already on ctx; a
    # stored exact count pins pyannote to that many speakers. NULL columns leave
    # the settings default in place (a legacy run, or one with no hint).
    if max_speakers_hint is not None or num_speakers_hint is not None:
        ctx = replace(
            ctx,
            diarization_max_speakers=(
                max_speakers_hint
                if max_speakers_hint is not None
                else ctx.diarization_max_speakers
            ),
            diarization_num_speakers=num_speakers_hint,
        )
    stage_fns = build_stage_fns(ctx)
    try:
        final = execute_run(factory, run_id, stage_fns, settings=settings, stages=segment)
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
        raise task.retry(  # type: ignore[attr-defined]  # noqa: B904 — celery Retry carries exc=
            exc=exc, countdown=delay + random.uniform(0, delay * 0.1)
        )
    else:
        if final.status is RunStatus.COMPLETED and segment == POST_SEGMENT:
            # Only the owning post lane performs completion side effects. A
            # late/redelivered GPU task may observe an already-COMPLETED row;
            # treating that observation as completion would enqueue assets twice.
            _autogenerate_run_assets(factory, run_id, settings)
            _autogenerate_segment_embeddings(factory, run_id, settings)
        elif final.status is RunStatus.QUEUED and final.current_stage in POST_SEGMENT:
            # This covers both the first GPU→post handoff and a duplicate GPU
            # delivery observing an already-parked run. Re-publishing is safe:
            # the post task's entry CAS + stage claim arbitrate duplicates.
            _publish_finish_or_defer(run_id)
        return final.status.value
    finally:
        # apply_run_preferences may build a per-run HttpLLMClient that owns its
        # httpx.Client; close it on every exit path (success, retry, hard fail) so
        # a long-lived worker doesn't leak a connection pool per run. A stage retry
        # re-enters run_pipeline and builds a fresh one.
        if isinstance(ctx.llm, HttpLLMClient):
            ctx.llm.close()


@app.task(bind=True, name="voxint.run_pipeline", max_retries=None, ignore_result=True)  # type: ignore[misc, untyped-decorator, unused-ignore]
def run_pipeline(self: object, run_id_str: str) -> str:
    """Advance one run through the GPU execution segment."""
    return _drive_segment(self, run_id_str, GPU_SEGMENT)


@app.task(bind=True, name="voxint.finish_pipeline", max_retries=None, ignore_result=True)  # type: ignore[misc, untyped-decorator, unused-ignore]
def finish_pipeline(self: object, run_id_str: str) -> str:
    """Advance one run through enhancement/matching and finalization."""
    return _drive_segment(self, run_id_str, POST_SEGMENT)


def pipeline_task_for_stage(stage: Stage | None) -> Any:
    """Return the lane task for ``stage`` — the single routing decision every
    publisher (API, CLI, sweep, and handoff) must share."""
    return finish_pipeline if stage in POST_SEGMENT else run_pipeline


def _publish_finish_or_defer(run_id: uuid.UUID) -> bool:
    """Publish a durable post-lane handoff; defer only on broker outage."""
    from celery.exceptions import OperationalError

    try:
        finish_pipeline.apply_async((str(run_id),), ignore_result=True)
    except OperationalError:
        logger.warning(
            "post-pipeline enqueue deferred (broker unavailable); run %s stays "
            "QUEUED for the recovery sweep",
            run_id,
            exc_info=True,
        )
        return False
    return True


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
    publish_ids = {*recovered, *stale_queued}
    if publish_ids:
        # Re-read after recovery commits: routing from an earlier snapshot can
        # send QUEUED/ENHANCE_MATCH to the GPU lane, whose entry guard would
        # correctly no-op forever while every later sweep repeated the mistake.
        with factory() as session:
            resumable = session.execute(
                select(PipelineRun.id, PipelineRun.current_stage).where(
                    PipelineRun.id.in_(publish_ids),
                    PipelineRun.status == RunStatus.QUEUED.value,
                )
            ).all()
        # One unavailable broker must not abort the sweep midway through this
        # durable set. Each row stays QUEUED and becomes eligible again after the
        # stale grace, while the remaining rows still get their publish attempt.
        from celery.exceptions import OperationalError

        for rid, stage_value in resumable:
            stage = Stage(stage_value) if stage_value else None
            try:
                pipeline_task_for_stage(stage).apply_async(
                    (str(rid),), ignore_result=True
                )
            except OperationalError:
                logger.warning(
                    "recovery enqueue deferred (broker unavailable); run %s stays "
                    "QUEUED for a later sweep",
                    rid,
                    exc_info=True,
                )
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


def _publish_watch_run(run_id: uuid.UUID) -> bool:
    """Publish a committed watch-sweep run, returning ``False`` (never raising) on a
    broker outage so the durable QUEUED row is simply left for ``recovery_sweep``.

    The worker must not import the API's ``_publish_or_defer``; this mirrors its
    intent inline. Only kombu's ``OperationalError`` (every transport/connection
    failure, re-exported by celery) is swallowed — a genuine publish bug still raises.
    """
    from celery.exceptions import OperationalError

    # Watch submissions are fresh runs, so stage=None mechanically selects the
    # default GPU lane through the same routing decision as every other publisher.
    try:
        pipeline_task_for_stage(None).apply_async((str(run_id),), ignore_result=True)
    except OperationalError:
        logger.warning(
            "watch_sweep enqueue deferred (broker unavailable); run %s stays QUEUED "
            "for the recovery sweep",
            run_id,
            exc_info=True,
        )
        return False
    return True


@app.task(name="voxint.watch_sweep")  # type: ignore[misc, untyped-decorator, unused-ignore]
def watch_sweep() -> dict[str, Any]:
    """Auto-ingest new media from the operator's registered folders (issue #60).

    Thin wrapper: :func:`voxint.ingest.watch.sweep_watch_folders` owns the pass
    (effective-gate recheck, bounded scan, settle-filter, race-safe submit,
    commit-before-publish, status persistence). ``_publish_watch_run`` is injected so
    the broker-defer path stays here and the sweep logic stays broker-free and
    directly testable.
    """
    settings = get_settings()
    factory, _ = _runtime()
    summary = sweep_watch_folders(factory, settings, publish=_publish_watch_run)
    logger.info("watch_sweep %s", summary.as_dict())
    return summary.as_dict()


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
    try:
        with factory() as session:
            row = app_settings.get_app_settings(session)
            # Effective (row-over-env) enablement, so a UI toggle actually
            # governs auto-generation — never enqueue LLM work after the
            # operator turned it off (issue #10/#74). create_jobs re-checks
            # the same gate.
            if not app_settings.resolve_effective_enrichment_run_assets_autogenerate(
                row, settings
            ):
                return
            if not asset_jobs.run_asset_gates_open(settings, row):
                return
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


@app.task(name="voxint.generate_segment_embeddings", ignore_result=True)  # type: ignore[misc, untyped-decorator, unused-ignore]
def generate_segment_embeddings(job_id_str: str) -> None:
    """Run one queued embedding-index job (issue #121).

    No Celery retries (the run-asset/research precedent): failures land as
    honest, bounded error text on the job row. A duplicate delivery no-ops on the
    guarded queued→running claim. The embedder is a process-wide singleton, so a
    long-lived worker loads the ONNX graph once."""
    factory, _ = _runtime()
    embedding_jobs.execute_job(factory, uuid.UUID(job_id_str), settings=get_settings())


def _autogenerate_segment_embeddings(
    factory: sessionmaker[Session], run_id: uuid.UUID, settings: Settings
) -> None:
    """Opt-in post-finalize step: enqueue an embedding job for a completed run so
    semantic search covers it automatically. Best-effort by contract — a
    completed run is COMPLETED whatever happens here, so every failure is logged
    and swallowed. Independent of the LLM run-asset autogenerate."""
    try:
        with factory() as session:
            row = app_settings.get_app_settings(session)
            if not app_settings.resolve_effective_semantic_index_autogenerate(row, settings):
                return
            if not embedding_jobs.embedding_gates_open(settings, row):
                return
            if not minilm_artifacts_available():
                # Semantic search is enabled but the vendored MiniLM weights are
                # absent (a native install that never fetched the minilm-onnx-v1
                # asset; the Docker image always bakes them). Skip rather than
                # enqueue a job that could only fail — the native `doctor` check
                # surfaces the missing weights, and `voxint embed backfill`
                # re-indexes once they are present.
                logger.warning(
                    "semantic index skipped for run %s: MiniLM ONNX weights not "
                    "found (fetch the minilm-onnx-v1 asset, then run "
                    "`voxint embed backfill`)",
                    run_id,
                )
                return
            job, _ = embedding_jobs.create_jobs(
                session, pipeline_run_id=run_id, settings=settings
            )
            session.commit()
            job_id = str(job.id) if job is not None else None
        if job_id is not None:
            generate_segment_embeddings.delay(job_id)
    except Exception:
        logger.exception("post-finalize embedding enqueue failed for run %s", run_id)


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
