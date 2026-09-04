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

from sqlalchemy import and_, func, or_, select
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
from voxint.enrichment import asset_jobs, embedding_jobs, translation_jobs
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
    close_paused_run_claims,
    execute_run,
    recover_interrupted_runs,
)
from voxint.pipeline.stages.context import (
    StageContext,
    apply_run_preferences,
    build_stage_context,
    build_stage_fns,
    parse_config_resolution_version,
    resolve_run_preferences,
)
from voxint.pipeline.transitions import (
    InvalidTransitionError,
    RunSnapshot,
    StaleRevisionError,
    cas_update_run,
)
from voxint.plugins import RunCompletedEvent, get_plugins
from voxint.plugins.hooks import dispatch_run_completed, redispatch_stale_lane_jobs
from voxint.worker.app import app

logger = logging.getLogger(__name__)

# Oldest-first cap on stale QUEUED embedding jobs re-dispatched per recovery
# sweep (#130). A mass stranding drains this many per sweep; the next sweep takes
# the next batch. Bounds one pass's broker traffic and keeps a broker-down sweep
# from stalling on a long series of connect timeouts.
STALE_EMBEDDING_REDISPATCH_LIMIT = 100


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


def is_saturation(exc: StageFailedError) -> bool:
    """True when the stage failed because a model service was at capacity.

    Saturation is flow control, not evidence that the model or the media is
    broken.  The retry budget should not be consumed by it.
    """
    return isinstance(exc.cause, ServiceError) and exc.cause.code == "saturated"


SATURATED_PREFIX = "saturated:"


def backoff_seconds(attempts: int, base: float, cap: float) -> float:
    """Exponential in completed attempts, capped; jitter is the caller's."""
    exp = min(max(attempts - 1, 0), 30)
    return float(min(base * 2 ** exp, cap))


def stage_attempts(
    session: Session,
    run_id: uuid.UUID,
    stage: Stage,
    *,
    exclude_saturated: bool = False,
) -> int:
    """Transient-failure attempts recorded in the ledger — the restart-proof
    retry budget.

    Interruption attempts (lease expiry: worker death, OOM, redeploy) are
    excluded — infra churn must not eat the budget for *service* failures.
    Crash loops are bounded separately by the recovery sweep's own ceiling.

    When *exclude_saturated* is set, saturation rejections (flow control) are
    also excluded — they should not consume the budget for genuine failures.
    """
    if exclude_saturated:
        error_filter = or_(
            StageRun.error.is_(None),
            and_(
                ~StageRun.error.like(f"{INTERRUPTED_PREFIX}%"),
                ~StageRun.error.like(f"{SATURATED_PREFIX}%"),
            ),
        )
    else:
        error_filter = or_(
            StageRun.error.is_(None),
            ~StageRun.error.like(f"{INTERRUPTED_PREFIX}%"),
        )
    return int(
        session.execute(
            select(func.count())
            .select_from(StageRun)
            .where(
                StageRun.pipeline_run_id == run_id,
                StageRun.stage == stage.value,
                StageRun.status == StageStatus.FAILED.value,
                error_filter,
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
        # The snapshot's config-resolution version (issue #153). Read off the RAW
        # mapping here — DomainPack.from_mapping drops unknown keys, so the decoded
        # pack cannot carry it. A NULL snapshot or a pre-#153 row with no key reads
        # as version 1 (the live-union vocabulary path); a #153 freeze carries 2.
        # Malformed metadata (a hand-edited or corrupt snapshot with a null or
        # non-numeric value) must NOT raise here: this runs OUTSIDE the execute_run
        # failure lane, so an exception would leave the run un-failed for the
        # recovery sweep to re-publish forever. Fall back to the live-union path (1),
        # the same corrupt-snapshot tolerance domain_pack_from_snapshot already takes.
        config_resolution_version = parse_config_resolution_version(pack_snapshot)
        max_speakers_hint = (
            run_row.diarization_max_speakers if run_row is not None else None
        )
        num_speakers_hint = (
            run_row.diarization_num_speakers if run_row is not None else None
        )
    pack = domain_pack_from_snapshot(pack_snapshot, settings)
    ctx = apply_run_preferences(
        base_ctx,
        settings,
        prefs,
        pack,
        llm_api_key=llm_api_key,
        bundled=bundled,
        config_resolution_version=config_resolution_version,
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
            budget = stage_attempts(
                session, run_id, exc.stage, exclude_saturated=True
            )
        if not is_saturation(exc) and budget >= settings.stage_max_attempts:
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
            _auto_enroll_speakers(factory, run_id, settings)
            _autogenerate_run_assets(factory, run_id, settings)
            _autogenerate_segment_embeddings(factory, run_id, settings)
            _autogenerate_translation(factory, run_id, settings)
            # Generic post-completion fan-out (issue #138): each active plugin's
            # on_run_completed runs alongside the three hard-coded producers above
            # (which convert into plugins in #139-#141). Per-plugin failures are
            # contained by dispatch_run_completed; the run stays COMPLETED. Empty
            # registry ⇒ no plugins ⇒ no-op.
            dispatch_run_completed(
                get_plugins().plugins,
                RunCompletedEvent(
                    run_id=run_id, session_factory=factory, settings=settings
                ),
            )
            _refresh_term_stats(factory, run_id)
            _refresh_speaker_insights(factory, run_id)
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
    """Reclaim expired-lease runs and re-enqueue stale QUEUED runs, and
    re-dispatch stale QUEUED built-in jobs whose dispatch was lost.

    Duplicate enqueues are safe — stage claims and job-claim CAS arbitrate — so
    this errs toward re-publishing. The staleness grace keeps it from stepping on
    freshly-submitted runs, pending retry countdowns, and just-enqueued jobs still
    waiting for a worker.
    """
    from celery.exceptions import OperationalError

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
        close_paused_run_claims(session)
        session.commit()
    cutoff = datetime.now(tz=UTC) - timedelta(seconds=settings.queued_run_stale_seconds)
    with factory() as session:
        stale_queued = (
            session.execute(
                select(PipelineRun.id)
                .where(
                    PipelineRun.status == RunStatus.QUEUED.value,
                    PipelineRun.updated_at < cutoff,
                )
                .order_by(PipelineRun.updated_at)
                .limit(settings.recovery_publish_batch_size)
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
        # Cap total dispatches per sweep so a mass stranding drains gradually.
        dispatched = 0
        for rid, stage_value in resumable:
            if dispatched >= settings.recovery_publish_batch_size:
                break
            stage = Stage(stage_value) if stage_value else None
            try:
                pipeline_task_for_stage(stage).apply_async(
                    (str(rid),), ignore_result=True
                )
                dispatched += 1
            except OperationalError:
                logger.warning(
                    "recovery enqueue deferred (broker unavailable); run %s stays "
                    "QUEUED for a later sweep",
                    rid,
                    exc_info=True,
                )
    # Stranded embedding jobs (issue #130): a job committed QUEUED whose Celery
    # dispatch evaporated (crash/broker down between the row commit and publish)
    # holds the one-active slot forever, so the run stays unindexed and
    # `voxint embed backfill` skips it as already-active. Re-dispatch each by id
    # past the SAME staleness grace — no row mutation; the guarded queued→running
    # claim CAS makes a duplicate delivery (or a live worker about to claim it) a
    # no-op. RUNNING is deliberately untouched (the lane's force-cancel policy).
    #
    # No dispatch lease is recorded (the deliberate no-mutation design), so a job
    # that stays QUEUED — broker up but no worker consuming — is re-published every
    # sweep until something claims it. Those duplicates are harmless (the claim CAS
    # collapses them) but do accumulate on the broker, so the batch is oldest-first
    # and bounded: a mass stranding drains STALE_EMBEDDING_REDISPATCH_LIMIT per
    # sweep rather than flooding one pass.
    with factory() as session:
        stale_embedding_jobs = embedding_jobs.stale_queued_job_ids(
            session, cutoff=cutoff, limit=STALE_EMBEDDING_REDISPATCH_LIMIT
        )
    if stale_embedding_jobs:
        for job_id in stale_embedding_jobs:
            try:
                generate_segment_embeddings.apply_async(
                    (str(job_id),), ignore_result=True
                )
            except OperationalError:
                logger.warning(
                    "embedding recovery enqueue deferred (broker unavailable); "
                    "job %s stays QUEUED for a later sweep",
                    job_id,
                    exc_info=True,
                )
    with factory() as session:
        stale_asset_jobs = asset_jobs.stale_queued_job_ids(
            session, cutoff=cutoff, limit=STALE_EMBEDDING_REDISPATCH_LIMIT
        )
    for job_id in stale_asset_jobs:
        try:
            generate_run_asset.apply_async((str(job_id),), ignore_result=True)
        except OperationalError:
            logger.warning(
                "run-asset recovery enqueue deferred (broker unavailable); "
                "job %s stays QUEUED for a later sweep",
                job_id,
                exc_info=True,
            )
    with factory() as session:
        stale_translation_jobs = translation_jobs.stale_queued_job_ids(
            session, cutoff=cutoff, limit=STALE_EMBEDDING_REDISPATCH_LIMIT
        )
    for job_id in stale_translation_jobs:
        try:
            translate_run.apply_async((str(job_id),), ignore_result=True)
        except OperationalError:
            logger.warning(
                "translation recovery enqueue deferred (broker unavailable); "
                "job %s stays QUEUED for a later sweep",
                job_id,
                exc_info=True,
            )
    # Generic stale-QUEUED recovery for plugin job lanes (issue #138): the same
    # oldest-first, bounded, no-mutation redispatch the embedding block above does
    # (issue #130), driven by each active plugin's JobLaneSpec. The embedding lane
    # stays hard-coded until #140 converts it and declares its own lane.
    lanes = get_plugins().job_lanes()
    result = {
        "recovered": len(recovered),
        "stale_queued": len(stale_queued),
        "cancelled_claims_closed": len(cancelled_claims),
        "stale_embedding_jobs": len(stale_embedding_jobs),
        "stale_asset_jobs": len(stale_asset_jobs),
        "stale_translation_jobs": len(stale_translation_jobs),
    }
    # Surface the plugin-lane count only when a plugin actually declares a lane;
    # built-in lane counts above are always present.
    if lanes:
        plugin_lane_counts = redispatch_stale_lane_jobs(
            lanes,
            session_factory=factory,
            send_task=app.send_task,
            cutoff=cutoff,
        )
        result["plugin_lanes"] = sum(plugin_lane_counts.values())
    return result


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


@app.task(name="voxint.activity_prune")  # type: ignore[misc, untyped-decorator, unused-ignore]
def activity_prune() -> dict[str, int]:
    """Prune the console activity outbox to its newest-N rows (issue #162).

    Bounded retention for ``activity_events`` (the browser polls a recent-activity
    feed, not an audit log). OFF unless ``console_activity_enabled`` — the gate is
    re-checked here (not just at beat registration) so a stale schedule entry can
    never act. Gap-safe newest-N (see ``prune_activity_events``).
    """
    from voxint.activity import prune_activity_events

    settings = get_settings()
    if not settings.console_activity_enabled:
        return {"pruned": 0}
    factory, _ = _runtime()
    with factory() as session:
        pruned = prune_activity_events(session)
        session.commit()
    logger.info("activity_prune pruned=%d", pruned)
    return {"pruned": pruned}


@app.task(name="voxint.media_reconcile")  # type: ignore[misc, untyped-decorator, unused-ignore]
def media_reconcile() -> dict[str, int]:
    """Drive interrupted media operations to a consistent terminal state (ADR 0007).

    Processes non-terminal journal rows (move, trash, restore) by classifying
    filesystem reality against recorded intent. Always registered on beat; a
    no-op when no non-terminal rows exist.
    """
    from voxint.media.reconcile import reconcile_operations

    settings = get_settings()
    factory, _ = _runtime()
    summary = reconcile_operations(
        factory,
        settings.media_root,
        batch_limit=settings.media_reconcile_batch_limit,
    )
    if summary.selected > 0:
        logger.info("media_reconcile %s", summary.as_dict())
    return summary.as_dict()


@app.task(name="voxint.watch_sweep")  # type: ignore[misc, untyped-decorator, unused-ignore]
def watch_sweep() -> dict[str, Any]:
    """Auto-ingest new media from the operator's registered folders (issue #60).

    Thin wrapper: :func:`voxint.ingest.watch.sweep_watch_folders` owns the pass
    (effective-gate recheck, bounded scan, settle-filter, race-safe submit,
    commit-before-publish via :class:`~voxint.ingest.service.SubmissionResult`,
    status persistence).
    """
    settings = get_settings()
    factory, _ = _runtime()
    summary = sweep_watch_folders(factory, settings)
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
            generate_run_asset.apply_async((job_id,), ignore_result=True)
    except Exception:
        logger.exception("post-finalize run-asset enqueue failed for run %s", run_id)


@app.task(name="voxint.translate_run", ignore_result=True)  # type: ignore[misc, untyped-decorator, unused-ignore]
def translate_run(job_id_str: str) -> None:
    """Run one queued transcript-translation job (issue #133).

    No Celery retries on purpose (the run-asset precedent): failures land as
    honest, bounded error text on the job row for the operator to see. A
    duplicate delivery no-ops on the guarded queued→running claim.
    """
    factory, _ = _runtime()
    translation_jobs.execute_job(factory, uuid.UUID(job_id_str), settings=get_settings())


def _autogenerate_translation(
    factory: sessionmaker[Session], run_id: uuid.UUID, settings: Settings
) -> None:
    """Opt-in post-finalize step: enqueue a translation job for a completed run
    when auto-translate is on, a target language is set, the LLM path is open,
    and the run's detected language differs from the target. Best-effort by
    contract — a completed run is COMPLETED whatever happens here, so every
    failure is logged and swallowed."""
    try:
        with factory() as session:
            row = app_settings.get_app_settings(session)
            if not app_settings.resolve_effective_translation_autogenerate(row, settings):
                return
            target = translation_jobs.normalized_language(
                app_settings.resolve_effective_translation_target_language(row, settings)
            )
            if target is None:
                return
            if not translation_jobs.translation_gates_open(settings, row):
                return
            run = session.get(PipelineRun, run_id)
            if run is None:
                return
            if translation_jobs.normalized_language(run.detected_language) == target:
                # Already in the preferred language — nothing to translate.
                return
            if not translation_jobs.translation_needed(session, run_id, target):
                return
            job, _already = translation_jobs.create_job(
                session, pipeline_run_id=run_id, target_language=target, settings=settings
            )
            session.commit()
            job_id = str(job.id) if job is not None else None
        if job_id is not None:
            translate_run.apply_async((job_id,), ignore_result=True)
    except Exception:
        logger.exception("post-finalize translation enqueue failed for run %s", run_id)


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
            generate_segment_embeddings.apply_async((job_id,), ignore_result=True)
    except Exception:
        logger.exception("post-finalize embedding enqueue failed for run %s", run_id)


def _auto_enroll_speakers(
    factory: sessionmaker[Session], run_id: uuid.UUID, settings: Settings
) -> None:
    """Post-completion side effect: auto-enroll unmatched voices (#275).

    Best-effort: a failure is logged and swallowed, never fails the run.
    """
    if not settings.auto_enroll:
        return
    try:
        from voxint.speakers.auto_enroll import auto_enroll_run
        from voxint.speakers.matching import gates_from_settings

        with factory() as session:
            auto_enroll_run(session, run_id, gates_from_settings(settings))
            session.commit()
    except Exception:
        logger.exception("post-finalize auto-enrollment failed for run %s", run_id)


def _refresh_term_stats(
    factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    """Best-effort post-completion term-stats refresh (issue #334).

    Enqueues a corpus-wide refresh so the explore word cloud stays warm.
    A completed run is COMPLETED whatever happens here.
    """
    from celery.exceptions import OperationalError

    try:
        compute_term_stats.apply_async((None,), ignore_result=True)
    except OperationalError:
        logger.warning(
            "term-stats refresh enqueue deferred (broker unavailable) for run %s",
            run_id,
            exc_info=True,
        )
    except Exception:
        logger.exception("post-finalize term-stats refresh failed for run %s", run_id)


@app.task(name="voxint.compute_term_stats", ignore_result=True)  # type: ignore[misc, untyped-decorator, unused-ignore]
def compute_term_stats(project_id_str: str | None = None) -> dict[str, int]:
    """Recompute corpus or project term stats and cache the artifact (issue #334).

    Dispatched after run completion to pre-warm the explore word cloud, or
    on-demand from the explore page. The synchronous in-request path handles
    small corpora; this task handles background refresh.
    """
    from voxint.api.explore_query import term_stats

    factory, _ = _runtime()
    project_id = uuid.UUID(project_id_str) if project_id_str else None
    with factory() as session:
        result = term_stats(session, project_id)
        session.commit()
    logger.info(
        "compute_term_stats project=%s terms=%d",
        project_id_str or "corpus",
        len(result.terms),
    )
    return {"terms": len(result.terms)}


def _refresh_speaker_insights(
    factory: sessionmaker[Session], run_id: uuid.UUID
) -> None:
    """Best-effort post-completion speaker-insights refresh (issue #335).

    Enqueues a corpus-wide refresh so speaker profile insights stay warm.
    A completed run is COMPLETED whatever happens here.
    """
    from celery.exceptions import OperationalError

    try:
        compute_speaker_insights.apply_async(ignore_result=True)
    except OperationalError:
        logger.warning(
            "speaker-insights refresh enqueue deferred (broker unavailable) for run %s",
            run_id,
            exc_info=True,
        )
    except Exception:
        logger.exception("post-finalize speaker-insights refresh failed for run %s", run_id)


@app.task(name="voxint.compute_speaker_insights", ignore_result=True)  # type: ignore[misc, untyped-decorator, unused-ignore]
def compute_speaker_insights() -> dict[str, int]:
    """Recompute all speakers' insights in one corpus fold (issue #335).

    Dispatched after run completion to pre-warm speaker profile insights.
    The profile page reads cached artifacts only.
    """
    from voxint.api.speaker_insights import compute_all_speaker_insights

    factory, _ = _runtime()
    with factory() as session:
        count = compute_all_speaker_insights(session)
        session.commit()
    logger.info("compute_speaker_insights speakers=%d", count)
    return {"speakers": count}


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
