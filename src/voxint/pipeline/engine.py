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
- A lane-limited executor completes its last owned claim and CAS-parks the run
  QUEUED at the next stage in that SAME transaction. Publication may happen
  later: the durable QUEUED row is the handoff, and stale-queue recovery can
  always rehydrate a broker message lost after the commit.
- Recovery touches only runs whose claim lease has expired — a healthy worker
  mid-transcription is never swept. Stage bodies remain at-least-once for
  non-transactional effects (filesystem, GPU calls) and must be idempotent.
"""

import os
import socket
import uuid
from collections.abc import Callable, Mapping
from collections.abc import Set as AbstractSet
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from voxint.config import Settings, get_settings
from voxint.db.models import PipelineRun, RunStatus, Stage, StageRun, StageStatus
from voxint.media.redaction import cap_length
from voxint.pipeline.model_identity import (
    METRICS_KEY,
    observe_stage_model_identity,
)
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


def submit(
    session: Session,
    media_item_id: uuid.UUID,
    *,
    domain_pack: dict[str, Any],
    sidecar: dict[str, Any] | None = None,
    operator_notes: str | None = None,
    diarization_max_speakers: int | None = None,
    diarization_num_speakers: int | None = None,
) -> PipelineRun:
    """Queue a fresh run, freezing its resolved domain-pack snapshot (issue #11).

    ``domain_pack`` is REQUIRED and keyword-only so a new run-creation path cannot
    silently ship an unstamped run: every caller must resolve the pack first (see
    :func:`voxint.domain_packs.registry.resolve_run_domain_pack`). The snapshot is
    write-once run provenance — the pipeline and enrichment read it, never the
    live global env.

    ``sidecar`` (issue #104) is the media file's parsed YAML sidecar mapping,
    stamped write-once alongside the pack snapshot; ``operator_notes`` seeds the
    run's notes at creation (the sidecar's ``notes`` field). Both stay optional so
    submit paths without a sidecar are unchanged.

    ``diarization_max_speakers`` / ``diarization_num_speakers`` (issue #128) freeze
    the run's optional speaker-count hint: a bound and/or an exact count. Left NULL
    ⇒ the worker uses the install-wide default at execution. Plain integer columns,
    so an explicit None is a genuine SQL NULL (unlike the JSON columns above).
    """
    run = PipelineRun(
        media_item_id=media_item_id,
        status=RunStatus.QUEUED.value,
        domain_pack=domain_pack,
        diarization_max_speakers=diarization_max_speakers,
        diarization_num_speakers=diarization_num_speakers,
    )
    # Set only when present: an explicit None on a JSON column serializes as a
    # JSON null (jsonb_typeof 'null'), not SQL NULL, and would trip the
    # sidecar object-shape check constraint.
    if sidecar is not None:
        run.sidecar = sidecar
    if operator_notes is not None:
        run.operator_notes = operator_notes
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
    *,
    model_identity: dict[str, Any] | None = None,
) -> None:
    claim = session.get(StageRun, claim_id)
    assert claim is not None
    claim.status = status.value
    claim.finished_at = datetime.now(tz=UTC)
    # General length cap for every StageRun.error write (any stage, any caller):
    # keeps a pathological diagnostic from bloating the ledger. Redaction of the
    # ACQUIRE stderr tail happens upstream at the raise site (born clean).
    claim.error = cap_length(error) if error is not None else None
    # Best-effort model-identity provenance for this attempt (see
    # voxint.pipeline.model_identity). Only the successful completion path passes
    # it, so a failed/superseded attempt never records identity; it is stored in
    # the SAME transaction that completes the claim, keeping the stamp attempt-safe.
    if model_identity is not None:
        # Merge rather than replace: model_identity is the only writer today, but a
        # future stage that records its own metrics on the claim row must not be
        # silently clobbered by the stamp.
        merged = dict(claim.metrics or {})
        merged[METRICS_KEY] = model_identity
        claim.metrics = merged


def _observe_stage_identity(
    settings: "Settings | None", stage: Stage
) -> dict[str, Any] | None:
    """Best-effort model-identity observation for a stage. Never raises.

    Returns None when there is no settings context (identity is advisory, never
    worth failing a run over) or the stage calls no model service.
    """
    if settings is None:
        return None
    try:
        return observe_stage_model_identity(settings, stage)
    except Exception:
        return None


def default_stage_leases() -> dict[Stage, int]:
    """Per-stage lease budgets from settings.

    Two stages get dedicated leases; the rest share stage_lease_seconds.
    diarize_embed: one diarization call plus every embedding batch must fit
    inside the lease, or a healthy worker gets robbed mid-stage. acquire: the
    yt-dlp download timeout plus its kill/hash/publish tail must fit, so its
    lease is sized against acquire_timeout_seconds, not a GPU call.
    """
    settings = get_settings()
    leases = {stage: settings.stage_lease_seconds for stage in Stage}
    leases[Stage.DIARIZE_EMBED] = settings.diarize_embed_lease_seconds
    leases[Stage.ACQUIRE] = settings.acquire_lease_seconds
    return leases


def _stop_if_cancelled(
    session_factory: sessionmaker[Session],
    run_id: uuid.UUID,
    claim_id: uuid.UUID,
) -> RunSnapshot | None:
    """Resolve a lost post-stage CAS: was the run cancelled out from under us?

    A worker's advance/complete/failure CAS can lose for exactly one benign
    reason — the operator cancelled the run mid-stage (issue #5) — and for
    genuine-race reasons that must stay loud (a recovery lease-steal moving
    RUNNING→FAILED→QUEUED, a stray duplicate worker, a future transition). We
    only swallow the *confirmed cancellation*: re-read on a fresh session and
    return the CANCELLED snapshot, else return None so the caller re-raises the
    ``StaleRevisionError``.

    Cancellation is cooperative — the stage body already ran and its DB effects
    were rolled back with the failed CAS, so we do not un-run it; we only close
    the now-abandoned RUNNING claim as SKIPPED so the console does not show a
    stage "running" forever on a cancelled run.
    """
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        if run is None or RunStatus(run.status) is not RunStatus.CANCELLED:
            return None
        snap = snapshot(run)
        claim = session.get(StageRun, claim_id)
        if claim is not None and claim.status == StageStatus.RUNNING.value:
            claim.status = StageStatus.SKIPPED.value
            claim.finished_at = datetime.now(tz=UTC)
            claim.error = "cancelled before commit"
        session.commit()
        return snap


def _stop_if_paused(
    session_factory: sessionmaker[Session],
    run_id: uuid.UUID,
    claim_id: uuid.UUID,
) -> RunSnapshot | None:
    """Resolve a lost post-stage CAS: was the run paused out from under us?

    Mirrors :func:`_stop_if_cancelled` — the stage body already ran and its DB
    effects were rolled back with the failed CAS. We close the abandoned RUNNING
    claim as SKIPPED so the console never shows a stage "running" on a paused
    run. Unlike cancellation, the run is not terminal — it can be resumed.
    """
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        if run is None or RunStatus(run.status) is not RunStatus.PAUSED:
            return None
        snap = snapshot(run)
        claim = session.get(StageRun, claim_id)
        if claim is not None and claim.status == StageStatus.RUNNING.value:
            claim.status = StageStatus.SKIPPED.value
            claim.finished_at = datetime.now(tz=UTC)
            claim.error = "paused before commit"
        session.commit()
        return snap


def execute_run(
    session_factory: sessionmaker[Session],
    run_id: uuid.UUID,
    stage_fns: dict[Stage, StageFn],
    *,
    worker_id: str | None = None,
    lease_seconds: int | Mapping[Stage, int] | None = None,
    settings: "Settings | None" = None,
    stages: AbstractSet[Stage] | None = None,
) -> RunSnapshot:
    """Advance a QUEUED (or requeued) run through its stages.

    Returns the run snapshot as this worker last saw it; if another worker holds
    an active claim, returns immediately without executing anything. Each stage
    executes in its own transaction: stage effects, the claim update, and the
    CAS advance commit atomically, so a crash between stages leaves a
    consistent, resumable run. ``stages=None`` preserves the original all-stage
    execution. A restricted stage set returns without mutation on wrong-lane
    delivery and parks QUEUED at the first stage outside its lane after a
    successful owned stage.
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
            assert stage is not None
            # A message delivered to the wrong lane is a pure no-op: in
            # particular, do not take the QUEUED -> RUNNING entry CAS and leave
            # an ownerless RUNNING run for lease recovery to repair.
            if stages is not None and stage not in stages:
                return held
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
            # Defensive counterpart to the entry guard. It covers a lane-limited
            # invocation that observes a RUNNING run already advanced by another
            # worker; it must never claim work assigned to the other lane.
            if stages is not None and stage not in stages:
                return held

        lease = leases[stage] if isinstance(leases, Mapping) else leases
        claim_id = _claim_stage(session_factory, held, stage, worker, lease)
        if claim_id is None:
            # Either another worker owns this stage right now, or the run moved
            # out from under us (e.g. an operator cancel landing between the
            # loop's top-of-iteration read and this claim, which _claim_stage
            # rejects under its row lock). Re-read so the returned snapshot
            # reflects the committed DB state, not the possibly-stale pre-claim
            # view (an honest return value / log for a cancelled run).
            with session_factory() as session:
                current = session.get(PipelineRun, run_id)
                return snapshot(current) if current is not None else held

        # Observe which model answers this stage, BEFORE opening the stage session
        # so no DB connection is held during the network probe. Best-effort and
        # bounded; stamped onto this attempt only if the stage completes.
        identity = _observe_stage_identity(settings, stage)

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
                try:
                    failed = cas_update_run(
                        session,
                        held,
                        status=RunStatus.FAILED,
                        current_stage=stage,
                        error=str(exc),
                        settings=settings,
                    )
                    session.commit()
                except StaleRevisionError:
                    # A cancel or pause landed while this stage was failing: the
                    # FAILED CAS lost. Honour a confirmed cancellation/pause over
                    # the failure; re-raise for any other stale status.
                    session.rollback()
                    cancelled = _stop_if_cancelled(session_factory, run_id, claim_id)
                    if cancelled is not None:
                        return cancelled
                    paused = _stop_if_paused(session_factory, run_id, claim_id)
                    if paused is not None:
                        return paused
                    raise
                raise StageFailedError(stage, exc, failed) from exc
            _finish_claim(session, claim_id, StageStatus.COMPLETED, model_identity=identity)
            upcoming = next_stage(stage)
            try:
                if upcoming is None:
                    held = cas_update_run(
                        session,
                        held,
                        status=RunStatus.COMPLETED,
                        current_stage=None,
                        settings=settings,
                    )
                    session.commit()
                    return held
                if stages is not None and upcoming not in stages:
                    # The completed claim, all stage-owned DB effects, and the
                    # durable QUEUED handoff commit atomically. A crash after
                    # this commit can lose only the broker publication, never
                    # the knowledge of which stage the other lane must resume.
                    held = cas_update_run(
                        session,
                        held,
                        status=RunStatus.QUEUED,
                        current_stage=upcoming,
                    )
                    session.commit()
                    return held
                cas_update_run(session, held, status=RunStatus.RUNNING, current_stage=upcoming)
                session.commit()
            except StaleRevisionError:
                # The operator cancelled or paused while this stage body ran, so
                # the post-stage advance/complete CAS lost. Discard this stage's
                # (rolled-back) effects and stop cleanly IFF the run is now
                # CANCELLED or PAUSED; any other stale status is a genuine
                # invariant breach and must propagate.
                session.rollback()
                cancelled = _stop_if_cancelled(session_factory, run_id, claim_id)
                if cancelled is not None:
                    return cancelled
                paused = _stop_if_paused(session_factory, run_id, claim_id)
                if paused is not None:
                    return paused
                raise


def recover_interrupted_runs(
    session: Session, *, max_attempts: int | None = None, settings: "Settings | None" = None
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
            # RUNNING -> FAILED -> QUEUED keeps the transition map honest. Pass
            # settings so the parked-FAILED case (budget exhausted, no requeue
            # below) emits; the immediately-requeued case emits too but the
            # delivery sweep suppresses it once the run has advanced to QUEUED.
            held = cas_update_run(
                session,
                held,
                status=RunStatus.FAILED,
                current_stage=held.current_stage,
                error=f"{INTERRUPTED_PREFIX} worker died mid-stage",
                settings=settings,
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


def close_cancelled_run_claims(session: Session) -> list[uuid.UUID]:
    """Close stage claims left RUNNING on a CANCELLED run — the crash-window
    backstop for cooperative cancellation (issue #5).

    :func:`execute_run` closes the abandoned claim itself in the happy path (via
    ``_stop_if_cancelled``), but if the worker dies AFTER ``cancel_run`` commits
    and BEFORE it reaches that cleanup, the claim is orphaned RUNNING forever:
    ``recover_interrupted_runs`` only scans RUNNING *runs*, so it never touches a
    cancelled one. This sweep (run from the beat) closes those orphans SKIPPED so
    the console never shows a stage "running" on a cancelled run. It never
    requeues — a CANCELLED run is terminal. Returns the run ids it touched.
    """
    claims = (
        session.execute(
            select(StageRun)
            .join(PipelineRun, PipelineRun.id == StageRun.pipeline_run_id)
            .where(
                StageRun.status == StageStatus.RUNNING.value,
                PipelineRun.status == RunStatus.CANCELLED.value,
            )
        )
        .scalars()
        .all()
    )
    closed: list[uuid.UUID] = []
    for claim in claims:
        _finish_claim(session, claim.id, StageStatus.SKIPPED, "cancelled before commit")
        closed.append(claim.pipeline_run_id)
    return closed


def close_paused_run_claims(session: Session) -> list[uuid.UUID]:
    """Close stage claims left RUNNING on a PAUSED run — the crash-window
    backstop for cooperative pause.

    Unlike :func:`close_cancelled_run_claims` (which closes unconditionally
    because CANCELLED is terminal), this checks lease expiry: a PAUSED run can
    be resumed, so closing a live worker's claim would let a resumed run start
    a second worker on the same stage concurrently.
    """
    now = datetime.now(tz=UTC)
    claims = (
        session.execute(
            select(StageRun)
            .join(PipelineRun, PipelineRun.id == StageRun.pipeline_run_id)
            .where(
                StageRun.status == StageStatus.RUNNING.value,
                PipelineRun.status == RunStatus.PAUSED.value,
                or_(
                    StageRun.lease_expires_at.is_(None),
                    StageRun.lease_expires_at <= now,
                ),
            )
        )
        .scalars()
        .all()
    )
    closed: list[uuid.UUID] = []
    for claim in claims:
        _finish_claim(session, claim.id, StageStatus.SKIPPED, "paused before commit")
        closed.append(claim.pipeline_run_id)
    return closed
