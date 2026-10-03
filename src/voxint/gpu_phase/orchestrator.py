"""One transaction-serialized phase step, followed by bounded lane publication.

Inspect services and broker ownership in llm even when idle: stopped containers
alone cannot prove a broker lease was returned. Active phases renew instead of
also issuing a redundant status request. Startup uses idempotent starts and
inspection, so restarting the worker needs no process-local startup marker.
Leases last at least 600 seconds (ten tick intervals) and renew before stopping
services so sequential Docker stops have time to finish before broker release.
"""

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from voxint.api.health_probe import ServiceHealth
from voxint.api.service_control import SERVICE_KEYS, ControlOutcome, ServiceController, ServiceState
from voxint.backoff import backoff_seconds
from voxint.config import Settings
from voxint.db.models import Stage
from voxint.gpu_phase import state
from voxint.gpu_phase.client import AcquireResult, LeaseClient, ReleaseResult, StatusResult
from voxint.gpu_phase.state import GpuPhase as P
from voxint.gpu_phase.state import GpuPhaseSnapshot, OperatorRequest, PhaseFields

logger = logging.getLogger(__name__)

GPU_PHASE_ADVISORY_LOCK_KEY = 0x766F78696E746770
SERVICE_READY_SECONDS = 600
ACQUIRE_PENDING_SECONDS = 600


@dataclass(frozen=True)
class Counts:
    gpu_lane_demand: int = 0
    gpu_in_flight: int = 0
    post_in_flight: int = 0
    running_llm_jobs: int = 0


@dataclass(frozen=True)
class Observed:
    controllable: bool = True
    services: tuple[ServiceState, ...] = ()
    status: StatusResult | None = None
    acquire: AcquireResult | None = None
    release: ReleaseResult | None = None
    started: bool = False
    ready: bool = False
    control_error: str | None = None
    stopped: bool = False


@dataclass(frozen=True)
class Transition:
    phase: P
    fields: PhaseFields = field(default_factory=lambda: PhaseFields())


@dataclass(frozen=True)
class TickResult:
    kind: str
    phase: P | None = None


def decide(
    snapshot: GpuPhaseSnapshot,
    counts: Counts,
    observed: Observed,
    now: datetime,
    settings: Settings,
) -> Transition:
    """Decide solely from persisted state and facts collected for this tick."""
    o = observed
    age = (now - snapshot.phase_since).total_seconds()
    due = snapshot.retry_after is None or now >= snapshot.retry_after
    release = snapshot.operator_request == OperatorRequest.RELEASE
    fields: PhaseFields = {}
    a = o.acquire
    if a and a.kind == "held" and a.expires_at is not None and a.expires_at <= now:
        a = AcquireResult("lost", reason="GPU lease lost")
    if snapshot.phase != P.ACQUIRING and a and a.kind == "held":
        fields.update({"lease_id": a.lease_id, "lease_expires_at": a.expires_at})

    def failure(phase: P, reason: str, delay: int = 0) -> Transition:
        return Transition(
            phase,
            {
                **fields,
                "last_error": reason[:240],
                "failures": snapshot.failures + 1,
                "retry_after": now
                + timedelta(
                    seconds=min(
                        max(delay, backoff_seconds(snapshot.failures + 1, 30, 900)),
                        (datetime.max.replace(tzinfo=now.tzinfo) - now).total_seconds() - 1,
                    )
                ),
            },
        )

    if not o.controllable:
        return failure(P.LLM if snapshot.phase == P.LLM else P.ERROR, "service control unavailable")
    if o.control_error:
        if snapshot.phase in (P.STOPPING_SERVICES, P.RELEASING):
            return failure(P.ERROR, "could not return the GPU: " + o.control_error)
        return failure(P.LLM if snapshot.phase == P.LLM else P.STOPPING_SERVICES, o.control_error)
    if (
        snapshot.phase == P.RELEASING
        and o.services
        and any(value != ServiceState.STOPPED for value in o.services)
    ):
        return failure(P.STOPPING_SERVICES, "audio services are not stopped")
    if snapshot.phase == P.LLM:
        if ServiceState.RUNNING in o.services or (
            o.status is not None and o.status.kind == "held" and o.status.holder == "voxint"
        ):
            if o.status and o.status.kind == "held" and o.status.holder == "voxint":
                fields.update(
                    {"lease_id": o.status.lease_id, "lease_expires_at": o.status.expires_at}
                )
            return Transition(P.STOPPING_SERVICES, fields)
        if release:
            return Transition(
                P.LLM,
                {
                    "operator_request": None,
                    "retry_after": now + timedelta(seconds=settings.gpu_phase_min_dwell_seconds),
                },
            )
        if snapshot.operator_request == OperatorRequest.AUDIO or (
            counts.gpu_lane_demand > 0 and age >= settings.gpu_phase_min_dwell_seconds and due
        ):
            return Transition(
                P.DRAINING_POST, {"operator_request": None, "last_error": None, "retry_after": None}
            )
    elif snapshot.phase == P.DRAINING_POST:
        if release:
            return Transition(P.LLM, {"operator_request": None})
        if counts.post_in_flight == counts.running_llm_jobs == 0:
            return Transition(P.ACQUIRING)
    elif snapshot.phase == P.ACQUIRING:
        if a:
            if a.kind == "held":
                return Transition(
                    P.STARTING_SERVICES,
                    {"lease_id": a.lease_id, "lease_expires_at": a.expires_at, "last_error": None},
                )
            if a.kind != "pending" or age >= ACQUIRE_PENDING_SECONDS:
                return failure(
                    P.LLM if a.kind == "busy" else P.RELEASING,
                    a.reason or "acquire pending window exceeded",
                    a.retry_after_seconds,
                )
    elif snapshot.phase in (P.STARTING_SERVICES, P.AUDIO, P.DRAINING):
        if a and a.kind == "held":
            fields.update({"lease_id": a.lease_id, "lease_expires_at": a.expires_at})
        elif (a and a.kind == "lost") or (
            snapshot.lease_expires_at is not None and now >= snapshot.lease_expires_at
        ):
            return failure(P.STOPPING_SERVICES, "GPU lease lost")
        if (
            snapshot.phase in (P.AUDIO, P.DRAINING)
            and o.services
            and ServiceState.STOPPED in o.services
        ):
            return failure(P.STOPPING_SERVICES, "audio service is not running")
        if snapshot.phase == P.STARTING_SERVICES:
            if o.ready:
                return Transition(P.AUDIO, fields)
            if age >= SERVICE_READY_SECONDS:
                return failure(P.STOPPING_SERVICES, "audio services did not become ready")
        elif snapshot.phase == P.AUDIO:
            if (
                release
                or age >= settings.gpu_phase_max_audio_seconds
                or (
                    counts.gpu_lane_demand == counts.gpu_in_flight == 0
                    and age >= settings.gpu_phase_min_dwell_seconds
                )
            ):
                return Transition(P.DRAINING, {**fields, "operator_request": None})
        elif release or counts.gpu_in_flight == 0:
            if release:
                fields["operator_request"] = None
            return Transition(P.STOPPING_SERVICES, fields)
    elif snapshot.phase == P.STOPPING_SERVICES:
        if o.stopped:
            return Transition(P.RELEASING, fields)
    elif snapshot.phase == P.RELEASING:
        if o.release and o.release.kind == "released":
            return Transition(
                P.LLM,
                {
                    "lease_id": None,
                    "lease_expires_at": None,
                    **({"operator_request": None} if release else {}),
                    "failures": snapshot.failures if snapshot.last_error else 0,
                    "retry_after": snapshot.retry_after if snapshot.last_error else None,
                },
            )
        if o.release:
            return failure(P.ERROR, "could not return the GPU: " + o.release.reason)
    elif snapshot.phase == P.ERROR:
        if a and a.kind == "lost":
            # Keep the original reason the hand-back failed; add the lease loss.
            prior = snapshot.last_error or ""
            if "GPU lease lost" not in prior:
                fields["last_error"] = (f"{prior}; GPU lease lost" if prior else "GPU lease lost")[
                    :240
                ]
        if due or release:
            if release:
                fields["operator_request"] = None
            return Transition(P.STOPPING_SERVICES, fields)
    return Transition(snapshot.phase, fields)


def tick(
    session_factory: sessionmaker[Session],
    settings: Settings,
    *,
    client: LeaseClient,
    controller: ServiceController,
    probe: Callable[[Settings], list[ServiceHealth]],
    clock: Callable[[], datetime],
) -> TickResult:
    if not settings.gpu_phase_enabled:
        return TickResult("disabled")
    with session_factory.begin() as session:
        if not session.scalar(select(func.pg_try_advisory_xact_lock(GPU_PHASE_ADVISORY_LOCK_KEY))):
            return TickResult("busy")
        s = state.read_phase(session)
        if s is None:
            state.set_phase(session, P.LLM, now=clock())
            s = state.read_phase(session)
            assert s is not None
        counts = Counts(
            state.gpu_lane_demand(session),
            state.gpu_lane_in_flight(session),
            state.post_lane_in_flight(session),
            state.running_llm_jobs(session),
        )
        o = _observe(s, settings, client, controller, probe)
        now = clock()
        transition = decide(s, counts, o, now, settings)
        state.set_phase(
            session,
            transition.phase,
            now=now,
            expected_request=s.operator_request,
            **transition.fields,
        )
    if transition.phase != s.phase and transition.phase in (P.AUDIO, P.LLM):
        _publish_lane(session_factory, settings, transition.phase, now)
    return TickResult("advanced" if transition.phase != s.phase else "waiting", transition.phase)


def _observe(
    s: GpuPhaseSnapshot,
    settings: Settings,
    client: LeaseClient,
    controller: ServiceController,
    probe: Callable[[Settings], list[ServiceHealth]],
) -> Observed:
    renewal = None
    if s.phase in (P.STARTING_SERVICES, P.AUDIO, P.DRAINING, P.STOPPING_SERVICES, P.ERROR):
        if s.lease_id:
            renewal = client.acquire(s.lease_id)
        elif s.phase in (P.STARTING_SERVICES, P.AUDIO, P.DRAINING):
            renewal = AcquireResult("lost")
    try:
        return _observe_services(s, settings, client, controller, probe, renewal)
    except Exception as exc:
        return Observed(acquire=renewal, control_error=type(exc).__name__[:240])


def _observe_services(
    s: GpuPhaseSnapshot,
    settings: Settings,
    client: LeaseClient,
    controller: ServiceController,
    probe: Callable[[Settings], list[ServiceHealth]],
    renewal: AcquireResult | None,
) -> Observed:
    if not controller.controllable:
        return Observed(controllable=False, acquire=renewal)
    if s.phase == P.LLM:
        return Observed(
            services=tuple(controller.inspect(k) for k in sorted(SERVICE_KEYS)),
            status=client.status(),
        )
    if s.phase == P.ACQUIRING:
        return Observed(acquire=client.acquire(s.lease_id))
    if s.phase in (P.STARTING_SERVICES, P.AUDIO, P.DRAINING):
        assert renewal is not None
        services = tuple(controller.inspect(k) for k in sorted(SERVICE_KEYS))
        if s.phase == P.STARTING_SERVICES and renewal.kind == "held":
            if all(x == ServiceState.RUNNING for x in services):
                health = probe(settings)
                return Observed(
                    acquire=renewal, ready=len(health) == 3 and all(h.up for h in health)
                )
            results = [controller.start(k) for k in sorted(SERVICE_KEYS)]
            error = any(
                r.outcome not in (ControlOutcome.STARTED, ControlOutcome.ALREADY_RUNNING)
                for r in results
            )
            return Observed(
                acquire=renewal,
                started=True,
                control_error="could not start audio services" if error else None,
            )
        return Observed(acquire=renewal, services=services)
    if s.phase == P.STOPPING_SERVICES:
        results = [controller.stop(k) for k in sorted(SERVICE_KEYS)]
        stopped = all(controller.inspect(k) == ServiceState.STOPPED for k in sorted(SERVICE_KEYS))
        ok = stopped and all(
            r.outcome in (ControlOutcome.STOPPED, ControlOutcome.ALREADY_STOPPED) for r in results
        )
        return Observed(
            acquire=renewal,
            stopped=ok,
            control_error=None if ok else "could not stop audio services",
        )
    if s.phase == P.RELEASING:
        services = tuple(controller.inspect(k) for k in sorted(SERVICE_KEYS))
        if any(value != ServiceState.STOPPED for value in services):
            return Observed(services=services)
        lease_id = s.lease_id
        if lease_id is None:
            status = client.status()
            if status.kind == "failed":
                return Observed(release=ReleaseResult("failed", "broker status unavailable"))
            if status.kind == "free" or status.holder != "voxint":
                return Observed(release=ReleaseResult("released"))
            lease_id = status.lease_id
        return Observed(release=client.release(lease_id or ""))
    return Observed(acquire=renewal)


def _publish_lane(
    factory: sessionmaker[Session], settings: Settings, phase: P, now: datetime
) -> None:
    # Lazy imports keep task registration out of the pure decision module's import path.
    from celery.exceptions import OperationalError

    from voxint.db.models import GPU_SEGMENT, POST_SEGMENT
    from voxint.enrichment import asset_jobs, research_jobs, translation_jobs
    from voxint.gpu_phase.dispatch import redispatch_queued_runs
    from voxint.worker import tasks

    def publish(run_id: uuid.UUID, stage: Stage | None) -> bool:
        tasks.pipeline_task_for_stage(stage).apply_async((str(run_id),), ignore_result=True)
        return True

    try:
        with factory() as session:
            redispatch_queued_runs(
                session,
                lanes=GPU_SEGMENT if phase == P.AUDIO else POST_SEGMENT,
                limit=settings.recovery_publish_batch_size,
                publish=publish,
            )
            if phase == P.LLM:
                for module, task in (
                    (asset_jobs, tasks.generate_run_asset),
                    (translation_jobs, tasks.translate_run),
                    (research_jobs, tasks.research_speaker),
                ):
                    for job_id in module.stale_queued_job_ids(
                        session, cutoff=now, limit=tasks.STALE_EMBEDDING_REDISPATCH_LIMIT
                    ):
                        task.apply_async((str(job_id),), ignore_result=True)
    except OperationalError:
        logger.warning("GPU phase lane publication deferred")
