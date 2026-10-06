"""Shared-GPU state and lane predicates. Transactions belong to callers."""

import enum
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypedDict, Unpack

from sqlalchemy import func, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from voxint.config import Settings
from voxint.db.models import GPU_SEGMENT, POST_SEGMENT, GpuPhaseState, PipelineRun, RunStatus, Stage


class GpuPhase(enum.StrEnum):
    LLM = "llm"
    DRAINING_POST = "draining_post"
    ACQUIRING = "acquiring"
    STARTING_SERVICES = "starting_services"
    AUDIO = "audio"
    DRAINING = "draining"
    STOPPING_SERVICES = "stopping_services"
    RELEASING = "releasing"
    ERROR = "error"


class OperatorRequest(enum.StrEnum):
    AUDIO = "audio"
    RELEASE = "release"


@dataclass(frozen=True)
class GpuPhaseSnapshot:
    id: int
    phase: GpuPhase
    phase_since: datetime
    lease_id: str | None
    lease_expires_at: datetime | None
    last_error: str | None
    failures: int
    retry_after: datetime | None
    operator_request: OperatorRequest | None
    updated_at: datetime
    llm_ready: bool = False
    llm_checked_at: datetime | None = None


def read_phase(session: Session) -> GpuPhaseSnapshot | None:
    row = session.get(GpuPhaseState, 1, populate_existing=True)
    if row is None:
        return None
    return GpuPhaseSnapshot(
        id=row.id,
        phase=GpuPhase(row.phase),
        phase_since=row.phase_since,
        lease_id=row.lease_id,
        lease_expires_at=row.lease_expires_at,
        last_error=row.last_error,
        failures=row.failures,
        retry_after=row.retry_after,
        operator_request=OperatorRequest(row.operator_request) if row.operator_request else None,
        updated_at=row.updated_at,
        llm_ready=row.llm_ready,
        llm_checked_at=row.llm_checked_at,
    )


def gates_enabled(settings: Settings) -> bool:
    return settings.gpu_phase_enabled


def read_phase_if_enabled(session: Session, settings: Settings) -> GpuPhaseSnapshot | None:
    return read_phase(session) if gates_enabled(settings) else None


def llm_freshness_seconds(tick_seconds: int) -> int:
    return max(3 * tick_seconds, 90)


def llm_lane_ready(
    phase: str | None,
    ready: bool,
    checked_at: datetime | None,
    *,
    tick_seconds: int,
    now: datetime | None = None,
) -> bool:
    return (
        phase == GpuPhase.LLM
        and ready
        and checked_at is not None
        and ((now or datetime.now(UTC)) - checked_at).total_seconds()
        <= llm_freshness_seconds(tick_seconds)
    )


def _phase_opens(
    phase: str | None,
    segment: frozenset[Stage],
    settings: Settings,
    llm_ready: bool = False,
    llm_checked_at: datetime | None = None,
    *,
    now: datetime | None = None,
) -> bool:
    # A missing row closes both lanes; the orchestrator's next write repairs it.
    if segment == GPU_SEGMENT:
        return phase == GpuPhase.AUDIO
    if segment == POST_SEGMENT:
        return llm_lane_ready(
            phase, llm_ready, llm_checked_at, tick_seconds=settings.gpu_phase_tick_seconds, now=now
        )
    raise ValueError("Unknown pipeline segment")


def gpu_lane_open(snapshot: GpuPhaseSnapshot | None, settings: Settings) -> bool:
    return lane_open_for_segment(snapshot, settings, GPU_SEGMENT)


def post_lane_open(
    snapshot: GpuPhaseSnapshot | None, settings: Settings, *, now: datetime | None = None
) -> bool:
    return lane_open_for_segment(snapshot, settings, POST_SEGMENT, now=now)


def lane_open_for_segment(
    snapshot: GpuPhaseSnapshot | None,
    settings: Settings,
    segment: frozenset[Stage],
    *,
    now: datetime | None = None,
) -> bool:
    if not gates_enabled(settings):
        return True
    return _phase_opens(
        snapshot.phase if snapshot else None,
        segment,
        settings,
        snapshot.llm_ready if snapshot else False,
        snapshot.llm_checked_at if snapshot else None,
        now=now,
    )


def admit_lane(
    session: Session,
    settings: Settings,
    segment: frozenset[Stage],
    *,
    now: datetime | None = None,
) -> bool:
    """Check admission in the transaction moving QUEUED work to RUNNING.

    FOR SHARE serializes admission with phase and readiness writes by set_phase.
    Issues no query when the feature is off.
    """
    if not gates_enabled(settings):
        return True
    row = session.execute(
        select(GpuPhaseState.phase, GpuPhaseState.llm_ready, GpuPhaseState.llm_checked_at)
        .where(GpuPhaseState.id == 1)
        .with_for_update(read=True)
    ).one_or_none()
    return _phase_opens(
        row.phase if row else None,
        segment,
        settings,
        row.llm_ready if row else False,
        row.llm_checked_at if row else None,
        now=now,
    )


class PhaseFields(TypedDict, total=False):
    llm_ready: bool
    llm_checked_at: datetime | None
    lease_id: str | None
    lease_expires_at: datetime | None
    last_error: str | None
    failures: int
    retry_after: datetime | None
    operator_request: OperatorRequest | None


# updated_at / phase_since of a row created by an operator request before the
# phase task ever ran. Doctor reads it as "the task has not run yet", never as a
# fresh write.
NEVER_TICKED = datetime(1970, 1, 1, tzinfo=UTC)


def _locked_row(
    session: Session, *, now: datetime | None = None, **insert_values: object
) -> GpuPhaseState:
    # Repair an absent singleton, with concurrent writers serialized on the row.
    values: dict[str, object] = {"phase_since": now or datetime.now(UTC), **insert_values}
    session.execute(
        insert(GpuPhaseState).values(id=1, **values).on_conflict_do_nothing(index_elements=["id"])
    )
    return session.execute(
        select(GpuPhaseState)
        .where(GpuPhaseState.id == 1)
        .with_for_update()
        .execution_options(populate_existing=True)
    ).scalar_one()


class _AnyRequest(enum.Enum):
    VALUE = "any"


def set_phase(
    session: Session,
    phase: GpuPhase,
    *,
    now: datetime,
    expected_request: OperatorRequest | _AnyRequest | None = _AnyRequest.VALUE,
    **fields: Unpack[PhaseFields],
) -> None:
    # Compare under the same lock as the write so operator I/O cannot be overwritten.
    row = _locked_row(session, now=now)
    if row.phase != phase:
        row.phase_since = now
    row.phase = phase.value
    row.updated_at = now
    for name, value in fields.items():
        if (
            name == "operator_request"
            and expected_request is not _AnyRequest.VALUE
            and row.operator_request != expected_request
        ):
            continue
        setattr(row, name, value)
    session.flush()


def set_request(session: Session, request: OperatorRequest | None) -> None:
    # A row created here has never been written by the phase task; the sentinel
    # keeps it from looking fresh.
    row = _locked_row(session, updated_at=NEVER_TICKED, phase_since=NEVER_TICKED)
    row.operator_request = request.value if request is not None else None
    # updated_at is left alone: it records the phase task's last write, which
    # is how doctor tells a stopped task from a running one.
    session.flush()


def gpu_lane_demand(session: Session) -> int:
    return session.execute(
        select(func.count())
        .select_from(PipelineRun)
        .where(
            PipelineRun.status == RunStatus.QUEUED.value,
            or_(PipelineRun.current_stage.is_(None), PipelineRun.current_stage.in_(GPU_SEGMENT)),
        )
    ).scalar_one()


def post_lane_queued(session: Session) -> int:
    return session.execute(
        select(func.count())
        .select_from(PipelineRun)
        .where(
            PipelineRun.status == RunStatus.QUEUED.value,
            PipelineRun.current_stage.in_(POST_SEGMENT),
        )
    ).scalar_one()


def _in_flight(session: Session, segment: frozenset[Stage]) -> int:
    stage_filter: ColumnElement[bool] = PipelineRun.current_stage.in_(segment)
    if segment == GPU_SEGMENT:
        stage_filter = or_(PipelineRun.current_stage.is_(None), stage_filter)
    return session.execute(
        select(func.count())
        .select_from(PipelineRun)
        .where(
            PipelineRun.status == RunStatus.RUNNING.value,
            stage_filter,
        )
    ).scalar_one()


def gpu_lane_in_flight(session: Session) -> int:
    """Count RUNNING NULL current_stage only on the GPU side because the engine sets
    it on entry CAS and never clears it mid-pipeline (completed runs have NULL
    with status COMPLETED).
    """
    return _in_flight(session, GPU_SEGMENT)


def post_lane_in_flight(session: Session) -> int:
    """Exclude RUNNING NULL current_stage (GPU only) because the engine sets it on
    entry CAS and never clears it mid-pipeline (completed runs are NULL with status COMPLETED).
    """
    return _in_flight(session, POST_SEGMENT)


def running_llm_jobs(session: Session) -> int:
    """Only active LLM work blocks borrowing; queued work stays behind the gate."""
    return _llm_jobs(session, "running")


def queued_llm_jobs(session: Session) -> int:
    return _llm_jobs(session, "queued")


def _llm_jobs(session: Session, status: str) -> int:
    from voxint.db.models import (
        CleanupJob,
        ResearchJob,
        RunAssetJob,
        TranslationJob,
    )

    return sum(
        session.execute(
            select(func.count()).select_from(model).where(model.status == status)
        ).scalar_one()
        for model in (ResearchJob, RunAssetJob, TranslationJob, CleanupJob)
    )


def llm_unavailable_message(session: Session, settings: Settings) -> str | None:
    """Why inline LLM work must wait, or None when the post lane is open.

    For CLI commands that run LLM jobs synchronously instead of through the
    gated worker tasks. Issues no query when the feature is off."""
    if not gates_enabled(settings):
        return None
    snapshot = read_phase(session)
    if post_lane_open(snapshot, settings):
        return None
    if snapshot is None:
        return (
            "GPU sharing has no phase record yet, so the language model is treated as"
            " unavailable. Try again after the next phase update."
        )
    if snapshot.phase == GpuPhase.LLM:
        return "Waiting for the language model to answer. Language-model work is paused."
    return (
        f"GPU sharing has the GPU in the {snapshot.phase.value} phase, so the language"
        " model is not available. Try again once the phase is back to llm."
    )
