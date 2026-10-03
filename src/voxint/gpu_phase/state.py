"""Shared-GPU state and lane predicates. Transactions belong to callers."""

import enum
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypedDict, Unpack

from sqlalchemy import func, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

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
    )


def gates_enabled(settings: Settings) -> bool:
    return settings.gpu_phase_enabled


def read_phase_if_enabled(session: Session, settings: Settings) -> GpuPhaseSnapshot | None:
    return read_phase(session) if gates_enabled(settings) else None


def _phase_opens(phase: str | None, segment: frozenset[Stage]) -> bool:
    # A missing row closes both lanes; the orchestrator's next write repairs it.
    if segment == GPU_SEGMENT:
        return phase == GpuPhase.AUDIO
    if segment == POST_SEGMENT:
        return phase == GpuPhase.LLM
    raise ValueError("Unknown pipeline segment")


def gpu_lane_open(snapshot: GpuPhaseSnapshot | None, settings: Settings) -> bool:
    return not gates_enabled(settings) or _phase_opens(
        snapshot.phase if snapshot is not None else None, GPU_SEGMENT
    )


def post_lane_open(snapshot: GpuPhaseSnapshot | None, settings: Settings) -> bool:
    return not gates_enabled(settings) or _phase_opens(
        snapshot.phase if snapshot is not None else None, POST_SEGMENT
    )


def lane_open_for_segment(
    snapshot: GpuPhaseSnapshot | None, settings: Settings, segment: frozenset[Stage]
) -> bool:
    if not gates_enabled(settings):
        return True
    return _phase_opens(snapshot.phase if snapshot is not None else None, segment)


def admit_lane(session: Session, settings: Settings, segment: frozenset[Stage]) -> bool:
    """Lane check for the transaction that moves work from QUEUED to RUNNING.

    Reads the phase under FOR SHARE. ``set_phase`` takes FOR UPDATE, so a phase
    change waits for an admitting transaction to commit (the orchestrator then
    counts the admitted work as in flight), or the admission sees the new phase.
    The early gate read in a worker task cannot give that guarantee on its own:
    a task can pass it, stall, and claim after the orchestrator has drained.
    Call this in the same transaction as the QUEUED to RUNNING write. Issues no
    query when the feature is off."""
    if not gates_enabled(settings):
        return True
    phase = session.execute(
        select(GpuPhaseState.phase).where(GpuPhaseState.id == 1).with_for_update(read=True)
    ).scalar_one_or_none()
    return _phase_opens(phase, segment)


class PhaseFields(TypedDict, total=False):
    lease_id: str | None
    lease_expires_at: datetime | None
    last_error: str | None
    failures: int
    retry_after: datetime | None
    operator_request: OperatorRequest | None


def _locked_row(session: Session, *, now: datetime | None = None) -> GpuPhaseState:
    # Repair an absent singleton, with concurrent writers serialized on the row.
    session.execute(
        insert(GpuPhaseState)
        .values(id=1, phase_since=now or datetime.now(UTC))
        .on_conflict_do_nothing(index_elements=["id"])
    )
    return session.execute(
        select(GpuPhaseState)
        .where(GpuPhaseState.id == 1)
        .with_for_update()
        .execution_options(populate_existing=True)
    ).scalar_one()


def set_phase(
    session: Session, phase: GpuPhase, *, now: datetime, **fields: Unpack[PhaseFields]
) -> None:
    row = _locked_row(session, now=now)
    if row.phase != phase:
        row.phase_since = now
    row.phase = phase.value
    row.updated_at = now
    for name, value in fields.items():
        setattr(row, name, value)
    session.flush()


def set_request(session: Session, request: OperatorRequest | None) -> None:
    row = _locked_row(session)
    row.operator_request = request.value if request is not None else None
    row.updated_at = datetime.now(UTC)
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


def _in_flight(session: Session, segment: frozenset[Stage]) -> int:
    return session.execute(
        select(func.count())
        .select_from(PipelineRun)
        .where(
            PipelineRun.status == RunStatus.RUNNING.value,
            PipelineRun.current_stage.in_(segment),
        )
    ).scalar_one()


def gpu_lane_in_flight(session: Session) -> int:
    return _in_flight(session, GPU_SEGMENT)


def post_lane_in_flight(session: Session) -> int:
    """RUNNING pipeline runs in the post lane. LLM jobs are not counted here."""
    return _in_flight(session, POST_SEGMENT)


def running_llm_jobs(session: Session) -> int:
    """Only active LLM work blocks borrowing; queued work stays behind the gate."""
    from voxint.db.models import (
        ResearchJob,
        ResearchJobStatus,
        RunAssetJob,
        RunAssetJobStatus,
        TranslationJob,
        TranslationJobStatus,
    )

    return sum(
        session.execute(
            select(func.count()).select_from(model).where(model.status == status)
        ).scalar_one()
        for model, status in (
            (ResearchJob, ResearchJobStatus.RUNNING.value),
            (RunAssetJob, RunAssetJobStatus.RUNNING.value),
            (TranslationJob, TranslationJobStatus.RUNNING.value),
        )
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
    return (
        f"GPU sharing has the GPU in the {snapshot.phase.value} phase, so the language"
        " model is not available. Try again once the phase is back to llm."
    )
