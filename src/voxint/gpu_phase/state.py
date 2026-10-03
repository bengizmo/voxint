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


def gpu_lane_open(snapshot: GpuPhaseSnapshot | None, settings: Settings) -> bool:
    return not gates_enabled(settings) or (
        snapshot is not None and snapshot.phase == GpuPhase.AUDIO
    )


def post_lane_open(snapshot: GpuPhaseSnapshot | None, settings: Settings) -> bool:
    return not gates_enabled(settings) or snapshot is None or snapshot.phase == GpuPhase.LLM


def lane_open_for_segment(
    snapshot: GpuPhaseSnapshot | None, settings: Settings, segment: frozenset[Stage]
) -> bool:
    if not gates_enabled(settings):
        return True
    if segment == GPU_SEGMENT:
        return gpu_lane_open(snapshot, settings)
    if segment == POST_SEGMENT:
        return post_lane_open(snapshot, settings)
    raise ValueError("Unknown pipeline segment")


class PhaseFields(TypedDict, total=False):
    lease_id: str | None
    lease_expires_at: datetime | None
    last_error: str | None
    failures: int
    retry_after: datetime | None
    operator_request: OperatorRequest | None


def _locked_row(session: Session) -> GpuPhaseState:
    # Repair an absent singleton, with concurrent writers serialized on the row.
    session.execute(
        insert(GpuPhaseState).values(id=1).on_conflict_do_nothing(index_elements=["id"])
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
    row = _locked_row(session)
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
    return _in_flight(session, POST_SEGMENT)
