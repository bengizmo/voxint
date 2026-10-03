"""Bounded queued-run dispatch with lane filtering before LIMIT."""

import uuid
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import or_, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from voxint.config import Settings
from voxint.db.models import GPU_SEGMENT, POST_SEGMENT, PipelineRun, RunStatus, Stage
from voxint.gpu_phase.state import gpu_lane_open, post_lane_open, read_phase_if_enabled


def open_lanes(session: Session, settings: Settings) -> frozenset[Stage] | None:
    """None means unrestricted, including legacy NULL-stage runs, with no phase query."""
    if not settings.gpu_phase_enabled:
        return None
    snapshot = read_phase_if_enabled(session, settings)
    return (GPU_SEGMENT if gpu_lane_open(snapshot, settings) else frozenset()) | (
        POST_SEGMENT if post_lane_open(snapshot, settings) else frozenset()
    )


def lane_filters(lanes: frozenset[Stage] | None) -> tuple[ColumnElement[bool], ...]:
    """NULL is GPU work; an empty set closes both lanes; None adds no SQL."""
    if lanes is None:
        return ()
    predicate: ColumnElement[bool] = PipelineRun.current_stage.in_(lanes)
    if lanes >= GPU_SEGMENT:
        predicate = or_(PipelineRun.current_stage.is_(None), predicate)
    return (predicate,)


@dataclass(frozen=True)
class DispatchResult:
    selected: int
    dispatched: int


def redispatch_queued_runs(
    session: Session,
    *,
    lanes: frozenset[Stage] | None,
    limit: int,
    publish: Callable[[uuid.UUID, Stage | None], bool],
) -> DispatchResult:
    """Publish oldest eligible rows without mutation. Publisher owns failure policy.

    A False return continues without counting a success; an exception propagates
    so callers that stop on broker failure retain that behavior.
    """
    queued = session.execute(
        select(PipelineRun.id, PipelineRun.current_stage)
        .where(PipelineRun.status == RunStatus.QUEUED.value, *lane_filters(lanes))
        .order_by(PipelineRun.updated_at, PipelineRun.id)
        .limit(limit)
    ).all()
    dispatched = 0
    for run_id, stage_value in queued:
        if publish(run_id, Stage(stage_value) if stage_value else None):
            dispatched += 1
    return DispatchResult(selected=len(queued), dispatched=dispatched)
