"""Read model for the pipeline progress dashboard strip (#423).

Pure, session-in / dataclass-out queries feeding the htmx-polled progress strip
on ``/runs``. Per-stage counts reuse :func:`voxint.api.jobs_query.stage_activity`
(same truth rules); timing and ETA use successful stage attempts only, not all
finished attempts (a failed attempt's wall-clock skews the average unhelpfully).

The strip uses "Current workload" semantics rather than a batch/cohort identity:
outstanding runs are those queued or running and not archived. Elapsed starts at
the oldest outstanding run's ``created_at``. There is no durable batch entity in
the schema, so exact "2 of N complete" counters would require a separate
migration; these counts are advisory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from voxint.api.jobs_query import StageActivity, stage_activity
from voxint.api.stats_query import run_status_counts
from voxint.db.models import (
    STAGE_ORDER,
    PipelineRun,
    RunStatus,
    Stage,
    StageRun,
    StageStatus,
)

_MIN_HISTORY_SAMPLES = 3

_HEURISTIC_GPU_SECONDS: dict[str, float] = {
    Stage.ACQUIRE.value: 45,
    Stage.PREPARE.value: 90,
    Stage.TRANSCRIBE.value: 600,
    Stage.DIARIZE_EMBED.value: 450,
    Stage.ENHANCE_MATCH.value: 120,
    Stage.FINALIZE.value: 30,
}

_COMPUTE_BOUND_STAGES: frozenset[str] = frozenset({
    Stage.PREPARE.value,
    Stage.TRANSCRIBE.value,
    Stage.DIARIZE_EMBED.value,
})

_CPU_TIER_FACTOR = 4.0


def _heuristic_seconds(stage: str, compute_tier: str) -> float:
    base = _HEURISTIC_GPU_SECONDS.get(stage, 120.0)
    if compute_tier == "cpu" and stage in _COMPUTE_BOUND_STAGES:
        return base * _CPU_TIER_FACTOR
    return base


@dataclass(frozen=True)
class StageProgress:
    """One pipeline stage's dashboard state."""

    stage: str
    queued: int
    active: int
    active_started_at: datetime | None
    avg_seconds: float | None
    eta_seconds: float | None
    using_heuristic: bool


@dataclass(frozen=True)
class WorkloadSummary:
    """Aggregate workload counters for the footer."""

    outstanding: int
    completed: int
    failed: int
    total: int
    elapsed_seconds: float | None
    drain_eta_seconds: float | None
    queue_paused: bool
    overrun: bool


@dataclass(frozen=True)
class PipelineDashboardState:
    """Everything the progress strip needs in one object."""

    stages: tuple[StageProgress, ...]
    workload: WorkloadSummary
    generated_at: datetime
    is_idle: bool


def _successful_avg_durations(session: Session) -> dict[str, tuple[int, float]]:
    """Average duration of *successful* stage attempts, keyed by stage value.

    Returns ``{stage: (sample_count, avg_seconds)}``. Only completed attempts
    with non-negative duration count, so failed/skipped/interrupted attempts
    do not distort the ETA.
    """
    seconds_expr = func.extract("epoch", StageRun.finished_at - StageRun.started_at)
    rows = session.execute(
        select(
            StageRun.stage,
            func.count(),
            func.avg(seconds_expr),
        )
        .where(
            StageRun.status == StageStatus.COMPLETED.value,
            StageRun.finished_at.isnot(None),
            StageRun.finished_at >= StageRun.started_at,
        )
        .group_by(StageRun.stage)
    ).all()
    return {stage: (count, float(avg)) for stage, count, avg in rows}


_LIVE_RUN_STATUSES: tuple[str, ...] = (
    RunStatus.QUEUED.value,
    RunStatus.RUNNING.value,
    RunStatus.AWAITING_ADJUDICATION.value,
)


def _active_started_at(session: Session) -> dict[str, datetime]:
    """Earliest running attempt ``started_at`` per stage, for active runs only.

    Anchors on ``current_stage`` (same join as ``stage_activity``) so a stale
    running row at a previous stage cannot seed the timer.
    """
    rows = session.execute(
        select(StageRun.stage, func.min(StageRun.started_at))
        .join(PipelineRun, PipelineRun.id == StageRun.pipeline_run_id)
        .where(
            StageRun.status == StageStatus.RUNNING.value,
            PipelineRun.archived_at.is_(None),
            PipelineRun.status.in_(_LIVE_RUN_STATUSES),
            StageRun.stage == PipelineRun.current_stage,
        )
        .group_by(StageRun.stage)
    ).all()
    return {stage: started for stage, started in rows}


def _oldest_outstanding_created(session: Session) -> datetime | None:
    """``created_at`` of the oldest non-archived outstanding run."""
    return session.execute(
        select(func.min(PipelineRun.created_at)).where(
            PipelineRun.status.in_(
                (RunStatus.QUEUED.value, RunStatus.RUNNING.value)
            ),
            PipelineRun.archived_at.is_(None),
        )
    ).scalar_one_or_none()


def compute_stage_eta(
    avg_seconds: float | None,
    elapsed_seconds: float | None,
    *,
    is_active: bool,
) -> float | None:
    """Per-stage ETA: remaining seconds for the active item, or full avg if queued."""
    if avg_seconds is None:
        return None
    if is_active and elapsed_seconds is not None:
        remaining = avg_seconds - elapsed_seconds
        return max(remaining, 0.0)
    if is_active:
        return avg_seconds
    return avg_seconds


def pipeline_dashboard_state(
    session: Session,
    now: datetime,
    compute_tier: str,
    queue_paused: bool,
) -> PipelineDashboardState:
    """Composite read model for the progress strip."""
    activity: list[StageActivity] = stage_activity(session)
    durations = _successful_avg_durations(session)
    started_at_by_stage = _active_started_at(session)
    status_counts = run_status_counts(session)
    oldest_created = _oldest_outstanding_created(session)

    activity_by_stage = {a.stage: a for a in activity}

    stages: list[StageProgress] = []
    for stage in STAGE_ORDER:
        sv = stage.value
        act = activity_by_stage.get(sv)
        queued = act.queued if act else 0
        active = act.active if act else 0

        sample_count, avg = durations.get(sv, (0, 0.0))
        using_heuristic = sample_count < _MIN_HISTORY_SAMPLES
        avg_seconds: float | None = (
            _heuristic_seconds(sv, compute_tier)
            if using_heuristic
            else avg
        )

        started = started_at_by_stage.get(sv)
        elapsed = (
            (now - started).total_seconds() if started else None
        )

        eta = compute_stage_eta(
            avg_seconds, elapsed, is_active=active > 0
        )

        stages.append(StageProgress(
            stage=sv,
            queued=queued,
            active=active,
            active_started_at=started,
            avg_seconds=avg_seconds,
            eta_seconds=eta,
            using_heuristic=using_heuristic,
        ))

    outstanding = status_counts.get(RunStatus.QUEUED.value, 0) + status_counts.get(
        RunStatus.RUNNING.value, 0
    )
    completed = status_counts.get(RunStatus.COMPLETED.value, 0)
    failed = status_counts.get(RunStatus.FAILED.value, 0)
    total = sum(status_counts.values())

    elapsed_seconds: float | None = None
    if oldest_created is not None:
        elapsed_seconds = (now - oldest_created).total_seconds()

    drain_eta = _estimate_drain(stages, outstanding)

    overrun = (
        elapsed_seconds is not None
        and drain_eta is not None
        and drain_eta <= 0
        and outstanding > 0
    )

    is_idle = outstanding == 0

    return PipelineDashboardState(
        stages=tuple(stages),
        workload=WorkloadSummary(
            outstanding=outstanding,
            completed=completed,
            failed=failed,
            total=total,
            elapsed_seconds=elapsed_seconds,
            drain_eta_seconds=drain_eta,
            queue_paused=queue_paused,
            overrun=overrun,
        ),
        generated_at=now,
        is_idle=is_idle,
    )


def _estimate_drain(
    stages: list[StageProgress], outstanding: int
) -> float | None:
    """Approximate seconds to drain all outstanding work (conservative serial)."""
    if outstanding == 0:
        return None
    total: float = 0.0
    for sp in stages:
        if sp.active > 0 and sp.eta_seconds is not None:
            total += sp.eta_seconds
        if sp.queued > 0 and sp.avg_seconds is not None:
            total += sp.queued * sp.avg_seconds
    return total if total > 0 else None
