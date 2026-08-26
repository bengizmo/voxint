"""The pipeline-run state machine and its compare-and-swap discipline.

Every mutation of a run's (status, current_stage) pair goes through
:func:`cas_update_run`: an UPDATE guarded by the revision the caller last read.
If another worker moved the run first, zero rows match and
:class:`StaleRevisionError` is raised — the caller re-reads and re-decides
instead of clobbering. Human pauses are plain DB state
(``AWAITING_ADJUDICATION``), never a held task.

Validation covers the full ``(status, stage) -> (status, stage)`` tuple, not
just status membership: a run cannot start at the wrong stage, advance
backwards, complete mid-pipeline, or requeue at an unrelated stage.
"""

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import CursorResult, update
from sqlalchemy.orm import Session

from voxint.db.models import STAGE_ORDER, PipelineRun, RunStatus, Stage
from voxint.media.redaction import cap_length

if TYPE_CHECKING:
    from voxint.config import Settings

ALLOWED_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.QUEUED: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED}),
    RunStatus.RUNNING: frozenset(
        {
            RunStatus.RUNNING,  # stage-to-stage advance
            RunStatus.QUEUED,  # segment handoff parks the run for the other lane
            RunStatus.AWAITING_ADJUDICATION,
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.AWAITING_ADJUDICATION: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED}),
    RunStatus.FAILED: frozenset({RunStatus.QUEUED}),  # explicit requeue only
    RunStatus.COMPLETED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}


class InvalidTransitionError(Exception):
    def __init__(self, current: RunStatus, target: RunStatus, detail: str = "") -> None:
        msg = f"illegal run transition {current.value!r} -> {target.value!r}"
        if detail:
            msg = f"{msg}: {detail}"
        super().__init__(msg)
        self.current = current
        self.target = target


class StaleRevisionError(Exception):
    def __init__(self, run_id: uuid.UUID, expected_revision: int) -> None:
        super().__init__(
            f"run {run_id} moved past revision {expected_revision}; re-read and retry"
        )
        self.run_id = run_id
        self.expected_revision = expected_revision


@dataclass(frozen=True)
class RunSnapshot:
    """What a worker holds while deciding a transition."""

    id: uuid.UUID
    status: RunStatus
    current_stage: Stage | None
    revision: int


def snapshot(run: PipelineRun) -> RunSnapshot:
    return RunSnapshot(
        id=run.id,
        status=RunStatus(run.status),
        current_stage=Stage(run.current_stage) if run.current_stage else None,
        revision=run.revision,
    )


def next_stage(current: Stage | None) -> Stage | None:
    """The stage after ``current`` in the canonical order; None when finished."""
    if current is None:
        return STAGE_ORDER[0]
    idx = STAGE_ORDER.index(current)
    return STAGE_ORDER[idx + 1] if idx + 1 < len(STAGE_ORDER) else None


def validate_transition(
    held: RunSnapshot, status: RunStatus, stage: Stage | None
) -> None:
    """Reject any (status, stage) pair the state machine does not define."""
    current, held_stage = held.status, held.current_stage
    if status not in ALLOWED_TRANSITIONS[current]:
        raise InvalidTransitionError(current, status)

    def reject(detail: str) -> InvalidTransitionError:
        return InvalidTransitionError(current, status, detail)

    if status is RunStatus.CANCELLED:
        return  # cancellation may happen at any stage and keeps or clears it freely
    if current is RunStatus.QUEUED and status is RunStatus.RUNNING:
        # fresh run starts at the first stage; requeued run resumes exactly where it stopped
        expected = held_stage or STAGE_ORDER[0]
        if stage is not expected:
            raise reject(f"must start at {expected.value!r}, got {stage!r}")
    elif current is RunStatus.RUNNING and status is RunStatus.RUNNING:
        if held_stage is None or stage is not next_stage(held_stage):
            raise reject(f"advance from {held_stage!r} must go to {next_stage(held_stage)!r}")
    elif current is RunStatus.RUNNING and status is RunStatus.QUEUED:
        # A lane handoff differs deliberately from a failure retry: the stage
        # that just committed is complete, so the other lane resumes at its
        # successor. FAILED -> QUEUED below must instead keep the failed stage.
        handoff_stage = next_stage(held_stage) if held_stage is not None else None
        if handoff_stage is None or stage is not handoff_stage:
            raise reject(
                f"segment handoff from {held_stage!r} must park at next stage "
                f"{handoff_stage!r}, got {stage!r}"
            )
    elif status in (RunStatus.AWAITING_ADJUDICATION, RunStatus.FAILED):
        if held_stage is None or stage is not held_stage:
            raise reject(f"must keep stage {held_stage!r}, got {stage!r}")
    elif current is RunStatus.AWAITING_ADJUDICATION and status is RunStatus.RUNNING:
        if stage is not held_stage:
            raise reject(f"resume must keep stage {held_stage!r}, got {stage!r}")
    elif current is RunStatus.FAILED and status is RunStatus.QUEUED:
        if stage is not held_stage:
            raise reject(f"requeue must keep stage {held_stage!r}, got {stage!r}")
    elif status is RunStatus.COMPLETED:
        if held_stage is not STAGE_ORDER[-1]:
            raise reject(f"cannot complete from stage {held_stage!r}")
        if stage is not None:
            raise reject("completed runs carry no current_stage")


def cas_update_run(
    session: Session,
    held: RunSnapshot,
    *,
    status: RunStatus,
    current_stage: Stage | None,
    error: str | None = None,
    settings: "Settings | None" = None,
) -> RunSnapshot:
    """Apply a validated transition iff the run is still at ``held.revision``.

    When ``settings`` is supplied (the worker paths), a notifiable transition
    also records a webhook outbox row in THIS transaction, so delivery intent is
    atomic with the state change (issue #12). Callers with no settings context
    (tests, ingest, CLI requeue) pass nothing and never emit — those transitions
    are non-notifiable anyway. Emission is a persistence-only insert; no network
    or broker work happens here.
    """
    validate_transition(held, status, current_stage)
    result = cast(
        CursorResult[Any],
        session.execute(
            update(PipelineRun)
            .where(PipelineRun.id == held.id, PipelineRun.revision == held.revision)
            .values(
                status=status.value,
                current_stage=current_stage.value if current_stage else None,
                # General length cap for every PipelineRun.error write; the
                # ACQUIRE stderr tail is already redacted at its raise site.
                error=cap_length(error) if error is not None else None,
                revision=held.revision + 1,
            )
        ),
    )
    if result.rowcount != 1:
        raise StaleRevisionError(held.id, held.revision)
    new_revision = held.revision + 1
    if settings is not None:
        # Local import keeps the transition primitive free of a module-level
        # dependency on the notify subsystem (and any import cycle through it).
        from voxint.notify import record_transition

        record_transition(
            session,
            run_id=held.id,
            status=status,
            transition_revision=new_revision,
            settings=settings,
        )
        # Console activity outbox (issue #162): a completion event announced in
        # THIS transaction, so it commits iff the completion does (and a stale
        # CAS above raises before we get here). Local import for the same
        # cycle-freedom reason as notify. Off unless the flag is on; the only
        # terminal status is COMPLETED.
        if status is RunStatus.COMPLETED and settings.console_activity_enabled:
            from voxint.activity import record_run_completed

            record_run_completed(session, held.id)
    return RunSnapshot(
        id=held.id, status=status, current_stage=current_stage, revision=new_revision
    )
