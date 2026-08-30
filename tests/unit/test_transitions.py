import uuid

import pytest

from voxint.db.models import GPU_SEGMENT, POST_SEGMENT, STAGE_ORDER, RunStatus, Stage
from voxint.pipeline.transitions import (
    ALLOWED_TRANSITIONS,
    InvalidTransitionError,
    RunSnapshot,
    StaleRevisionError,
    next_stage,
    validate_transition,
)


def snap(status: RunStatus, stage: Stage | None, revision: int = 0) -> RunSnapshot:
    return RunSnapshot(id=uuid.uuid4(), status=status, current_stage=stage, revision=revision)


def test_stage_order_walk() -> None:
    assert next_stage(None) is Stage.ACQUIRE
    assert next_stage(Stage.ACQUIRE) is Stage.PREPARE
    walked = []
    stage = next_stage(None)
    while stage is not None:
        walked.append(stage)
        stage = next_stage(stage)
    assert tuple(walked) == STAGE_ORDER


def test_completed_cancelled_exit_only_to_queued() -> None:
    assert ALLOWED_TRANSITIONS[RunStatus.COMPLETED] == frozenset({RunStatus.QUEUED})
    assert ALLOWED_TRANSITIONS[RunStatus.CANCELLED] == frozenset({RunStatus.QUEUED})


def test_failed_only_requeues() -> None:
    assert ALLOWED_TRANSITIONS[RunStatus.FAILED] == {RunStatus.QUEUED}


def test_every_status_has_a_policy() -> None:
    assert set(ALLOWED_TRANSITIONS) == set(RunStatus)


def test_fresh_run_must_start_at_first_stage() -> None:
    held = snap(RunStatus.QUEUED, None)
    validate_transition(held, RunStatus.RUNNING, Stage.ACQUIRE)
    # PREPARE is no longer the first stage — a fresh run must start at ACQUIRE.
    for bad in (Stage.PREPARE, Stage.FINALIZE):
        with pytest.raises(InvalidTransitionError):
            validate_transition(held, RunStatus.RUNNING, bad)


def test_requeued_run_must_resume_at_its_stage() -> None:
    held = snap(RunStatus.QUEUED, Stage.TRANSCRIBE)
    validate_transition(held, RunStatus.RUNNING, Stage.TRANSCRIBE)
    with pytest.raises(InvalidTransitionError):
        validate_transition(held, RunStatus.RUNNING, Stage.PREPARE)


def test_advance_must_be_exactly_one_stage_forward() -> None:
    held = snap(RunStatus.RUNNING, Stage.PREPARE)
    validate_transition(held, RunStatus.RUNNING, Stage.TRANSCRIBE)
    for bad in (Stage.PREPARE, Stage.DIARIZE_EMBED, None):
        with pytest.raises(InvalidTransitionError):
            validate_transition(held, RunStatus.RUNNING, bad)


def test_acquire_advances_only_to_prepare() -> None:
    held = snap(RunStatus.RUNNING, Stage.ACQUIRE)
    validate_transition(held, RunStatus.RUNNING, Stage.PREPARE)
    for bad in (Stage.ACQUIRE, Stage.TRANSCRIBE, None):
        with pytest.raises(InvalidTransitionError):
            validate_transition(held, RunStatus.RUNNING, bad)


def test_cannot_complete_mid_pipeline() -> None:
    with pytest.raises(InvalidTransitionError):
        validate_transition(snap(RunStatus.RUNNING, Stage.TRANSCRIBE), RunStatus.COMPLETED, None)
    validate_transition(snap(RunStatus.RUNNING, Stage.FINALIZE), RunStatus.COMPLETED, None)
    with pytest.raises(InvalidTransitionError):
        validate_transition(
            snap(RunStatus.RUNNING, Stage.FINALIZE), RunStatus.COMPLETED, Stage.FINALIZE
        )


def test_failure_and_pause_keep_their_stage() -> None:
    held = snap(RunStatus.RUNNING, Stage.DIARIZE_EMBED)
    validate_transition(held, RunStatus.FAILED, Stage.DIARIZE_EMBED)
    validate_transition(held, RunStatus.AWAITING_ADJUDICATION, Stage.DIARIZE_EMBED)
    validate_transition(held, RunStatus.PAUSED, Stage.DIARIZE_EMBED)
    for bad in (Stage.PREPARE, None):
        with pytest.raises(InvalidTransitionError):
            validate_transition(held, RunStatus.FAILED, bad)


def test_requeue_from_failed_keeps_stage_or_restarts() -> None:
    held = snap(RunStatus.FAILED, Stage.TRANSCRIBE)
    validate_transition(held, RunStatus.QUEUED, Stage.TRANSCRIBE)
    validate_transition(held, RunStatus.QUEUED, None)  # restart from scratch
    with pytest.raises(InvalidTransitionError):
        validate_transition(held, RunStatus.QUEUED, Stage.FINALIZE)


def test_running_handoff_parks_at_exactly_next_stage() -> None:
    held = snap(RunStatus.RUNNING, Stage.DIARIZE_EMBED)
    validate_transition(held, RunStatus.QUEUED, Stage.ENHANCE_MATCH)

    for bad in (Stage.DIARIZE_EMBED, Stage.FINALIZE):
        with pytest.raises(InvalidTransitionError, match="must park at next stage"):
            validate_transition(held, RunStatus.QUEUED, bad)


def test_running_final_stage_cannot_handoff_past_pipeline_end() -> None:
    held = snap(RunStatus.RUNNING, Stage.FINALIZE)
    with pytest.raises(InvalidTransitionError, match="must park at next stage"):
        validate_transition(held, RunStatus.QUEUED, None)


def test_execution_segments_partition_pipeline_with_post_as_suffix() -> None:
    assert GPU_SEGMENT.isdisjoint(POST_SEGMENT)
    assert set(Stage) == GPU_SEGMENT | POST_SEGMENT
    assert tuple(stage for stage in STAGE_ORDER if stage in POST_SEGMENT) == tuple(
        STAGE_ORDER[-len(POST_SEGMENT) :]
    )


def test_resume_from_adjudication_keeps_stage() -> None:
    held = snap(RunStatus.AWAITING_ADJUDICATION, Stage.ENHANCE_MATCH)
    validate_transition(held, RunStatus.RUNNING, Stage.ENHANCE_MATCH)
    with pytest.raises(InvalidTransitionError):
        validate_transition(held, RunStatus.RUNNING, Stage.FINALIZE)


def test_cancel_allowed_from_any_live_state() -> None:
    for status, stage in (
        (RunStatus.QUEUED, None),
        (RunStatus.RUNNING, Stage.PREPARE),
        (RunStatus.AWAITING_ADJUDICATION, Stage.ENHANCE_MATCH),
        (RunStatus.PAUSED, Stage.TRANSCRIBE),
    ):
        validate_transition(snap(status, stage), RunStatus.CANCELLED, None)


def test_status_membership_still_enforced() -> None:
    with pytest.raises(InvalidTransitionError):
        validate_transition(snap(RunStatus.QUEUED, None), RunStatus.COMPLETED, None)


def test_pause_from_running_keeps_stage() -> None:
    held = snap(RunStatus.RUNNING, Stage.TRANSCRIBE)
    validate_transition(held, RunStatus.PAUSED, Stage.TRANSCRIBE)
    with pytest.raises(InvalidTransitionError):
        validate_transition(held, RunStatus.PAUSED, Stage.PREPARE)


def test_pause_from_queued_keeps_none() -> None:
    held = snap(RunStatus.QUEUED, None)
    validate_transition(held, RunStatus.PAUSED, None)


def test_resume_from_paused_keeps_stage() -> None:
    held = snap(RunStatus.PAUSED, Stage.TRANSCRIBE)
    validate_transition(held, RunStatus.QUEUED, Stage.TRANSCRIBE)
    with pytest.raises(InvalidTransitionError):
        validate_transition(held, RunStatus.QUEUED, Stage.PREPARE)


def test_paused_cannot_go_directly_to_running() -> None:
    held = snap(RunStatus.PAUSED, Stage.TRANSCRIBE)
    with pytest.raises(InvalidTransitionError):
        validate_transition(held, RunStatus.RUNNING, Stage.TRANSCRIBE)


def test_restart_from_completed_clears_stage() -> None:
    held = snap(RunStatus.COMPLETED, None)
    validate_transition(held, RunStatus.QUEUED, None)
    with pytest.raises(InvalidTransitionError):
        validate_transition(held, RunStatus.QUEUED, Stage.ACQUIRE)


def test_restart_from_cancelled_clears_stage() -> None:
    held = snap(RunStatus.CANCELLED, Stage.PREPARE)
    validate_transition(held, RunStatus.QUEUED, None)
    with pytest.raises(InvalidTransitionError):
        validate_transition(held, RunStatus.QUEUED, Stage.PREPARE)


def test_paused_has_entry_in_allowed() -> None:
    assert RunStatus.PAUSED in ALLOWED_TRANSITIONS
    assert ALLOWED_TRANSITIONS[RunStatus.PAUSED] == frozenset(
        {RunStatus.QUEUED, RunStatus.CANCELLED}
    )


def test_error_types_carry_context() -> None:
    run_id = uuid.uuid4()
    err = StaleRevisionError(run_id, 3)
    assert err.run_id == run_id and err.expected_revision == 3
