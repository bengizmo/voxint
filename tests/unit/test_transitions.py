import uuid

import pytest

from voxint.db.models import STAGE_ORDER, RunStatus, Stage
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
    assert next_stage(None) is Stage.PREPARE
    walked = []
    stage = next_stage(None)
    while stage is not None:
        walked.append(stage)
        stage = next_stage(stage)
    assert tuple(walked) == STAGE_ORDER


def test_terminal_states_have_no_exits() -> None:
    assert ALLOWED_TRANSITIONS[RunStatus.COMPLETED] == frozenset()
    assert ALLOWED_TRANSITIONS[RunStatus.CANCELLED] == frozenset()


def test_failed_only_requeues() -> None:
    assert ALLOWED_TRANSITIONS[RunStatus.FAILED] == {RunStatus.QUEUED}


def test_every_status_has_a_policy() -> None:
    assert set(ALLOWED_TRANSITIONS) == set(RunStatus)


def test_fresh_run_must_start_at_first_stage() -> None:
    held = snap(RunStatus.QUEUED, None)
    validate_transition(held, RunStatus.RUNNING, Stage.PREPARE)
    with pytest.raises(InvalidTransitionError):
        validate_transition(held, RunStatus.RUNNING, Stage.FINALIZE)


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
    for bad in (Stage.PREPARE, None):
        with pytest.raises(InvalidTransitionError):
            validate_transition(held, RunStatus.FAILED, bad)


def test_requeue_from_failed_keeps_stage() -> None:
    held = snap(RunStatus.FAILED, Stage.TRANSCRIBE)
    validate_transition(held, RunStatus.QUEUED, Stage.TRANSCRIBE)
    with pytest.raises(InvalidTransitionError):
        validate_transition(held, RunStatus.QUEUED, Stage.FINALIZE)


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
    ):
        validate_transition(snap(status, stage), RunStatus.CANCELLED, None)


def test_status_membership_still_enforced() -> None:
    with pytest.raises(InvalidTransitionError):
        validate_transition(snap(RunStatus.QUEUED, None), RunStatus.COMPLETED, None)


def test_error_types_carry_context() -> None:
    run_id = uuid.uuid4()
    err = StaleRevisionError(run_id, 3)
    assert err.run_id == run_id and err.expected_revision == 3
