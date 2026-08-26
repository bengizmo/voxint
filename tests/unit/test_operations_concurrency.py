"""Unit tests for media-operation concurrency primitives."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session

from voxint.db.models import OperationState
from voxint.media.operations import (
    TransitionError,
    cas_pointer,
    cas_transition,
    claim_operation,
    has_active_operation,
    has_active_run,
)


def _session_with_result(*, rowcount: int = 0, scalar: bool = False) -> MagicMock:
    session = MagicMock(spec=Session)
    session.execute.return_value.rowcount = rowcount
    session.execute.return_value.scalar_one.return_value = scalar
    return session


def test_claim_operation_success() -> None:
    session = _session_with_result(rowcount=1)
    assert claim_operation(session, uuid.uuid4(), "worker") is True


def test_claim_operation_already_held() -> None:
    session = _session_with_result(rowcount=0)
    assert claim_operation(session, uuid.uuid4(), "worker") is False


def test_cas_transition_success() -> None:
    session = _session_with_result(rowcount=1)
    assert (
        cas_transition(
            session,
            uuid.uuid4(),
            OperationState.PLANNED,
            OperationState.FS_APPLIED,
            "worker",
        )
        is True
    )


def test_cas_transition_illegal() -> None:
    session = _session_with_result(rowcount=1)
    with pytest.raises(TransitionError):
        cas_transition(
            session,
            uuid.uuid4(),
            OperationState.COMPLETED,
            OperationState.PLANNED,
            "worker",
        )
    session.execute.assert_not_called()


def test_cas_transition_stale() -> None:
    session = _session_with_result(rowcount=0)
    assert (
        cas_transition(
            session,
            uuid.uuid4(),
            OperationState.PLANNED,
            OperationState.FS_APPLIED,
            "worker",
        )
        is False
    )


def test_cas_pointer_success() -> None:
    session = _session_with_result(rowcount=1)
    assert cas_pointer(session, uuid.uuid4(), "old.wav", "new.wav") is True


def test_cas_pointer_stale() -> None:
    session = _session_with_result(rowcount=0)
    assert cas_pointer(session, uuid.uuid4(), "old.wav", "new.wav") is False


@pytest.mark.parametrize(("active", "expected"), [(True, True), (False, False)])
def test_has_active_operation(active: bool, expected: bool) -> None:
    session = _session_with_result(scalar=active)
    assert has_active_operation(session, uuid.uuid4()) is expected


@pytest.mark.parametrize(("active", "expected"), [(True, True), (False, False)])
def test_has_active_run(active: bool, expected: bool) -> None:
    session = _session_with_result(scalar=active)
    assert has_active_run(session, uuid.uuid4()) is expected
