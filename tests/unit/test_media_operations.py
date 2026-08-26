"""Unit tests for media/operations.py pure helpers (ADR 0007, issue #155)."""

from __future__ import annotations

import uuid

import pytest

from voxint.db.models import OperationState
from voxint.media.operations import (
    TRASH_TREE,
    PointerClass,
    TransitionError,
    classify_pointer,
    extract_filename,
    guard_transition,
    is_in_trash_tree,
    is_operation_owned_temp,
    is_terminal,
    temp_path,
    trash_path,
)


class TestTrashPath:
    def test_basic(self) -> None:
        op_id = uuid.UUID("12345678-1234-1234-1234-123456789abc")
        result = trash_path(op_id, "recording.wav")
        assert result == f"{TRASH_TREE}/{op_id}/recording.wav"

    def test_preserves_filename(self) -> None:
        op_id = uuid.uuid4()
        result = trash_path(op_id, "My Recording (2).mp3")
        assert result.endswith("/My Recording (2).mp3")
        assert result.startswith(f"{TRASH_TREE}/")

    def test_uses_posix_separators(self) -> None:
        op_id = uuid.uuid4()
        result = trash_path(op_id, "file.wav")
        assert "\\" not in result


class TestTempPath:
    def test_deterministic(self) -> None:
        op_id = uuid.UUID("abcdef01-2345-6789-abcd-ef0123456789")
        dest_dir = "some/folder"
        result = temp_path(op_id, dest_dir)
        assert result == f"some/folder/.voxint-op-{op_id}.tmp"

    def test_stable_across_calls(self) -> None:
        op_id = uuid.uuid4()
        assert temp_path(op_id, "d") == temp_path(op_id, "d")


class TestClassifyPointer:
    def test_origin(self) -> None:
        assert classify_pointer("a/b.wav", "a/b.wav", "c/d.wav") == PointerClass.ORIGIN

    def test_destination(self) -> None:
        assert classify_pointer("c/d.wav", "a/b.wav", "c/d.wav") == PointerClass.DESTINATION

    def test_other(self) -> None:
        assert classify_pointer("x/y.wav", "a/b.wav", "c/d.wav") == PointerClass.OTHER

    def test_none_current_path(self) -> None:
        assert classify_pointer(None, "a/b.wav", "c/d.wav") == PointerClass.OTHER

    def test_none_origin(self) -> None:
        assert classify_pointer("a/b.wav", None, "c/d.wav") == PointerClass.OTHER

    def test_none_destination(self) -> None:
        assert classify_pointer("a/b.wav", "a/b.wav", None) == PointerClass.ORIGIN


class TestGuardTransition:
    @pytest.mark.parametrize(
        "src, dst",
        [
            (OperationState.PLANNED, OperationState.FS_APPLIED),
            (OperationState.PLANNED, OperationState.DB_APPLIED),
            (OperationState.PLANNED, OperationState.AWAITING_RETRY),
            (OperationState.PLANNED, OperationState.FAILED),
            (OperationState.FS_APPLIED, OperationState.DB_APPLIED),
            (OperationState.FS_APPLIED, OperationState.AWAITING_RETRY),
            (OperationState.FS_APPLIED, OperationState.FAILED),
            (OperationState.DB_APPLIED, OperationState.COMPLETED),
            (OperationState.DB_APPLIED, OperationState.AWAITING_RETRY),
            (OperationState.DB_APPLIED, OperationState.FAILED),
            (OperationState.AWAITING_RETRY, OperationState.PLANNED),
            (OperationState.AWAITING_RETRY, OperationState.FS_APPLIED),
            (OperationState.AWAITING_RETRY, OperationState.FAILED),
        ],
    )
    def test_legal_transitions(self, src: str, dst: str) -> None:
        guard_transition(src, dst)

    @pytest.mark.parametrize(
        "src, dst",
        [
            (OperationState.COMPLETED, OperationState.PLANNED),
            (OperationState.COMPLETED, OperationState.FAILED),
            (OperationState.FAILED, OperationState.PLANNED),
            (OperationState.FAILED, OperationState.COMPLETED),
            (OperationState.PLANNED, OperationState.COMPLETED),
            (OperationState.FS_APPLIED, OperationState.PLANNED),
        ],
    )
    def test_illegal_transitions(self, src: str, dst: str) -> None:
        with pytest.raises(TransitionError):
            guard_transition(src, dst)


class TestIsTerminal:
    def test_completed(self) -> None:
        assert is_terminal(OperationState.COMPLETED) is True

    def test_failed(self) -> None:
        assert is_terminal(OperationState.FAILED) is True

    def test_planned(self) -> None:
        assert is_terminal(OperationState.PLANNED) is False

    def test_awaiting_retry(self) -> None:
        assert is_terminal(OperationState.AWAITING_RETRY) is False


class TestIsInTrashTree:
    def test_trash_path(self) -> None:
        assert is_in_trash_tree(f"{TRASH_TREE}/abc/file.wav") is True

    def test_not_trash(self) -> None:
        assert is_in_trash_tree("incoming/abc/file.wav") is False

    def test_similar_prefix(self) -> None:
        assert is_in_trash_tree("_trashcan/file.wav") is False

    def test_bare_trash_dir(self) -> None:
        assert is_in_trash_tree(f"{TRASH_TREE}/") is True


class TestExtractFilename:
    def test_basic(self) -> None:
        assert extract_filename("folder/sub/file.wav") == "file.wav"

    def test_no_directory(self) -> None:
        assert extract_filename("file.wav") == "file.wav"


class TestIsOperationOwnedTemp:
    def test_match(self) -> None:
        op_id = uuid.uuid4()
        path = f"dest/.voxint-op-{op_id}.tmp"
        assert is_operation_owned_temp(path, op_id) is True

    def test_wrong_id(self) -> None:
        op_id = uuid.uuid4()
        other_id = uuid.uuid4()
        path = f"dest/.voxint-op-{other_id}.tmp"
        assert is_operation_owned_temp(path, op_id) is False

    def test_not_temp(self) -> None:
        op_id = uuid.uuid4()
        assert is_operation_owned_temp("dest/file.wav", op_id) is False
