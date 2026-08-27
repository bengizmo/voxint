"""Helpers and concurrency primitives for the media operations journal."""

from __future__ import annotations

import uuid
from datetime import timedelta
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, NamedTuple, cast

from sqlalchemy import CursorResult, exists, func, or_, select, update
from sqlalchemy.orm import Session

from voxint.db.models import (
    OPERATION_TERMINAL_STATES,
    MediaItem,
    MediaOperation,
    OperationState,
    PipelineRun,
)

TRASH_TREE = "_trash"
TEMP_PREFIX = ".voxint-op-"
TEMP_SUFFIX = ".tmp"


class TransitionError(Exception):
    """A state transition was refused by a guard."""


class OperationRefused(Exception):
    """An operation cannot proceed (item is trashed, purged, or has an active op)."""


class FsReality(StrEnum):
    """Filesystem classification for a single path: exists as a regular file,
    absent, or present but not a regular file (directory, symlink target, etc.)."""

    PRESENT = "present"
    ABSENT = "absent"
    NOT_REGULAR = "not_regular"


class FsSnapshot(NamedTuple):
    """Filesystem state observed at reconciliation time."""

    origin: FsReality
    destination: FsReality
    temp: FsReality
    origin_digest: str | None
    dest_digest: str | None


class PointerClass(StrEnum):
    """Where current_path points relative to the operation's recorded paths."""

    ORIGIN = "origin"
    DESTINATION = "destination"
    OTHER = "other"


def claim_operation(
    session: Session,
    operation_id: uuid.UUID,
    claim_token: str,
    lease_duration_seconds: int = 300,
) -> bool:
    """Claim an available operation lease."""
    result = cast(
        CursorResult[Any],
        session.execute(
            update(MediaOperation)
            .where(
                MediaOperation.id == operation_id,
                or_(
                    MediaOperation.claimed_by.is_(None),
                    MediaOperation.lease_expires_at < func.now(),
                ),
            )
            .values(
                claimed_by=claim_token,
                lease_expires_at=func.now()
                + timedelta(seconds=lease_duration_seconds),
            )
        ),
    )
    return result.rowcount == 1


def cas_transition(
    session: Session,
    operation_id: uuid.UUID,
    expected_state: str,
    new_state: str,
    claim_token: str,
    **extra_columns: Any,
) -> bool:
    """Transition an operation if its state and claim still match."""
    guard_transition(expected_state, new_state)
    result = cast(
        CursorResult[Any],
        session.execute(
            update(MediaOperation)
            .where(
                MediaOperation.id == operation_id,
                MediaOperation.state == expected_state,
                MediaOperation.claimed_by == claim_token,
            )
            .values(state=new_state, **extra_columns)
        ),
    )
    return result.rowcount == 1


def cas_pointer(
    session: Session,
    media_id: uuid.UUID,
    expected_path: str,
    new_path: str,
) -> bool:
    """Update a media pointer if its current path still matches."""
    result = cast(
        CursorResult[Any],
        session.execute(
            update(MediaItem)
            .where(MediaItem.id == media_id, MediaItem.current_path == expected_path)
            .values(current_path=new_path)
        ),
    )
    return result.rowcount == 1


def has_active_operation(session: Session, media_id: uuid.UUID) -> bool:
    """Return whether a media operation is non-terminal."""
    stmt = select(
        exists().where(
            MediaOperation.media_id == media_id,
            MediaOperation.state.not_in(("completed", "failed")),
        )
    )
    return bool(session.execute(stmt).scalar_one())


def has_active_run(session: Session, media_id: uuid.UUID) -> bool:
    """Return whether a pipeline run is non-terminal."""
    stmt = select(
        exists().where(
            PipelineRun.media_item_id == media_id,
            PipelineRun.status.not_in(("completed", "failed", "cancelled")),
        )
    )
    return bool(session.execute(stmt).scalar_one())


def lock_media_row(session: Session, media_id: uuid.UUID) -> MediaItem | None:
    """Lock and return a media row."""
    stmt = select(MediaItem).where(MediaItem.id == media_id).with_for_update()
    return session.execute(stmt).scalar_one_or_none()


def trash_path(operation_id: uuid.UUID, filename: str) -> str:
    """Build the trash-tree destination for a media file.

    Returns a media_root-relative POSIX path: ``_trash/{op_id}/{filename}``.
    """
    return str(PurePosixPath(TRASH_TREE, str(operation_id), filename))


def temp_path(operation_id: uuid.UUID, dest_dir: str) -> str:
    """Build the operation-owned temp path for an EXDEV copy.

    Deterministic from the operation id so the reconciler can find and clean
    orphaned temps without a directory scan.

    Returns a media_root-relative POSIX path in the destination directory.
    """
    name = f"{TEMP_PREFIX}{operation_id}{TEMP_SUFFIX}"
    return str(PurePosixPath(dest_dir, name))


def classify_pointer(
    current_path: str | None,
    origin_path: str | None,
    destination_path: str | None,
) -> PointerClass:
    """Classify where current_path points relative to the operation."""
    if current_path is not None and current_path == origin_path:
        return PointerClass.ORIGIN
    if current_path is not None and current_path == destination_path:
        return PointerClass.DESTINATION
    return PointerClass.OTHER


def guard_transition(
    current_state: str,
    target_state: str,
) -> None:
    """Validate that a state transition is legal per the ADR 0007 state machine.

    Legal forward transitions:
    - planned -> fs_applied, db_applied (same-device fold), awaiting_retry, failed
    - fs_applied -> db_applied, awaiting_retry, failed
    - db_applied -> completed, awaiting_retry, failed
    - awaiting_retry -> planned (retry re-enters), fs_applied, failed
    - completed -> (none, terminal)
    - failed -> (none, terminal)
    """
    legal: dict[str, frozenset[str]] = {
        OperationState.PLANNED: frozenset({
            OperationState.FS_APPLIED,
            OperationState.DB_APPLIED,
            OperationState.AWAITING_RETRY,
            OperationState.FAILED,
        }),
        OperationState.FS_APPLIED: frozenset({
            OperationState.DB_APPLIED,
            OperationState.AWAITING_RETRY,
            OperationState.FAILED,
        }),
        OperationState.DB_APPLIED: frozenset({
            OperationState.COMPLETED,
            OperationState.AWAITING_RETRY,
            OperationState.FAILED,
        }),
        OperationState.AWAITING_RETRY: frozenset({
            OperationState.PLANNED,
            OperationState.FS_APPLIED,
            OperationState.FAILED,
        }),
    }
    allowed = legal.get(current_state, frozenset())
    if target_state not in allowed:
        raise TransitionError(
            f"cannot transition from {current_state!r} to {target_state!r}"
        )


def is_terminal(state: str) -> bool:
    """Whether the state is terminal (completed or failed)."""
    return state in OPERATION_TERMINAL_STATES


def is_in_trash_tree(path: str) -> bool:
    """Whether a media_root-relative path is inside the managed trash tree."""
    return PurePosixPath(path).parts[:1] == (TRASH_TREE,)


def extract_filename(path: str) -> str:
    """Extract the filename component from a media_root-relative path."""
    return PurePosixPath(path).name


def is_operation_owned_temp(path: str, operation_id: uuid.UUID) -> bool:
    """Whether a filename matches the operation-owned temp pattern."""
    expected = f"{TEMP_PREFIX}{operation_id}{TEMP_SUFFIX}"
    return PurePosixPath(path).name == expected
