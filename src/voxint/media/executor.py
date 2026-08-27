"""Filesystem executor for journaled move, trash, and restore operations."""

from __future__ import annotations

import ctypes
import errno
import logging
import os
import shutil
import stat
import sys
import uuid
from datetime import timedelta
from pathlib import Path, PurePosixPath

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from voxint.db.models import MediaItem, MediaOperation, OperationState, OperationType
from voxint.media.integrity import sha256_file
from voxint.media.operations import (
    TRASH_TREE,
    OperationRefused,
    cas_pointer,
    cas_transition,
    claim_operation,
    extract_filename,
    has_active_run,
    lock_media_row,
    temp_path,
    trash_path,
)

logger = logging.getLogger(__name__)

_TRANSIENT_ERRNOS = frozenset({errno.EACCES, errno.EIO, errno.ENOSPC})
_RENAME_NOREPLACE = 1
_AT_FDCWD = -100


def _fsync_directory(dirpath: Path) -> None:
    """Fsync a directory entry."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(dirpath, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _renameat2_noreplace(src: Path, dst: Path) -> None:
    """Atomically rename without replacement on Linux."""
    if not sys.platform.startswith("linux"):
        raise OSError(errno.ENOSYS, "renameat2 is unavailable", dst)

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable", dst)
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(src),
        _AT_FDCWD,
        os.fsencode(dst),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), dst)


def _link_no_clobber(src: Path, dst: Path) -> None:
    """Publish src at dst atomically without replacing dst.

    Uses hard link (fails EEXIST if dst exists).  EXDEV propagates to the
    caller so the cross-device handler takes over.
    """
    os.link(src, dst)


def _publish_same_device(origin: Path, destination: Path) -> None:
    """Publish a same-device destination durably without clobbering."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    _link_no_clobber(origin, destination)
    _fsync_directory(destination.parent)


def _publish_cross_device(
    origin: Path,
    destination: Path,
    temp: Path,
    expected_digest: str | None,
) -> None:
    """Copy through an inode-pinned source and durably publish it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp.parent.mkdir(parents=True, exist_ok=True)

    with origin.open("rb") as source:
        origin_size = os.fstat(source.fileno()).st_size
        with temp.open("xb") as staged:
            shutil.copyfileobj(source, staged)
            staged.flush()
            os.fsync(staged.fileno())

    if temp.stat().st_size != origin_size:
        raise OSError(errno.EIO, "cross-device copy size mismatch", temp)
    if expected_digest is not None and sha256_file(temp) != expected_digest:
        raise OSError(errno.EIO, "cross-device copy digest mismatch", temp)

    os.link(temp, destination)
    _fsync_directory(destination.parent)
    temp.unlink()


def _safe_unlink(path: Path) -> bool:
    """Unlink path, returning whether it existed."""
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def _lease_still_held(
    session: Session,
    operation_id: uuid.UUID,
    claim_token: str,
) -> bool:
    """Re-check that this executor still holds the lease (ADR 0007 §2c).

    Must be called before every destructive filesystem action (origin unlink,
    temp cleanup) so a stale executor whose lease expired does not race with
    the new lease holder.
    """
    from datetime import UTC, datetime

    row = session.execute(
        select(MediaOperation.claimed_by, MediaOperation.lease_expires_at)
        .where(MediaOperation.id == operation_id)
    ).one_or_none()
    if row is None:
        return False
    if row.claimed_by != claim_token:
        return False
    return row.lease_expires_at is None or row.lease_expires_at >= datetime.now(
        tz=UTC
    )


def _plan_operation(
    session: Session,
    media_id: uuid.UUID,
    op_type: OperationType,
    origin_path: str,
    destination_path: str,
    origin_digest: str | None,
    restores_operation_id: uuid.UUID | None,
    claim_token: str | None,
    operation_id: uuid.UUID | None,
) -> MediaOperation:
    media = lock_media_row(session, media_id)
    if media is None:
        raise OperationRefused("media item does not exist")
    if has_active_run(session, media_id):
        raise OperationRefused("media item has an active run")
    if media.purged_at is not None:
        raise OperationRefused("media item is purged")
    if media.trashed_at is not None and op_type != OperationType.RESTORE:
        raise OperationRefused("media item is trashed")
    if media.current_path != origin_path:
        raise OperationRefused("media item path changed while planning operation")

    operation = MediaOperation(
        id=operation_id or uuid.uuid4(),
        media_id=media_id,
        operation_type=op_type.value,
        state=OperationState.PLANNED.value,
        origin_path=origin_path,
        destination_path=destination_path,
        origin_digest=origin_digest,
        restores_operation_id=restores_operation_id,
    )
    session.add(operation)
    session.flush()
    if claim_token is not None and not claim_operation(session, operation.id, claim_token):
        raise OperationRefused("operation could not be claimed")
    return operation


def plan_operation(
    session: Session,
    media_id: uuid.UUID,
    op_type: OperationType,
    origin_path: str,
    destination_path: str,
    origin_digest: str | None = None,
    restores_operation_id: uuid.UUID | None = None,
    claim_token: str | None = None,
) -> MediaOperation:
    """Create and optionally claim a planned media operation."""
    return _plan_operation(
        session,
        media_id,
        op_type,
        origin_path,
        destination_path,
        origin_digest,
        restores_operation_id,
        claim_token,
        None,
    )


def _load_media(session: Session, media_id: uuid.UUID) -> MediaItem:
    media = session.get(MediaItem, media_id)
    if media is None:
        raise OperationRefused("media item does not exist")
    if media.current_path is None:
        raise OperationRefused("media item has no current path")
    return media


def plan_move(
    session: Session,
    media_id: uuid.UUID,
    destination_path: str,
    claim_token: str,
) -> MediaOperation:
    """Plan and claim a move operation."""
    media = _load_media(session, media_id)
    current_path = media.current_path
    assert current_path is not None
    return plan_operation(
        session,
        media_id,
        OperationType.MOVE,
        current_path,
        destination_path,
        claim_token=claim_token,
    )


def plan_trash(
    session: Session,
    media_id: uuid.UUID,
    claim_token: str,
) -> MediaOperation:
    """Plan and claim a move into the operation-owned trash tree."""
    media = _load_media(session, media_id)
    current_path = media.current_path
    assert current_path is not None
    operation_id = uuid.uuid4()
    destination = trash_path(operation_id, extract_filename(current_path))
    return _plan_operation(
        session,
        media_id,
        OperationType.TRASH,
        current_path,
        destination,
        None,
        None,
        claim_token,
        operation_id,
    )


def plan_restore(
    session: Session,
    media_id: uuid.UUID,
    trash_operation_id: uuid.UUID,
    claim_token: str,
) -> MediaOperation:
    """Plan and claim restoration of a completed trash operation."""
    media = _load_media(session, media_id)
    current_path = media.current_path
    assert current_path is not None
    trash_operation = session.execute(
        select(MediaOperation).where(
            MediaOperation.id == trash_operation_id,
            MediaOperation.media_id == media_id,
            MediaOperation.operation_type == OperationType.TRASH.value,
            MediaOperation.state == OperationState.COMPLETED.value,
        )
    ).scalar_one_or_none()
    if trash_operation is None or trash_operation.origin_path is None:
        raise OperationRefused("completed trash operation does not exist")
    return plan_operation(
        session,
        media_id,
        OperationType.RESTORE,
        current_path,
        trash_operation.origin_path,
        restores_operation_id=trash_operation_id,
        claim_token=claim_token,
    )


def _error_code(exc: OSError) -> str:
    if isinstance(exc, FileNotFoundError):
        return "origin_missing"
    if isinstance(exc, PermissionError) or exc.errno == errno.EACCES:
        return "permission_denied"
    if exc.errno == errno.ENOSPC:
        return "no_space"
    if exc.errno == errno.EIO:
        return "io_error"
    return "filesystem_error"


def _is_transient(exc: OSError) -> bool:
    return isinstance(exc, PermissionError) or exc.errno in _TRANSIENT_ERRNOS


def _record_filesystem_error(
    session: Session,
    operation: MediaOperation,
    from_state: OperationState,
    claim_token: str,
    exc: OSError,
) -> None:
    attempts = operation.attempt_count or 0
    if _is_transient(exc):
        delay = min(300 * (2**attempts), 3600)
        target = OperationState.AWAITING_RETRY
        extra = {
            "error_code": _error_code(exc),
            "attempt_count": attempts + 1,
            "last_attempt_at": func.now(),
            "next_attempt_at": func.now() + timedelta(seconds=delay),
        }
    else:
        target = OperationState.FAILED
        extra = {"error_code": _error_code(exc)}

    transitioned = cas_transition(
        session,
        operation.id,
        from_state,
        target,
        claim_token,
        **extra,
    )
    if not transitioned:
        session.rollback()
        logger.warning("lost claim while recording failure for operation %s", operation.id)
        return
    session.commit()


def _destination_is_owned(operation: MediaOperation) -> bool:
    destination = operation.destination_path
    if destination is None:
        return False
    if operation.operation_type == OperationType.TRASH.value:
        expected_prefix = PurePosixPath(TRASH_TREE, str(operation.id))
        path = PurePosixPath(destination)
        return path.parts[: len(expected_prefix.parts)] == expected_prefix.parts
    return destination == operation.destination_path


def _is_publication_replay(operation: MediaOperation, destination: Path) -> bool:
    if operation.origin_digest is None or not _destination_is_owned(operation):
        return False
    try:
        if not stat.S_ISREG(destination.lstat().st_mode):
            return False
    except FileNotFoundError:
        return False
    return sha256_file(destination) == operation.origin_digest


def _fail_destination_collision(
    session: Session,
    operation: MediaOperation,
    claim_token: str,
) -> None:
    transitioned = cas_transition(
        session,
        operation.id,
        OperationState.PLANNED,
        OperationState.FAILED,
        claim_token,
        error_code="destination_exists",
    )
    if not transitioned:
        session.rollback()
        return
    session.commit()


def execute_operation(
    session: Session,
    media_root: Path,
    operation: MediaOperation,
    claim_token: str,
) -> None:
    """Execute a claimed move, trash, or restore operation."""
    if operation.operation_type in {
        OperationType.MOVE.value,
        OperationType.TRASH.value,
        OperationType.RESTORE.value,
    }:
        _execute_move_like(session, media_root, operation, claim_token)
        return
    if operation.operation_type == OperationType.PURGE.value:
        from voxint.media.purge import execute_purge

        execute_purge(session, media_root, operation, claim_token)
        return
    raise OperationRefused(f"unsupported operation type: {operation.operation_type}")


def _execute_move_like(
    session: Session,
    media_root: Path,
    operation: MediaOperation,
    claim_token: str,
) -> None:
    """Drive a move-like operation through publication, pointer CAS, and cleanup."""
    if operation.state != OperationState.PLANNED.value:
        raise OperationRefused(f"operation is not planned: {operation.state}")
    if operation.origin_path is None or operation.destination_path is None:
        raise OperationRefused("move-like operation is missing a path")

    origin = media_root / operation.origin_path
    destination = media_root / operation.destination_path
    destination_dir = str(PurePosixPath(operation.destination_path).parent)
    temp = media_root / temp_path(operation.id, destination_dir)
    from_state = OperationState.PLANNED

    try:
        try:
            _publish_same_device(origin, destination)
        except FileExistsError:
            if not _is_publication_replay(operation, destination):
                _fail_destination_collision(session, operation, claim_token)
                return
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            try:
                _publish_cross_device(
                    origin,
                    destination,
                    temp,
                    operation.origin_digest,
                )
            except FileExistsError:
                if not destination.exists():
                    raise
                _safe_unlink(temp)
                if not _is_publication_replay(operation, destination):
                    _fail_destination_collision(session, operation, claim_token)
                    return
            transitioned = cas_transition(
                session,
                operation.id,
                OperationState.PLANNED,
                OperationState.FS_APPLIED,
                claim_token,
            )
            if not transitioned:
                session.rollback()
                return
            session.commit()
            from_state = OperationState.FS_APPLIED

        pointer_ok = cas_pointer(
            session,
            operation.media_id,
            operation.origin_path,
            operation.destination_path,
        )
        if not pointer_ok:
            cas_transition(
                session,
                operation.id,
                from_state,
                OperationState.FAILED,
                claim_token,
                error_code="cas_conflict",
            )
            session.commit()
            return

        if operation.operation_type == OperationType.TRASH.value:
            session.execute(
                update(MediaItem)
                .where(MediaItem.id == operation.media_id)
                .values(trashed_at=func.now())
            )
        elif operation.operation_type == OperationType.RESTORE.value:
            session.execute(
                update(MediaItem)
                .where(MediaItem.id == operation.media_id)
                .values(trashed_at=None)
            )

        transitioned = cas_transition(
            session,
            operation.id,
            from_state,
            OperationState.DB_APPLIED,
            claim_token,
        )
        if not transitioned:
            session.rollback()
            return
        session.commit()
        from_state = OperationState.DB_APPLIED

        if not _lease_still_held(session, operation.id, claim_token):
            logger.warning("lease lost before cleanup for operation %s", operation.id)
            return
        _safe_unlink(origin)
        _fsync_directory(origin.parent)
        _safe_unlink(temp)

        transitioned = cas_transition(
            session,
            operation.id,
            OperationState.DB_APPLIED,
            OperationState.COMPLETED,
            claim_token,
        )
        if not transitioned:
            session.rollback()
            return
        session.commit()
    except OSError as exc:
        logger.warning("filesystem error executing operation %s: %s", operation.id, exc)
        _record_filesystem_error(session, operation, from_state, claim_token, exc)
