"""Reconcile interrupted media filesystem operations."""

from __future__ import annotations

import logging
import os
import stat
import uuid
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path, PurePosixPath
from typing import Literal

from sqlalchemy import exists, func, or_, select, update
from sqlalchemy.orm import Session, sessionmaker

from voxint.db.models import (
    OPERATION_TERMINAL_STATES,
    MediaItem,
    MediaOperation,
    OperationState,
    OperationType,
)
from voxint.media.executor import _fsync_directory, _safe_unlink, execute_operation
from voxint.media.integrity import sha256_file
from voxint.media.operations import (
    TRASH_TREE,
    FsReality,
    FsSnapshot,
    OperationRefused,
    PointerClass,
    TransitionError,
    cas_pointer,
    cas_transition,
    claim_operation,
    classify_pointer,
    extract_filename,
    is_in_trash_tree,
    is_operation_owned_temp,
    temp_path,
)

logger = logging.getLogger(__name__)

_Outcome = Literal["completed", "failed", "retried", "skipped"]


@dataclass(frozen=True)
class ReconcileSummary:
    selected: int = 0
    completed: int = 0
    failed: int = 0
    retried: int = 0
    skipped: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "selected": self.selected,
            "completed": self.completed,
            "failed": self.failed,
            "retried": self.retried,
            "skipped": self.skipped,
        }


def _reality(path: Path | None) -> FsReality:
    if path is None:
        return FsReality.ABSENT
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return FsReality.ABSENT
    return FsReality.PRESENT if stat.S_ISREG(mode) else FsReality.NOT_REGULAR


def _snapshot(
    media_root: Path,
    operation: MediaOperation,
) -> tuple[FsSnapshot, Path | None, Path | None, Path | None]:
    origin = media_root / operation.origin_path if operation.origin_path else None
    destination = (
        media_root / operation.destination_path
        if operation.destination_path
        else None
    )
    temporary: Path | None = None
    if operation.destination_path is not None:
        destination_dir = str(PurePosixPath(operation.destination_path).parent)
        temporary_path = temp_path(operation.id, destination_dir)
        if is_operation_owned_temp(temporary_path, operation.id):
            temporary = media_root / temporary_path

    origin_reality = _reality(origin)
    destination_reality = _reality(destination)
    temp_reality = _reality(temporary)
    snapshot = FsSnapshot(
        origin=origin_reality,
        destination=destination_reality,
        temp=temp_reality,
        origin_digest=(
            sha256_file(origin) if origin_reality == FsReality.PRESENT and origin else None
        ),
        dest_digest=(
            sha256_file(destination)
            if destination_reality == FsReality.PRESENT and destination
            else None
        ),
    )
    return snapshot, origin, destination, temporary


def _destination_owned(operation: MediaOperation) -> bool:
    destination = operation.destination_path
    if destination is None:
        return False
    if operation.operation_type != OperationType.TRASH.value:
        return True
    expected = PurePosixPath(TRASH_TREE, str(operation.id))
    candidate = PurePosixPath(destination)
    return (
        is_in_trash_tree(destination)
        and candidate.parts[: len(expected.parts)] == expected.parts
        and extract_filename(destination) == candidate.name
    )


def _fail(
    session: Session,
    operation: MediaOperation,
    expected_state: str,
    claim_token: str,
    error_code: str,
) -> _Outcome:
    if cas_transition(
        session,
        operation.id,
        expected_state,
        OperationState.FAILED,
        claim_token,
        error_code=error_code,
    ):
        session.commit()
        return "failed"
    session.rollback()
    logger.warning("CAS conflict failing media operation %s", operation.id)
    return "skipped"


def _retry(
    session: Session,
    operation: MediaOperation,
    expected_state: str,
    claim_token: str,
    error_code: str,
) -> _Outcome:
    attempts = operation.attempt_count or 0
    delay = min(300 * (2**attempts), 3600)
    if cas_transition(
        session,
        operation.id,
        expected_state,
        OperationState.AWAITING_RETRY,
        claim_token,
        error_code=error_code,
        attempt_count=attempts + 1,
        last_attempt_at=func.now(),
        next_attempt_at=func.now() + timedelta(seconds=delay),
    ):
        session.commit()
        return "retried"
    session.rollback()
    logger.warning("CAS conflict scheduling retry for media operation %s", operation.id)
    return "skipped"


def _metadata_values(operation: MediaOperation) -> dict[str, object]:
    if operation.operation_type == OperationType.TRASH.value:
        return {"trashed_at": func.now()}
    if operation.operation_type == OperationType.RESTORE.value:
        return {"trashed_at": None}
    return {}


def _complete(
    session: Session,
    operation: MediaOperation,
    expected_state: str,
    claim_token: str,
) -> _Outcome:
    state = expected_state
    if state in {
        OperationState.PLANNED.value,
        OperationState.FS_APPLIED.value,
    }:
        if not cas_transition(
            session,
            operation.id,
            state,
            OperationState.DB_APPLIED,
            claim_token,
        ):
            session.rollback()
            return _fail_after_conflict(session, operation.id, claim_token)
        state = OperationState.DB_APPLIED.value

    metadata = _metadata_values(operation)
    if metadata:
        session.execute(
            update(MediaItem)
            .where(MediaItem.id == operation.media_id)
            .values(**metadata)
        )
    if not cas_transition(
        session,
        operation.id,
        state,
        OperationState.COMPLETED,
        claim_token,
    ):
        session.rollback()
        return _fail_after_conflict(session, operation.id, claim_token)
    session.commit()
    return "completed"


def _fail_after_conflict(
    session: Session,
    operation_id: uuid.UUID,
    claim_token: str,
) -> _Outcome:
    operation = session.get(MediaOperation, operation_id)
    if operation is None or operation.state in OPERATION_TERMINAL_STATES:
        logger.warning("lost operation while recording CAS conflict: %s", operation_id)
        return "skipped"
    return _fail(session, operation, operation.state, claim_token, "cas_conflict")


def _pointer_then_complete(
    session: Session,
    operation: MediaOperation,
    expected_state: str,
    claim_token: str,
    origin: Path | None,
    temporary: Path | None,
) -> _Outcome:
    if operation.origin_path is None or operation.destination_path is None:
        return _fail(
            session, operation, expected_state, claim_token, "ambiguous_state"
        )
    if not cas_pointer(
        session,
        operation.media_id,
        operation.origin_path,
        operation.destination_path,
    ):
        session.rollback()
        return _fail_after_conflict(session, operation.id, claim_token)
    if not cas_transition(
        session,
        operation.id,
        expected_state,
        OperationState.DB_APPLIED,
        claim_token,
    ):
        session.rollback()
        return _fail_after_conflict(session, operation.id, claim_token)
    session.commit()
    if origin is not None and _safe_unlink(origin):
        _fsync_directory(origin.parent)
    if temporary is not None:
        _safe_unlink(temporary)
    return _complete(
        session,
        operation,
        OperationState.DB_APPLIED.value,
        claim_token,
    )


def _clean_and_complete(
    session: Session,
    operation: MediaOperation,
    expected_state: str,
    claim_token: str,
    origin: Path | None,
    temporary: Path | None,
) -> _Outcome:
    if origin is not None and _safe_unlink(origin):
        _fsync_directory(origin.parent)
    if temporary is not None:
        _safe_unlink(temporary)
    return _complete(session, operation, expected_state, claim_token)


def _reenter_retry(
    session: Session,
    operation: MediaOperation,
    pointer: PointerClass,
    claim_token: str,
) -> str | None:
    target = (
        OperationState.FS_APPLIED
        if operation.error_code == "destination_lost"
        or pointer == PointerClass.DESTINATION
        else OperationState.PLANNED
    )
    if not cas_transition(
        session,
        operation.id,
        OperationState.AWAITING_RETRY,
        target,
        claim_token,
        next_attempt_at=None,
    ):
        session.rollback()
        return None
    session.flush()
    operation.state = target.value
    return target.value


def _apply_purge_table(
    session: Session,
    media_root: Path,
    operation: MediaOperation,
    claim_token: str,
) -> _Outcome:
    """Reconcile a purge operation per the ADR 0007 purge table."""
    from voxint.media.purge import execute_purge

    execute_purge(session, media_root, operation, claim_token)
    session.expire_all()
    refreshed = session.get(MediaOperation, operation.id)
    if refreshed is None or refreshed.state not in OPERATION_TERMINAL_STATES:
        if (
            refreshed is not None
            and refreshed.state == OperationState.AWAITING_RETRY.value
        ):
            return "retried"
        return "skipped"
    if refreshed.state == OperationState.COMPLETED.value:
        return "completed"
    return "failed"


def _apply_table(
    session: Session,
    media_root: Path,
    operation: MediaOperation,
    claim_token: str,
) -> _Outcome:
    media = session.get(MediaItem, operation.media_id)
    pointer = classify_pointer(
        media.current_path if media is not None else None,
        operation.origin_path,
        operation.destination_path,
    )
    state = operation.state
    if state == OperationState.AWAITING_RETRY.value:
        reentered = _reenter_retry(session, operation, pointer, claim_token)
        if reentered is None:
            return _fail_after_conflict(session, operation.id, claim_token)
        state = reentered

    snapshot, origin, destination, temporary = _snapshot(media_root, operation)
    origin_exists = snapshot.origin == FsReality.PRESENT
    dest_exists = snapshot.destination == FsReality.PRESENT
    temp_exists = snapshot.temp == FsReality.PRESENT
    expected_digest = operation.origin_digest or snapshot.origin_digest
    digest_matches = (
        expected_digest is not None and snapshot.dest_digest == expected_digest
    )

    if state == OperationState.PLANNED.value:
        if pointer == PointerClass.OTHER:
            return _fail(session, operation, state, claim_token, "superseded")
        if pointer == PointerClass.DESTINATION:
            if dest_exists and digest_matches:
                return _complete(session, operation, state, claim_token)
            return _fail(session, operation, state, claim_token, "pointer_dangling")
        if origin_exists and not dest_exists:
            if temp_exists and temporary is not None:
                _safe_unlink(temporary)
            execute_operation(session, media_root, operation, claim_token)
            session.expire_all()
            refreshed = session.get(MediaOperation, operation.id)
            if refreshed is None or refreshed.state not in OPERATION_TERMINAL_STATES:
                if (
                    refreshed is not None
                    and refreshed.state == OperationState.AWAITING_RETRY.value
                ):
                    return "retried"
                return "skipped"
            if refreshed.state == OperationState.COMPLETED.value:
                return "completed"
            return "failed"
        if origin_exists and dest_exists:
            if digest_matches and _destination_owned(operation):
                return _pointer_then_complete(
                    session, operation, state, claim_token, origin, temporary
                )
            return _fail(session, operation, state, claim_token, "destination_exists")
        if not origin_exists and not dest_exists:
            if temp_exists and temporary is not None:
                _safe_unlink(temporary)
            return _retry(session, operation, state, claim_token, "origin_missing")
        if digest_matches:
            return _pointer_then_complete(
                session, operation, state, claim_token, origin, temporary
            )
        return _fail(session, operation, state, claim_token, "ambiguous_state")

    if state == OperationState.FS_APPLIED.value:
        if pointer == PointerClass.OTHER:
            if temporary is not None:
                _safe_unlink(temporary)
            return _fail(session, operation, state, claim_token, "superseded")
        if pointer == PointerClass.DESTINATION:
            if not dest_exists:
                return _fail(
                    session, operation, state, claim_token, "pointer_dangling"
                )
            if not digest_matches:
                return _fail(session, operation, state, claim_token, "digest_mismatch")
            return _clean_and_complete(
                session, operation, state, claim_token, origin, temporary
            )
        if dest_exists:
            if not digest_matches:
                return _fail(session, operation, state, claim_token, "digest_mismatch")
            return _pointer_then_complete(
                session, operation, state, claim_token, origin, temporary
            )
        if temp_exists and temporary is not None:
            temp_digest = sha256_file(temporary)
            if expected_digest is None or temp_digest != expected_digest:
                _safe_unlink(temporary)
                return _fail(session, operation, state, claim_token, "temp_corrupt")
            assert destination is not None
            try:
                os.link(temporary, destination)
                _fsync_directory(destination.parent)
            except FileExistsError:
                pass
            published_digest = (
                sha256_file(destination)
                if _reality(destination) == FsReality.PRESENT
                else None
            )
            if published_digest != expected_digest:
                return _fail(session, operation, state, claim_token, "digest_mismatch")
            return _pointer_then_complete(
                session, operation, state, claim_token, origin, temporary
            )
        if origin_exists and snapshot.origin_digest == expected_digest:
            return _retry(session, operation, state, claim_token, "destination_lost")
        return _fail(session, operation, state, claim_token, "both_absent")

    if state == OperationState.DB_APPLIED.value:
        if pointer != PointerClass.DESTINATION:
            return _fail(session, operation, state, claim_token, "superseded")
        if not dest_exists:
            return _fail(session, operation, state, claim_token, "pointer_dangling")
        return _clean_and_complete(
            session, operation, state, claim_token, origin, temporary
        )
    return _fail(session, operation, state, claim_token, "superseded")


def _newer_operation_exists(session: Session, operation: MediaOperation) -> bool:
    return bool(
        session.execute(
            select(
                exists().where(
                    MediaOperation.media_id == operation.media_id,
                    MediaOperation.created_at > operation.created_at,
                    MediaOperation.state.not_in(OPERATION_TERMINAL_STATES),
                )
            )
        ).scalar_one()
    )


def _process_one(
    session_factory: sessionmaker[Session],
    media_root: Path,
    operation_id: uuid.UUID,
    lease_duration_seconds: int,
) -> _Outcome:
    claim_token = f"reconciler:{uuid.uuid4()}"
    with session_factory() as session:
        operation = session.execute(
            select(MediaOperation)
            .where(MediaOperation.id == operation_id)
            .with_for_update(skip_locked=True)
        ).scalar_one_or_none()
        if operation is None or operation.state in OPERATION_TERMINAL_STATES:
            return "skipped"
        if not claim_operation(
            session, operation.id, claim_token, lease_duration_seconds
        ):
            session.rollback()
            return "skipped"
        session.commit()
        # Blocking on purpose: after winning the claim, any concurrent lock
        # holder is transient (a losing reconciler about to roll back, or the
        # batch sweep). SKIP LOCKED here made the winner skip its own claimed
        # row and strand it until lease expiry (#346). A wedged holder (an
        # idle-in-transaction session) would stall this pass instead, which
        # beats stranding the claim.
        # populate_existing: the identity map would otherwise hand back the
        # pre-commit instance and these checks would read stale attributes,
        # not the row this FOR UPDATE just locked.
        operation = session.execute(
            select(MediaOperation)
            .where(MediaOperation.id == operation_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        ).scalar_one_or_none()
        if (
            operation is None
            or operation.state in OPERATION_TERMINAL_STATES
            # Ownership can be gone by the time the lock is granted: if the
            # wait outlived the lease, another reconciler stole the claim.
            # Proceeding with a dead token would run the pre-CAS filesystem
            # mutations against the new owner's in-flight work.
            or operation.claimed_by != claim_token
        ):
            return "skipped"
        if _newer_operation_exists(session, operation):
            return _fail(
                session, operation, operation.state, claim_token, "superseded"
            )
        try:
            if operation.operation_type == OperationType.PURGE.value:
                return _apply_purge_table(
                    session, media_root, operation, claim_token
                )
            return _apply_table(session, media_root, operation, claim_token)
        except (OSError, OperationRefused, TransitionError) as exc:
            session.rollback()
            logger.warning("could not reconcile media operation %s: %s", operation.id, exc)
            with session_factory() as failure_session:
                current = failure_session.get(MediaOperation, operation.id)
                if current is None or current.state in OPERATION_TERMINAL_STATES:
                    return "skipped"
                if not claim_operation(
                    failure_session,
                    current.id,
                    claim_token,
                    lease_duration_seconds,
                ) and current.claimed_by != claim_token:
                    failure_session.rollback()
                    return "skipped"
                if current.state == OperationState.AWAITING_RETRY.value:
                    media = failure_session.get(MediaItem, current.media_id)
                    pointer = classify_pointer(
                        media.current_path if media is not None else None,
                        current.origin_path,
                        current.destination_path,
                    )
                    target = _reenter_retry(
                        failure_session, current, pointer, claim_token
                    )
                    if target is None:
                        return "skipped"
                    current.state = target
                return _retry(
                    failure_session,
                    current,
                    current.state,
                    claim_token,
                    "filesystem_error",
                )


def reconcile_operations(
    session_factory: sessionmaker[Session],
    media_root: Path,
    *,
    batch_limit: int = 50,
    lease_duration_seconds: int = 300,
) -> ReconcileSummary:
    """Process one oldest-first batch of non-terminal media operations."""
    if not media_root.is_dir():
        logger.warning("media root is unavailable: %s", media_root)
        return ReconcileSummary()
    if batch_limit <= 0:
        return ReconcileSummary()

    with session_factory() as session:
        operation_ids = list(
            session.execute(
                select(MediaOperation.id)
                .where(
                    MediaOperation.state.not_in(OPERATION_TERMINAL_STATES),
                    or_(
                        MediaOperation.state != OperationState.AWAITING_RETRY.value,
                        MediaOperation.next_attempt_at <= func.now(),
                    ),
                )
                .order_by(MediaOperation.created_at.asc())
                .limit(batch_limit)
                .with_for_update(skip_locked=True)
            ).scalars()
        )
        session.rollback()

    counts: dict[_Outcome, int] = {
        "completed": 0,
        "failed": 0,
        "retried": 0,
        "skipped": 0,
    }
    for operation_id in operation_ids:
        outcome = _process_one(
            session_factory, media_root, operation_id, lease_duration_seconds
        )
        counts[outcome] += 1
    return ReconcileSummary(
        selected=len(operation_ids),
        completed=counts["completed"],
        failed=counts["failed"],
        retried=counts["retried"],
        skipped=counts["skipped"],
    )
