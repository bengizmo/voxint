"""Purge executor for the media operations journal (ADR 0007).

Purge is designed separately from move/trash/restore because a single
origin-destination pair cannot track multi-file deletion.  The executor
builds a durable per-file manifest, unlinks each target (committing
per-child for crash safety), and only after convergence deletes the
artifact database rows and sets ``purged_at``.
"""

from __future__ import annotations

import logging
import uuid
from datetime import timedelta
from pathlib import Path

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from voxint.db.models import (
    AudioArtifact,
    AudioChunk,
    MediaItem,
    MediaOperation,
    MediaOperationFile,
    OperationFileStatus,
    OperationState,
    OperationType,
    PipelineRun,
)
from voxint.media.operations import (
    OperationRefused,
    cas_transition,
    claim_operation,
    has_active_run,
    lock_media_row,
)

logger = logging.getLogger(__name__)

_ARTIFACT_KIND_MAP: dict[str, str] = {
    "preprocessed_audio": "preprocessed_audio",
    "waveform_peaks": "peaks",
    "audio_clip": "audio_clip",
    "chunk": "chunk",
    "transcript_export": "transcript_export",
}


def plan_purge(
    session: Session,
    media_id: uuid.UUID,
    claim_token: str,
) -> MediaOperation:
    """Plan and claim a purge operation.  Item must be trashed."""
    media = lock_media_row(session, media_id)
    if media is None:
        raise OperationRefused("media item does not exist")
    if has_active_run(session, media_id):
        raise OperationRefused("media item has an active run")
    if media.purged_at is not None:
        raise OperationRefused("media item is already purged")
    if media.trashed_at is None:
        raise OperationRefused("media item is not trashed")
    if media.current_path is None:
        raise OperationRefused("media item has no current path")

    operation = MediaOperation(
        id=uuid.uuid4(),
        media_id=media_id,
        operation_type=OperationType.PURGE.value,
        state=OperationState.PLANNED.value,
        origin_path=media.current_path,
        destination_path=None,
        origin_digest=None,
    )
    session.add(operation)
    session.flush()
    if not claim_operation(session, operation.id, claim_token):
        raise OperationRefused("purge operation could not be claimed")
    return operation


def build_manifest(session: Session, operation: MediaOperation) -> int:
    """Build the durable per-file manifest for a purge operation.

    Returns the number of child rows created.  Idempotent: skips if children
    already exist for this operation.
    """
    existing = session.execute(
        select(func.count())
        .select_from(MediaOperationFile)
        .where(MediaOperationFile.operation_id == operation.id)
    ).scalar_one()
    if existing > 0:
        return 0

    children: list[MediaOperationFile] = []
    seen_paths: set[str] = set()

    if operation.origin_path is not None:
        children.append(
            MediaOperationFile(
                operation_id=operation.id,
                file_path=operation.origin_path,
                file_kind="source",
                status=OperationFileStatus.PENDING.value,
            )
        )
        seen_paths.add(operation.origin_path)

    run_ids = list(
        session.execute(
            select(PipelineRun.id).where(
                PipelineRun.media_item_id == operation.media_id
            )
        ).scalars()
    )

    for run_id in run_ids:
        artifacts = (
            session.execute(
                select(AudioArtifact).where(
                    AudioArtifact.pipeline_run_id == run_id
                )
            )
            .scalars()
            .all()
        )
        for artifact in artifacts:
            file_kind = _ARTIFACT_KIND_MAP.get(artifact.kind)
            if file_kind is None or artifact.path in seen_paths:
                continue
            seen_paths.add(artifact.path)
            children.append(
                MediaOperationFile(
                    operation_id=operation.id,
                    file_path=artifact.path,
                    file_kind=file_kind,
                    status=OperationFileStatus.PENDING.value,
                )
            )

        chunks = (
            session.execute(
                select(AudioChunk).where(AudioChunk.pipeline_run_id == run_id)
            )
            .scalars()
            .all()
        )
        for chunk in chunks:
            if chunk.path in seen_paths:
                continue
            seen_paths.add(chunk.path)
            children.append(
                MediaOperationFile(
                    operation_id=operation.id,
                    file_path=chunk.path,
                    file_kind="chunk",
                    status=OperationFileStatus.PENDING.value,
                )
            )

    session.add_all(children)
    session.flush()
    return len(children)


def _confined_path(root: Path, rel_path: str) -> Path | None:
    """Resolve a relative path under *root*, returning None if it escapes."""
    resolved_root = root.resolve()
    candidate = (root / rel_path).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError:
        logger.warning("skipping purge path outside media root: %r", rel_path)
        return None
    return candidate


def _attempt_children(
    session: Session,
    operation_id: uuid.UUID,
    media_root: Path,
    claim_token: str,
) -> None:
    """Attempt all pending/failed children, committing per child."""
    from voxint.media.executor import _lease_still_held

    rows = session.execute(
        select(MediaOperationFile.id, MediaOperationFile.file_path)
        .where(
            MediaOperationFile.operation_id == operation_id,
            MediaOperationFile.status.in_([
                OperationFileStatus.PENDING.value,
                OperationFileStatus.FAILED.value,
            ]),
        )
    ).all()

    for child_id, file_path in rows:
        if not _lease_still_held(session, operation_id, claim_token):
            logger.warning("lease lost during purge children for %s", operation_id)
            return

        confined = _confined_path(media_root, file_path)
        if confined is None:
            session.execute(
                update(MediaOperationFile)
                .where(MediaOperationFile.id == child_id)
                .values(
                    status=OperationFileStatus.FAILED.value,
                    error_detail="path escapes media root",
                )
            )
            session.commit()
            continue

        try:
            confined.unlink()
            new_status = OperationFileStatus.DONE.value
            error = None
        except FileNotFoundError:
            new_status = OperationFileStatus.MISSING.value
            error = None
        except OSError as exc:
            new_status = OperationFileStatus.FAILED.value
            error = str(exc)

        session.execute(
            update(MediaOperationFile)
            .where(MediaOperationFile.id == child_id)
            .values(status=new_status, error_detail=error)
        )
        session.commit()


def _check_convergence(session: Session, operation_id: uuid.UUID) -> bool:
    """Return True when every child is resolved (done or missing)."""
    unresolved = session.execute(
        select(func.count())
        .select_from(MediaOperationFile)
        .where(
            MediaOperationFile.operation_id == operation_id,
            MediaOperationFile.status.in_([
                OperationFileStatus.PENDING.value,
                OperationFileStatus.FAILED.value,
            ]),
        )
    ).scalar_one()
    return unresolved == 0


def _delete_artifact_rows(session: Session, media_id: uuid.UUID) -> None:
    """Delete AudioArtifact and AudioChunk rows for all runs of this item."""
    run_ids = list(
        session.execute(
            select(PipelineRun.id).where(
                PipelineRun.media_item_id == media_id
            )
        ).scalars()
    )
    if not run_ids:
        return
    session.execute(
        delete(AudioChunk).where(AudioChunk.pipeline_run_id.in_(run_ids))
    )
    session.execute(
        delete(AudioArtifact).where(AudioArtifact.pipeline_run_id.in_(run_ids))
    )


def execute_purge(
    session: Session,
    media_root: Path,
    operation: MediaOperation,
    claim_token: str,
) -> None:
    """Drive a purge operation from any non-terminal state to completion.

    Each state block advances to the next in a fallthrough design so the
    function can be entered at any interrupted state.
    """
    from_state = operation.state

    if from_state in {
        OperationState.PLANNED.value,
        OperationState.AWAITING_RETRY.value,
    }:
        build_manifest(session, operation)
        session.commit()

        attempts = operation.attempt_count or 0
        _attempt_children(session, operation.id, media_root, claim_token)

        if not _check_convergence(session, operation.id):
            delay = min(300 * (2**attempts), 3600)
            if not cas_transition(
                session,
                operation.id,
                from_state,
                OperationState.AWAITING_RETRY,
                claim_token,
                error_code="partial_purge",
                attempt_count=attempts + 1,
                last_attempt_at=func.now(),
                next_attempt_at=func.now() + timedelta(seconds=delay),
            ):
                session.rollback()
            else:
                session.commit()
            return

        if not cas_transition(
            session,
            operation.id,
            from_state,
            OperationState.FS_APPLIED,
            claim_token,
        ):
            session.rollback()
            return
        session.commit()
        from_state = OperationState.FS_APPLIED.value

    if from_state == OperationState.FS_APPLIED.value:
        _delete_artifact_rows(session, operation.media_id)
        session.execute(
            update(MediaItem)
            .where(MediaItem.id == operation.media_id)
            .values(purged_at=func.now(), current_path=None)
        )
        if not cas_transition(
            session,
            operation.id,
            OperationState.FS_APPLIED,
            OperationState.DB_APPLIED,
            claim_token,
        ):
            session.rollback()
            return
        session.commit()
        from_state = OperationState.DB_APPLIED.value

    if from_state == OperationState.DB_APPLIED.value:
        if not cas_transition(
            session,
            operation.id,
            OperationState.DB_APPLIED,
            OperationState.COMPLETED,
            claim_token,
        ):
            session.rollback()
            return
        session.commit()
