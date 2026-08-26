"""Concurrency and live-location integration tests for media operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from voxint.db.models import (
    ArtifactKind,
    AudioArtifact,
    MediaItem,
    MediaOperation,
    OperationState,
    OperationType,
    PipelineRun,
    RunStatus,
)
from voxint.ingest import ItemTrashedError, OperationInProgressError, submit_media_item
from voxint.media.integrity import openable_current
from voxint.media.reclaim import reclaim_expired_intermediates


def _existing_media(session: Session, source_path: str = "incoming/source.wav") -> MediaItem:
    media = MediaItem(source_path=source_path)
    session.add(media)
    session.flush()
    return media


def test_run_admission_blocked_by_active_operation(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        media = _existing_media(session)
        session.add(
            MediaOperation(
                media_id=media.id,
                operation_type=OperationType.MOVE.value,
                state=OperationState.PLANNED.value,
            )
        )
        session.flush()
        with pytest.raises(OperationInProgressError, match="has an active operation"):
            submit_media_item(session, media.source_path)


def test_run_admission_blocked_by_trash(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        media = _existing_media(session)
        media.trashed_at = datetime.now(tz=UTC)
        session.flush()
        with pytest.raises(ItemTrashedError, match="is trashed"):
            submit_media_item(session, media.source_path)


def test_run_admission_blocked_by_purge(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        media = _existing_media(session)
        media.purged_at = datetime.now(tz=UTC)
        session.flush()
        with pytest.raises(ItemTrashedError, match="is purged"):
            submit_media_item(session, media.source_path)


def test_openable_current_uses_current_path(tmp_path: Path) -> None:
    source = tmp_path / "original.wav"
    current = tmp_path / "moved.wav"
    source.write_bytes(b"old")
    current.write_bytes(b"live")
    media = MediaItem(source_path=source.name, current_path=current.name)
    assert openable_current(tmp_path, media) == current.resolve()


def test_reclaim_alias_uses_current_path(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    with session_factory() as session:
        media = MediaItem(
            source_path="incoming/original.wav",
            current_path=f"artifacts/{uuid.uuid4()}/moved.wav",
        )
        session.add(media)
        session.flush()
        run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
        session.add(run)
        session.flush()
        run.updated_at = datetime.now(tz=UTC) - timedelta(days=10)
        artifact = AudioArtifact(
            pipeline_run_id=run.id,
            kind=ArtifactKind.PREPROCESSED_AUDIO.value,
            path=media.current_path,
        )
        session.add(artifact)
        session.commit()

        moved = media_root / media.current_path
        moved.parent.mkdir(parents=True)
        moved.write_bytes(b"live source")
        summary = reclaim_expired_intermediates(
            session,
            media_root=media_root,
            cutoff=datetime.now(tz=UTC) - timedelta(days=1),
            batch_limit=100,
            tutorial_run_id=None,
        )

        assert summary.selected == 0
        assert moved.exists()
