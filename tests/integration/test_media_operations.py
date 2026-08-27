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
    AudioChunk,
    MediaItem,
    MediaOperation,
    MediaOperationFile,
    OperationFileStatus,
    OperationState,
    OperationType,
    PipelineRun,
    RunStatus,
)
from voxint.ingest import ItemTrashedError, OperationInProgressError, submit_media_item
from voxint.media.integrity import openable_current
from voxint.media.operations import OperationRefused
from voxint.media.purge import build_manifest, execute_purge, plan_purge
from voxint.media.reclaim import reclaim_expired_intermediates
from voxint.media.reconcile import reconcile_operations


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


def _trashed_media_with_artifacts(
    session: Session,
    media_root: Path,
) -> tuple[MediaItem, PipelineRun]:
    """Create a trashed media item with derived files on disk."""
    trash_path = "_trash/deadbeef/source.wav"
    media = MediaItem(
        source_path="incoming/source.wav",
        current_path=trash_path,
        trashed_at=datetime.now(tz=UTC),
    )
    session.add(media)
    session.flush()

    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()

    artifact_path = f"artifacts/{run.id}/normalized.wav"
    peaks_path = f"artifacts/{run.id}/peaks.json"
    chunk_path = f"chunks/{run.id}/chunk_0.wav"

    session.add(
        AudioArtifact(
            pipeline_run_id=run.id,
            kind=ArtifactKind.PREPROCESSED_AUDIO.value,
            path=artifact_path,
        )
    )
    session.add(
        AudioArtifact(
            pipeline_run_id=run.id,
            kind=ArtifactKind.WAVEFORM_PEAKS.value,
            path=peaks_path,
        )
    )
    session.add(
        AudioChunk(
            pipeline_run_id=run.id,
            chunk_index=0,
            start_seconds=0.0,
            end_seconds=10.0,
            path=chunk_path,
        )
    )
    session.flush()

    for rel in [trash_path, artifact_path, peaks_path, chunk_path]:
        p = media_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"test data")

    return media, run


def test_plan_purge_requires_trashed(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        media = _existing_media(session)
        session.flush()
        with pytest.raises(OperationRefused, match="not trashed"):
            plan_purge(session, media.id, "worker")


def test_plan_purge_refuses_already_purged(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        media = _existing_media(session)
        media.trashed_at = datetime.now(tz=UTC)
        media.purged_at = datetime.now(tz=UTC)
        session.flush()
        with pytest.raises(OperationRefused, match="already purged"):
            plan_purge(session, media.id, "worker")


def test_build_manifest_enumerates_all_derived(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    with session_factory() as session:
        media, _run = _trashed_media_with_artifacts(session, media_root)
        op = plan_purge(session, media.id, "worker")
        session.commit()

        count = build_manifest(session, op)
        session.commit()

        assert count == 4  # source + artifact + peaks + chunk
        children = (
            session.execute(
                MediaOperationFile.__table__.select().where(
                    MediaOperationFile.operation_id == op.id
                )
            )
            .mappings()
            .all()
        )
        kinds = {c["file_kind"] for c in children}
        assert kinds == {"source", "preprocessed_audio", "peaks", "chunk"}

        second_count = build_manifest(session, op)
        assert second_count == 0


def test_build_manifest_deduplicates_paths(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    with session_factory() as session:
        trash_path = "_trash/op1/source.wav"
        media = MediaItem(
            source_path="incoming/source.wav",
            current_path=trash_path,
            trashed_at=datetime.now(tz=UTC),
        )
        session.add(media)
        session.flush()
        run = PipelineRun(
            media_item_id=media.id, status=RunStatus.COMPLETED.value
        )
        session.add(run)
        session.flush()
        session.add(
            AudioArtifact(
                pipeline_run_id=run.id,
                kind=ArtifactKind.PREPROCESSED_AUDIO.value,
                path=trash_path,
            )
        )
        session.flush()

        p = media_root / trash_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"data")

        op = plan_purge(session, media.id, "worker")
        session.commit()
        count = build_manifest(session, op)
        session.commit()
        assert count == 1


def test_execute_purge_full_flow(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    with session_factory() as session:
        media, run_row = _trashed_media_with_artifacts(session, media_root)
        run_id = run_row.id
        op = plan_purge(session, media.id, "worker")
        session.commit()

        execute_purge(session, media_root, op, "worker")

        session.expire_all()
        refreshed_op = session.get(MediaOperation, op.id)
        assert refreshed_op is not None
        assert refreshed_op.state == OperationState.COMPLETED.value

        refreshed_media = session.get(MediaItem, media.id)
        assert refreshed_media is not None
        assert refreshed_media.purged_at is not None
        assert refreshed_media.current_path is None

        artifacts = (
            session.execute(
                AudioArtifact.__table__.select().where(
                    AudioArtifact.pipeline_run_id == run_id
                )
            )
            .mappings()
            .all()
        )
        assert len(artifacts) == 0

        chunks = (
            session.execute(
                AudioChunk.__table__.select().where(
                    AudioChunk.pipeline_run_id == run_id
                )
            )
            .mappings()
            .all()
        )
        assert len(chunks) == 0

        children = (
            session.execute(
                MediaOperationFile.__table__.select().where(
                    MediaOperationFile.operation_id == op.id
                )
            )
            .mappings()
            .all()
        )
        resolved = {c["status"] for c in children}
        assert resolved <= {
            OperationFileStatus.DONE.value,
            OperationFileStatus.MISSING.value,
        }

        for rel in [
            "_trash/deadbeef/source.wav",
            f"artifacts/{run_id}/normalized.wav",
            f"artifacts/{run_id}/peaks.json",
            f"chunks/{run_id}/chunk_0.wav",
        ]:
            assert not (media_root / rel).exists()

        preserved_run = session.get(PipelineRun, run_id)
        assert preserved_run is not None


def test_execute_purge_partial_failure_schedules_retry(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    with session_factory() as session:
        trash_path = "_trash/partial/source.wav"
        media = MediaItem(
            source_path="incoming/source.wav",
            current_path=trash_path,
            trashed_at=datetime.now(tz=UTC),
        )
        session.add(media)
        session.flush()
        run = PipelineRun(
            media_item_id=media.id, status=RunStatus.COMPLETED.value
        )
        session.add(run)
        session.flush()

        ok_path = f"artifacts/{run.id}/normalized.wav"
        session.add(
            AudioArtifact(
                pipeline_run_id=run.id,
                kind=ArtifactKind.PREPROCESSED_AUDIO.value,
                path=ok_path,
            )
        )
        session.flush()

        (media_root / trash_path).parent.mkdir(parents=True, exist_ok=True)
        (media_root / trash_path).write_bytes(b"source")
        ok_abs = media_root / ok_path
        ok_abs.parent.mkdir(parents=True, exist_ok=True)
        ok_abs.write_bytes(b"artifact")
        ok_abs.parent.chmod(0o444)

        op = plan_purge(session, media.id, "worker")
        session.commit()

        try:
            execute_purge(session, media_root, op, "worker")
        finally:
            ok_abs.parent.chmod(0o755)

        session.expire_all()
        refreshed = session.get(MediaOperation, op.id)
        assert refreshed is not None
        assert refreshed.state == OperationState.AWAITING_RETRY.value
        assert refreshed.error_code == "partial_purge"


def test_reconciler_drives_purge_to_completion(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    with session_factory() as session:
        media, _run = _trashed_media_with_artifacts(session, media_root)
        op = plan_purge(session, media.id, "worker")
        session.flush()
        op.lease_expires_at = datetime.now(tz=UTC) - timedelta(seconds=1)
        session.commit()

    summary = reconcile_operations(
        session_factory, media_root, batch_limit=10
    )
    assert summary.completed == 1

    with session_factory() as session:
        refreshed = session.get(MediaOperation, op.id)
        assert refreshed is not None
        assert refreshed.state == OperationState.COMPLETED.value

        refreshed_media = session.get(MediaItem, media.id)
        assert refreshed_media is not None
        assert refreshed_media.purged_at is not None
