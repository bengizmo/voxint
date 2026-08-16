"""Soft-archive + derived-media deletion service layer (voxint.ingest.service).

Issue #5, slice 2. Archive is operator-visibility metadata (last-write-wins,
idempotent, terminal-only); derived-media deletion is a separate destructive
action that removes only a run's own AudioArtifact/AudioChunk files and never the
shared MediaItem.source_path. Exercised against real Postgres.
"""

import uuid
from collections.abc import Callable
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from voxint.db.models import (
    STAGE_ORDER,
    ArtifactKind,
    AudioArtifact,
    AudioChunk,
    MediaItem,
    PipelineRun,
    RunStatus,
    Stage,
)
from voxint.ingest import (
    RunMediaNotDeletableError,
    RunNotArchivableError,
    RunNotFoundError,
    archive_run,
    cancel_run,
    delete_run_derived_media,
    submit_media_item,
    unarchive_run,
    unlink_media_paths,
)
from voxint.pipeline.transitions import cas_update_run, next_stage, snapshot


def _make_completed(session: Session, source_path: str) -> uuid.UUID:
    run_id = submit_media_item(session, source_path).id
    session.commit()
    held = snapshot(session.get(PipelineRun, run_id))  # type: ignore[arg-type]
    held = cas_update_run(session, held, status=RunStatus.RUNNING, current_stage=STAGE_ORDER[0])
    while held.current_stage is not STAGE_ORDER[-1]:
        held = cas_update_run(
            session, held, status=RunStatus.RUNNING, current_stage=next_stage(held.current_stage)
        )
    cas_update_run(session, held, status=RunStatus.COMPLETED, current_stage=None)
    session.commit()
    return run_id


def _make_failed(session: Session, source_path: str) -> uuid.UUID:
    run_id = submit_media_item(session, source_path).id
    session.commit()
    held = snapshot(session.get(PipelineRun, run_id))  # type: ignore[arg-type]
    held = cas_update_run(
        session, held, status=RunStatus.RUNNING, current_stage=STAGE_ORDER[0]
    )
    cas_update_run(
        session, held, status=RunStatus.FAILED, current_stage=STAGE_ORDER[0], error="boom"
    )
    session.commit()
    return run_id


def _make_cancelled(session: Session, source_path: str) -> uuid.UUID:
    run_id = submit_media_item(session, source_path).id
    session.commit()
    cancel_run(session, run_id)
    session.commit()
    return run_id


def _make_queued(session: Session, source_path: str) -> uuid.UUID:
    run_id = submit_media_item(session, source_path).id
    session.commit()
    return run_id


# --------------------------------------------------------------------------- #
# archive_run                                                                  #
# --------------------------------------------------------------------------- #


def test_archive_missing_run_raises_not_found(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session, pytest.raises(RunNotFoundError):
        archive_run(session, uuid.uuid4())


@pytest.mark.parametrize("maker", [_make_completed, _make_failed, _make_cancelled])
def test_archive_terminal_run_stamps_archived_at(
    session_factory: sessionmaker[Session],
    maker: Callable[[Session, str], uuid.UUID],
) -> None:
    with session_factory() as session:
        run_id = maker(session, f"incoming/{maker.__name__}.wav")

    with session_factory() as session:
        run = archive_run(session, run_id)
        session.commit()
        assert run.archived_at is not None

    with session_factory() as session:
        stored = session.get(PipelineRun, run_id)
        assert stored is not None and stored.archived_at is not None


def test_archive_queued_run_refused(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id = _make_queued(session, "incoming/q-arch.wav")

    with session_factory() as session:
        with pytest.raises(RunNotArchivableError) as exc:
            archive_run(session, run_id)
        assert exc.value.status is RunStatus.QUEUED
        stored = session.get(PipelineRun, run_id)
        assert stored is not None and stored.archived_at is None  # untouched


def test_archive_running_run_refused(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id = _make_queued(session, "incoming/r-arch.wav")
        held = snapshot(session.get(PipelineRun, run_id))  # type: ignore[arg-type]
        cas_update_run(session, held, status=RunStatus.RUNNING, current_stage=STAGE_ORDER[0])
        session.commit()

    with session_factory() as session:
        with pytest.raises(RunNotArchivableError) as exc:
            archive_run(session, run_id)
        assert exc.value.status is RunStatus.RUNNING


def test_archive_awaiting_adjudication_refused(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = _make_queued(session, "incoming/await-arch.wav")
        held = snapshot(session.get(PipelineRun, run_id))  # type: ignore[arg-type]
        held = cas_update_run(session, held, status=RunStatus.RUNNING, current_stage=STAGE_ORDER[0])
        while held.current_stage is not Stage.DIARIZE_EMBED:
            nxt = next_stage(held.current_stage)
            held = cas_update_run(session, held, status=RunStatus.RUNNING, current_stage=nxt)
        cas_update_run(
            session, held, status=RunStatus.AWAITING_ADJUDICATION, current_stage=Stage.DIARIZE_EMBED
        )
        session.commit()

    with session_factory() as session:
        with pytest.raises(RunNotArchivableError) as exc:
            archive_run(session, run_id)
        assert exc.value.status is RunStatus.AWAITING_ADJUDICATION


def test_archive_is_idempotent(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id = _make_completed(session, "incoming/idem-arch.wav")

    with session_factory() as session:
        first = archive_run(session, run_id)
        session.commit()
        first_stamp = first.archived_at

    with session_factory() as session:
        # A second archive (double-click / stale tab) returns the run unchanged —
        # the original stamp is preserved, not overwritten.
        again = archive_run(session, run_id)
        session.commit()
        assert again.archived_at == first_stamp


def test_archive_does_not_change_status_or_revision(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = _make_completed(session, "incoming/orthogonal.wav")
        before = session.get(PipelineRun, run_id)
        assert before is not None
        status_before, revision_before = before.status, before.revision

    with session_factory() as session:
        archive_run(session, run_id)
        session.commit()

    with session_factory() as session:
        after = session.get(PipelineRun, run_id)
        assert after is not None
        assert after.status == status_before  # archive is orthogonal to status
        assert after.revision == revision_before  # last-write-wins, no CAS bump


# --------------------------------------------------------------------------- #
# unarchive_run                                                                #
# --------------------------------------------------------------------------- #


def test_unarchive_clears_stamp(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id = _make_completed(session, "incoming/unarch.wav")
        archive_run(session, run_id)
        session.commit()

    with session_factory() as session:
        run = unarchive_run(session, run_id)
        session.commit()
        assert run.archived_at is None

    with session_factory() as session:
        stored = session.get(PipelineRun, run_id)
        assert stored is not None and stored.archived_at is None


def test_unarchive_non_archived_is_idempotent(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = _make_completed(session, "incoming/unarch-noop.wav")

    with session_factory() as session:
        run = unarchive_run(session, run_id)  # never archived — no-op success
        session.commit()
        assert run.archived_at is None


def test_unarchive_missing_run_raises_not_found(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session, pytest.raises(RunNotFoundError):
        unarchive_run(session, uuid.uuid4())


# --------------------------------------------------------------------------- #
# delete_run_derived_media                                                     #
# --------------------------------------------------------------------------- #


def _add_derived(
    session: Session, run_id: uuid.UUID, media_root: Path
) -> tuple[Path, Path]:
    """Add one preprocessed-audio artifact + one chunk, with real files on disk."""
    art_rel = f"derived/{run_id}/audio.wav"
    chunk_rel = f"derived/{run_id}/chunk-0.wav"
    for rel in (art_rel, chunk_rel):
        p = media_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"RIFF")
    session.add(
        AudioArtifact(
            pipeline_run_id=run_id, kind=ArtifactKind.PREPROCESSED_AUDIO.value, path=art_rel
        )
    )
    session.add(
        AudioChunk(
            pipeline_run_id=run_id,
            chunk_index=0,
            start_seconds=0.0,
            end_seconds=1.0,
            path=chunk_rel,
        )
    )
    session.commit()
    return media_root / art_rel, media_root / chunk_rel


def test_delete_media_missing_run_raises_not_found(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_factory() as session, pytest.raises(RunNotFoundError):
        delete_run_derived_media(session, uuid.uuid4(), media_root=tmp_path)


def test_delete_media_on_live_run_refused(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_factory() as session:
        run_id = _make_queued(session, "incoming/live-del.wav")

    with session_factory() as session:
        with pytest.raises(RunMediaNotDeletableError) as exc:
            delete_run_derived_media(session, run_id, media_root=tmp_path)
        assert exc.value.status is RunStatus.QUEUED


def test_delete_media_removes_derived_rows_and_returns_paths(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_factory() as session:
        run_id = _make_completed(session, "incoming/del.wav")
        art_file, chunk_file = _add_derived(session, run_id, tmp_path)

    with session_factory() as session:
        plan = delete_run_derived_media(session, run_id, media_root=tmp_path)
        session.commit()
        result = unlink_media_paths(plan.paths)

    assert plan.rows_deleted == 2
    assert set(plan.paths) == {art_file.resolve(), chunk_file.resolve()}
    assert result.files_deleted == 2 and result.files_missing == 0 and result.files_failed == 0
    assert not art_file.exists() and not chunk_file.exists()

    with session_factory() as session:
        assert session.execute(
            select(AudioArtifact).where(AudioArtifact.pipeline_run_id == run_id)
        ).scalars().all() == []
        assert session.execute(
            select(AudioChunk).where(AudioChunk.pipeline_run_id == run_id)
        ).scalars().all() == []


def test_delete_media_never_touches_shared_source(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    # The original source file + its MediaItem row must survive derived deletion —
    # it is shared across runs and only a separate v2 action may remove it.
    source = tmp_path / "incoming" / "shared.wav"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"SOURCE")
    with session_factory() as session:
        run_id = _make_completed(session, "incoming/shared.wav")
        _add_derived(session, run_id, tmp_path)
        media_id = session.get(PipelineRun, run_id).media_item_id  # type: ignore[union-attr]

    with session_factory() as session:
        plan = delete_run_derived_media(session, run_id, media_root=tmp_path)
        session.commit()
        unlink_media_paths(plan.paths)

    assert source.exists()  # source file untouched
    with session_factory() as session:
        media = session.get(MediaItem, media_id)
        assert media is not None and media.source_path == "incoming/shared.wav"


def test_delete_media_is_idempotent_on_missing_files(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_factory() as session:
        run_id = _make_completed(session, "incoming/idem-del.wav")
        _add_derived(session, run_id, tmp_path)

    with session_factory() as session:
        plan = delete_run_derived_media(session, run_id, media_root=tmp_path)
        session.commit()
        first = unlink_media_paths(plan.paths)
        # A retried unlink of the same (already-gone) paths counts them missing.
        again = unlink_media_paths(plan.paths)

    assert first.files_deleted == 2
    assert again.files_deleted == 0 and again.files_missing == 2


def test_delete_media_skips_path_escaping_media_root(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    # A malformed artifact path that escapes MEDIA_ROOT is dropped from the unlink
    # plan (defense in depth) but its row is still removed.
    outside = tmp_path.parent / "escape.wav"
    outside.write_bytes(b"KEEP")
    with session_factory() as session:
        run_id = _make_completed(session, "incoming/escape.wav")
        session.add(
            AudioArtifact(
                pipeline_run_id=run_id,
                kind=ArtifactKind.PREPROCESSED_AUDIO.value,
                path="../escape.wav",
            )
        )
        session.commit()

    with session_factory() as session:
        plan = delete_run_derived_media(session, run_id, media_root=tmp_path)
        session.commit()
        unlink_media_paths(plan.paths)

    assert plan.rows_deleted == 1
    assert plan.paths == ()  # escaping path skipped
    assert outside.exists()  # never touched
    with session_factory() as session:
        assert session.execute(
            select(AudioArtifact).where(AudioArtifact.pipeline_run_id == run_id)
        ).scalars().all() == []


def test_delete_media_on_run_without_artifacts_succeeds_empty(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    # A FAILED-early run with no derived files: success, zero deletes.
    with session_factory() as session:
        run_id = _make_failed(session, "incoming/empty-del.wav")

    with session_factory() as session:
        plan = delete_run_derived_media(session, run_id, media_root=tmp_path)
        session.commit()
        result = unlink_media_paths(plan.paths)

    assert plan.rows_deleted == 0 and plan.paths == ()
    assert result.files_deleted == 0 and result.files_missing == 0
