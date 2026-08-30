"""Crash recovery, concurrency, and path-hardening tests for media operations."""

from __future__ import annotations

import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import cast

import pytest
from sqlalchemy import event, select, text
from sqlalchemy.orm import Session, sessionmaker

from voxint.db.models import (
    OPERATION_TERMINAL_STATES,
    MediaItem,
    MediaOperation,
    MediaOperationFile,
    OperationFileStatus,
    OperationState,
    OperationType,
)
from voxint.ingest import OperationInProgressError, submit_media_item
from voxint.media.executor import execute_operation, plan_operation, plan_trash
from voxint.media.integrity import openable_path, sha256_file
from voxint.media.operations import (
    OperationRefused,
    cas_transition,
    claim_operation,
    temp_path,
)
from voxint.media.reconcile import _process_one, reconcile_operations


def _media_file(media_root: Path) -> Path:
    incoming = media_root / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    source = incoming / "test.wav"
    source.write_bytes(b"test audio content")
    return source


def _expired_trash(
    session: Session,
    *,
    source_path: str = "incoming/test.wav",
) -> tuple[MediaItem, MediaOperation]:
    media = MediaItem(source_path=source_path, current_path=source_path)
    session.add(media)
    session.flush()
    operation = plan_trash(session, media.id, "crashed-worker")
    operation.lease_expires_at = datetime.now(tz=UTC) - timedelta(seconds=1)
    return media, operation


def _assert_completed_trash(
    session_factory: sessionmaker[Session],
    media_root: Path,
    operation_id: uuid.UUID,
) -> None:
    with session_factory() as session:
        operation = session.get(MediaOperation, operation_id)
        assert operation is not None
        assert operation.state == OperationState.COMPLETED.value
        assert operation.destination_path is not None
        media = session.get(MediaItem, operation.media_id)
        assert media is not None
        assert media.current_path == operation.destination_path
        assert media.trashed_at is not None
        assert not (media_root / "incoming/test.wav").exists()
        assert (media_root / operation.destination_path).read_bytes() == (b"test audio content")


def test_crash_after_plan_commit_before_filesystem_publish(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """A planned operation is replayed from scratch after its worker crashes."""
    media_root = tmp_path / "media"
    media_root.mkdir()
    _media_file(media_root)

    with session_factory() as session:
        _media, operation = _expired_trash(session)
        operation_id = operation.id
        destination_path = operation.destination_path
        session.commit()

    assert destination_path is not None
    assert not (media_root / destination_path).exists()
    summary = reconcile_operations(session_factory, media_root, batch_limit=10)

    assert summary.completed == 1
    _assert_completed_trash(session_factory, media_root, operation_id)


def test_crash_after_destination_publish_before_pointer_cas(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """A replayed same-device publication advances the pointer and converges."""
    media_root = tmp_path / "media"
    media_root.mkdir()
    source = _media_file(media_root)

    with session_factory() as session:
        _media, operation = _expired_trash(session)
        operation.origin_digest = sha256_file(source)
        operation_id = operation.id
        assert operation.destination_path is not None
        destination = media_root / operation.destination_path
        destination.parent.mkdir(parents=True)
        destination.hardlink_to(source)
        session.commit()

    assert source.exists() and destination.exists()
    summary = reconcile_operations(session_factory, media_root, batch_limit=10)

    assert summary.completed == 1
    _assert_completed_trash(session_factory, media_root, operation_id)


def test_crash_after_pointer_cas_before_origin_unlink(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """DB_APPLIED with both files removes the origin and completes."""
    media_root = tmp_path / "media"
    media_root.mkdir()
    source = _media_file(media_root)

    with session_factory() as session:
        media, operation = _expired_trash(session)
        operation.origin_digest = sha256_file(source)
        operation.state = OperationState.DB_APPLIED.value
        assert operation.destination_path is not None
        destination = media_root / operation.destination_path
        destination.parent.mkdir(parents=True)
        destination.hardlink_to(source)
        media.current_path = operation.destination_path
        media.trashed_at = datetime.now(tz=UTC)
        operation_id = operation.id
        session.commit()

    assert source.exists() and destination.exists()
    summary = reconcile_operations(session_factory, media_root, batch_limit=10)

    assert summary.completed == 1
    _assert_completed_trash(session_factory, media_root, operation_id)


def test_crash_after_origin_unlink_before_completion_cas(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """DB_APPLIED with only the destination advances to completed."""
    media_root = tmp_path / "media"
    media_root.mkdir()
    source = _media_file(media_root)

    with session_factory() as session:
        media, operation = _expired_trash(session)
        operation.origin_digest = sha256_file(source)
        operation.state = OperationState.DB_APPLIED.value
        assert operation.destination_path is not None
        destination = media_root / operation.destination_path
        destination.parent.mkdir(parents=True)
        source.replace(destination)
        media.current_path = operation.destination_path
        media.trashed_at = datetime.now(tz=UTC)
        operation_id = operation.id
        session.commit()

    assert not source.exists() and destination.exists()
    summary = reconcile_operations(session_factory, media_root, batch_limit=10)

    assert summary.completed == 1
    _assert_completed_trash(session_factory, media_root, operation_id)


def test_purge_crash_after_partial_child_completion(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """A purge resumes pending children without retrying a completed child."""
    media_root = tmp_path / "media"
    media_root.mkdir()
    paths = ["_trash/op/test.wav", "artifacts/a.wav", "chunks/c.wav"]
    for path in paths[1:]:
        target = media_root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.encode())

    with session_factory() as session:
        media = MediaItem(
            source_path="incoming/test.wav",
            current_path=paths[0],
            trashed_at=datetime.now(tz=UTC),
        )
        session.add(media)
        session.flush()
        operation = MediaOperation(
            media_id=media.id,
            operation_type=OperationType.PURGE.value,
            state=OperationState.PLANNED.value,
            origin_path=paths[0],
        )
        session.add(operation)
        session.flush()
        session.add_all(
            [
                MediaOperationFile(
                    operation_id=operation.id,
                    file_path=paths[0],
                    file_kind="source",
                    status=OperationFileStatus.DONE.value,
                ),
                MediaOperationFile(
                    operation_id=operation.id,
                    file_path=paths[1],
                    file_kind="preprocessed_audio",
                    status=OperationFileStatus.PENDING.value,
                ),
                MediaOperationFile(
                    operation_id=operation.id,
                    file_path=paths[2],
                    file_kind="chunk",
                    status=OperationFileStatus.PENDING.value,
                ),
            ]
        )
        operation_id = operation.id
        media_id = media.id
        session.commit()

    summary = reconcile_operations(session_factory, media_root, batch_limit=10)

    assert summary.completed == 1
    assert all(not (media_root / path).exists() for path in paths)
    with session_factory() as session:
        operation_row = session.get(MediaOperation, operation_id)
        media_row = session.get(MediaItem, media_id)
        assert operation_row is not None
        assert operation_row.state == OperationState.COMPLETED.value
        assert media_row is not None and media_row.purged_at is not None
        assert media_row.current_path is None
        assert {child.status for child in operation_row.files} == {OperationFileStatus.DONE.value}


def test_duplicate_claim_attempt_has_exactly_one_winner(
    session_factory: sessionmaker[Session],
) -> None:
    """An operation lease cannot be acquired by two independent sessions."""
    with session_factory() as session:
        media = MediaItem(source_path="incoming/test.wav")
        session.add(media)
        session.flush()
        operation = MediaOperation(
            media_id=media.id,
            operation_type=OperationType.MOVE.value,
            state=OperationState.PLANNED.value,
            origin_path=media.current_path,
            destination_path="archive/test.wav",
        )
        session.add(operation)
        session.commit()
        operation_id = operation.id

    barrier = threading.Barrier(2)
    outcomes: list[bool] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def claim(token: str) -> None:
        try:
            with session_factory() as contender:
                barrier.wait(timeout=10)
                won = claim_operation(contender, operation_id, token)
                contender.commit()
                with lock:
                    outcomes.append(won)
        except BaseException as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=claim, args=(token,)) for token in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert all(not thread.is_alive() for thread in threads)
    assert not errors, errors
    assert sorted(outcomes) == [False, True]


def test_move_vs_rerun_race_blocks_run(
    session_factory: sessionmaker[Session],
) -> None:
    """A committed trash plan prevents a second session from admitting a run."""
    with session_factory() as planner:
        media = MediaItem(source_path="incoming/test.wav")
        planner.add(media)
        planner.flush()
        plan_trash(planner, media.id, "trash-worker")
        planner.commit()

    with (
        session_factory() as submitter,
        pytest.raises(OperationInProgressError, match="active operation"),
    ):
        submit_media_item(submitter, "incoming/test.wav")


def test_reconciler_vs_reconciler_processes_operation_once(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """Concurrent reconciliation passes produce one completion, not two."""
    media_root = tmp_path / "media"
    media_root.mkdir()
    _media_file(media_root)
    with session_factory() as session:
        _media, operation = _expired_trash(session)
        operation_id = operation.id
        session.commit()

    barrier = threading.Barrier(2)
    summaries = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def reconcile() -> None:
        try:
            barrier.wait(timeout=10)
            result = reconcile_operations(session_factory, media_root, batch_limit=10)
            with lock:
                summaries.append(result)
        except BaseException as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=reconcile) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert all(not thread.is_alive() for thread in threads)
    assert not errors, errors
    assert len(summaries) == 2
    assert sum(summary.completed for summary in summaries) == 1
    _assert_completed_trash(session_factory, media_root, operation_id)


def test_stale_cas_rejected(session_factory: sessionmaker[Session]) -> None:
    """A second session cannot repeat a transition from stale state."""
    with session_factory() as session:
        media = MediaItem(source_path="incoming/test.wav")
        session.add(media)
        session.flush()
        operation = MediaOperation(
            media_id=media.id,
            operation_type=OperationType.MOVE.value,
            state=OperationState.PLANNED.value,
            origin_path=media.current_path,
            destination_path="archive/test.wav",
            claimed_by="worker",
            lease_expires_at=datetime.now(tz=UTC) + timedelta(minutes=5),
        )
        session.add(operation)
        session.commit()
        operation_id = operation.id

    with session_factory() as first, session_factory() as stale:
        stale.get(MediaOperation, operation_id)
        assert cas_transition(
            first,
            operation_id,
            OperationState.PLANNED,
            OperationState.FS_APPLIED,
            "worker",
        )
        first.commit()
        assert not cas_transition(
            stale,
            operation_id,
            OperationState.PLANNED,
            OperationState.FS_APPLIED,
            "worker",
        )
        stale.rollback()


@pytest.mark.parametrize("destination", ["/etc/passwd", "../../etc/passwd"])
def test_unsafe_destination_path_rejected(
    session_factory: sessionmaker[Session], destination: str
) -> None:
    """Planning rejects absolute and parent-traversing destinations."""
    with session_factory() as session:
        media = MediaItem(source_path="incoming/test.wav")
        session.add(media)
        session.flush()
        with pytest.raises(OperationRefused, match="destination path"):
            plan_operation(
                session,
                media.id,
                OperationType.MOVE,
                "incoming/test.wav",
                destination,
            )


def test_openable_path_rejects_symlink_escape(tmp_path: Path) -> None:
    """A path component symlinked outside media_root fails closed."""
    media_root = tmp_path / "media"
    outside = tmp_path / "outside"
    media_root.mkdir()
    outside.mkdir()
    (outside / "secret.wav").write_bytes(b"secret")
    (media_root / "escape").symlink_to(outside, target_is_directory=True)

    assert openable_path(media_root, "escape/secret.wav") is None


def test_executor_creates_trash_destination_directory(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """Trash execution creates its operation-owned destination directory."""
    media_root = tmp_path / "media"
    media_root.mkdir()
    _media_file(media_root)
    with session_factory() as session:
        media = MediaItem(source_path="incoming/test.wav")
        session.add(media)
        session.flush()
        operation = plan_trash(session, media.id, "worker")
        session.commit()
        operation_id = operation.id
        destination_path = operation.destination_path

        execute_operation(session, media_root, operation, "worker")

    assert destination_path is not None
    assert PurePosixPath(destination_path).parent == PurePosixPath("_trash", str(operation_id))
    assert (media_root / destination_path).is_file()


def test_reconciler_cleans_preexisting_cross_device_temp(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """A stale operation-owned copy temp is removed before planned replay."""
    media_root = tmp_path / "media"
    media_root.mkdir()
    _media_file(media_root)
    with session_factory() as session:
        _media, operation = _expired_trash(session)
        assert operation.destination_path is not None
        destination_dir = str(PurePosixPath(operation.destination_path).parent)
        stale_temp = media_root / temp_path(operation.id, destination_dir)
        stale_temp.parent.mkdir(parents=True)
        stale_temp.write_bytes(b"partial copy")
        operation_id = operation.id
        session.commit()

    summary = reconcile_operations(session_factory, media_root, batch_limit=10)

    assert summary.completed == 1
    assert not stale_temp.exists()
    _assert_completed_trash(session_factory, media_root, operation_id)


def test_claim_winner_waits_out_transient_lock_holder(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """The claim winner waits for a transient row lock instead of skipping (#346).

    Deterministic replay of the race behind the two-reconciler flake: after the
    winner commits its claim (releasing its row lock), a second session briefly
    holds the row lock, as a losing reconciler's pre-claim select does. The
    winner's post-claim re-select must block until the holder lets go and then
    complete the operation, never return "skipped" while holding the claim.
    """
    media_root = tmp_path / "media"
    media_root.mkdir()
    _media_file(media_root)
    with session_factory() as session:
        _media, operation = _expired_trash(session)
        operation_id = operation.id
        session.commit()

    claim_committed = threading.Event()
    holder_locked = threading.Event()
    release_holder = threading.Event()
    winner_done = threading.Event()
    hook_installed = threading.Event()
    pids: dict[str, int] = {}
    outcomes: list[str] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def instrumented_factory() -> Session:
        # _process_one's first session gets a hook that pauses the winner
        # right after its claim commit, so the holder can take the row lock
        # in exactly the window the race needs.
        session = session_factory()
        if not hook_installed.is_set():
            hook_installed.set()
            pids["winner"] = session.execute(
                text("SELECT pg_backend_pid()")
            ).scalar_one()

            @event.listens_for(session, "after_commit")
            def _pause_after_claim_commit(_session: Session) -> None:
                if claim_committed.is_set():
                    return
                claim_committed.set()
                if not holder_locked.wait(timeout=10):
                    raise RuntimeError("holder never took the row lock")

        return session

    def winner() -> None:
        try:
            outcome = _process_one(
                cast("sessionmaker[Session]", instrumented_factory),
                media_root,
                operation_id,
                300,
            )
            with lock:
                outcomes.append(outcome)
        except BaseException as exc:
            with lock:
                errors.append(exc)
        finally:
            winner_done.set()

    def holder() -> None:
        try:
            with session_factory() as session:
                assert claim_committed.wait(timeout=10)
                session.execute(
                    select(MediaOperation)
                    .where(MediaOperation.id == operation_id)
                    .with_for_update()
                ).scalar_one()
                pids["holder"] = session.execute(
                    text("SELECT pg_backend_pid()")
                ).scalar_one()
                holder_locked.set()
                assert release_holder.wait(timeout=15)
                session.rollback()
        except BaseException as exc:
            with lock:
                errors.append(exc)

    winner_thread = threading.Thread(target=winner)
    holder_thread = threading.Thread(target=holder)
    winner_thread.start()
    holder_thread.start()
    observed_blocked = False
    try:
        assert holder_locked.wait(timeout=10)
        with session_factory() as control:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if winner_done.is_set():
                    break
                blocking = control.execute(
                    text("SELECT pg_blocking_pids(:pid)"),
                    {"pid": pids["winner"]},
                ).scalar_one()
                control.rollback()
                if pids["holder"] in blocking:
                    observed_blocked = True
                    break
                time.sleep(0.02)
    finally:
        release_holder.set()
    winner_thread.join(timeout=15)
    holder_thread.join(timeout=15)

    assert not winner_thread.is_alive() and not holder_thread.is_alive()
    assert not errors, errors
    assert observed_blocked, "winner returned without blocking on the lock holder"
    assert outcomes == ["completed"]
    _assert_completed_trash(session_factory, media_root, operation_id)


def test_claim_winner_skips_after_lease_stolen_during_wait(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """A winner whose claim was stolen while it waited skips, touching nothing.

    Companion to the blocking post-claim re-select (#346): blocking means the
    winner can wake after an arbitrarily long wait. If that wait outlived the
    lease and another reconciler stole the claim, the woken winner holds a
    dead token and must not run the operation's filesystem work against the
    new owner's in-flight pass. DB_APPLIED is the discriminating state: its
    replay unlinks the origin before any claim-guarded CAS, so a stale winner
    that proceeds does observable damage.
    """
    media_root = tmp_path / "media"
    media_root.mkdir()
    source = _media_file(media_root)
    with session_factory() as session:
        media, operation = _expired_trash(session)
        operation.origin_digest = sha256_file(source)
        operation.state = OperationState.DB_APPLIED.value
        assert operation.destination_path is not None
        destination = media_root / operation.destination_path
        destination.parent.mkdir(parents=True)
        destination.hardlink_to(source)
        media.current_path = operation.destination_path
        media.trashed_at = datetime.now(tz=UTC)
        operation_id = operation.id
        session.commit()

    claim_committed = threading.Event()
    stolen = threading.Event()
    hook_installed = threading.Event()
    outcomes: list[str] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def instrumented_factory() -> Session:
        session = session_factory()
        if not hook_installed.is_set():
            hook_installed.set()

            @event.listens_for(session, "after_commit")
            def _pause_after_claim_commit(_session: Session) -> None:
                if claim_committed.is_set():
                    return
                claim_committed.set()
                if not stolen.wait(timeout=10):
                    raise RuntimeError("thief never stole the claim")

        return session

    def winner() -> None:
        try:
            outcome = _process_one(
                cast("sessionmaker[Session]", instrumented_factory),
                media_root,
                operation_id,
                300,
            )
            with lock:
                outcomes.append(outcome)
        except BaseException as exc:
            with lock:
                errors.append(exc)

    winner_thread = threading.Thread(target=winner)
    winner_thread.start()
    try:
        assert claim_committed.wait(timeout=10)
        # While the winner is paused between its claim commit and its
        # re-select, expire its lease and steal the claim, exactly as a
        # concurrent reconciler does after a >lease-duration stall.
        with session_factory() as thief:
            current = thief.get(MediaOperation, operation_id)
            assert current is not None
            current.lease_expires_at = datetime.now(tz=UTC) - timedelta(seconds=1)
            thief.flush()
            assert claim_operation(thief, operation_id, "reconciler:thief", 300)
            thief.commit()
    finally:
        stolen.set()
    winner_thread.join(timeout=15)

    assert not winner_thread.is_alive()
    assert not errors, errors
    assert outcomes == ["skipped"]
    assert source.is_file(), "stale winner must not touch the filesystem"
    with session_factory() as session:
        operation = session.get(MediaOperation, operation_id)
        assert operation is not None
        assert operation.claimed_by == "reconciler:thief"
        assert operation.state not in OPERATION_TERMINAL_STATES
