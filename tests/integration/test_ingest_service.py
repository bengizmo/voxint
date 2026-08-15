"""The shared DB-only submission + requeue service (voxint.ingest.service).

Exercised against real Postgres — the service owns no broker, so these tests
never touch Redis; the caller's lazy publish is out of scope here.
"""

import threading
import time
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from voxint.db.models import STAGE_ORDER, MediaItem, PipelineRun, RunStatus, Stage
from voxint.ingest import (
    MissingStageError,
    RunNotCancellableError,
    RunNotFailedError,
    RunNotFoundError,
    UploadConflictError,
    UploadValidationError,
    UrlValidationError,
    cancel_run,
    requeue_failed_run,
    submit_media_item,
    submit_media_item_if_new,
    submit_url,
)
from voxint.pipeline.transitions import (
    RunSnapshot,
    StaleRevisionError,
    cas_update_run,
    next_stage,
    snapshot,
)


def _drive_to_failed(session: Session, run_id: uuid.UUID, stage: Stage) -> RunSnapshot:
    """Walk a QUEUED run to FAILED at ``stage`` through real CAS transitions.

    Using the actual state machine (rather than hand-setting columns) means the
    run's ``revision`` reflects the path it took, so requeue/CAS assertions test
    a state the machine can genuinely produce. Works from a fresh run or one
    just requeued (QUEUED with its stage already held).
    """
    held = snapshot(session.get(PipelineRun, run_id))  # type: ignore[arg-type]
    held = cas_update_run(
        session, held, status=RunStatus.RUNNING, current_stage=held.current_stage or STAGE_ORDER[0]
    )
    while held.current_stage is not stage:
        nxt = next_stage(held.current_stage)
        assert nxt is not None, f"{stage!r} not reachable in STAGE_ORDER"
        held = cas_update_run(session, held, status=RunStatus.RUNNING, current_stage=nxt)
    held = cas_update_run(
        session, held, status=RunStatus.FAILED, current_stage=stage, error="boom"
    )
    session.commit()
    return held


def _corrupt_to_failed_without_stage(session: Session, run_id: uuid.UUID) -> None:
    """Fabricate the impossible FAILED-with-no-stage state via a direct write.

    The state machine can never produce this, so we bypass it deliberately to
    prove requeue refuses to guess. This is the ONLY test that side-steps CAS.
    """
    run = session.get(PipelineRun, run_id)
    assert run is not None
    run.status = RunStatus.FAILED.value
    run.current_stage = None
    session.commit()


def test_submit_media_item_creates_media_and_queued_run(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run = submit_media_item(session, "incoming/a.wav")
        session.commit()
        run_id = run.id

    with session_factory() as session:
        media = session.execute(select(MediaItem)).scalar_one()
        assert media.source_path == "incoming/a.wav"
        stored = session.get(PipelineRun, run_id)
        assert stored is not None
        assert stored.status == RunStatus.QUEUED.value
        assert stored.current_stage is None  # fresh run resolves to STAGE_ORDER[0]
        assert stored.media_item_id == media.id


def test_submit_media_item_reuses_media_but_mints_new_run(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        first = submit_media_item(session, "incoming/a.wav").id
        second = submit_media_item(session, "incoming/a.wav").id
        session.commit()

    assert first != second
    with session_factory() as session:
        assert len(session.execute(select(MediaItem)).scalars().all()) == 1
        assert len(session.execute(select(PipelineRun)).scalars().all()) == 2


def test_submit_media_item_if_new_creates_once_then_skips(
    session_factory: sessionmaker[Session],
) -> None:
    """The scan-confirm primitive: a fresh path queues one run; an existing path
    returns None (no duplicate run), unlike submit_media_item which always mints one."""
    with session_factory() as session:
        run = submit_media_item_if_new(session, "scan/a.wav")
        assert run is not None
        session.commit()
        run_id = run.id

    with session_factory() as session:
        # Same path again → skip, so a double-clicked/re-scanned confirm can't spam runs.
        assert submit_media_item_if_new(session, "scan/a.wav") is None
        session.commit()

    with session_factory() as session:
        assert len(session.execute(select(MediaItem)).scalars().all()) == 1
        runs = session.execute(select(PipelineRun)).scalars().all()
        assert len(runs) == 1 and runs[0].id == run_id


def test_submit_media_item_if_new_recovers_from_concurrent_insert(
    session_factory: sessionmaker[Session],
) -> None:
    """Two racers on the same brand-new path: the loser's SAVEPOINT conflict is
    reported as None (skip), never an error, and only one run is created."""
    path = f"scan/{uuid.uuid4()}.wav"
    barrier = threading.Barrier(2)
    outcomes: dict[str, bool] = {}
    errors: list[BaseException] = []
    lock = threading.Lock()

    def racer(key: str) -> None:
        try:
            with session_factory() as session:
                barrier.wait(timeout=10)
                run = submit_media_item_if_new(session, path)
                session.commit()
                with lock:
                    outcomes[key] = run is not None
        except BaseException as exc:  # surfaced to the test body
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=racer, args=(k,)) for k in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert not errors, errors
    assert sorted(outcomes.values()) == [False, True]  # exactly one created a run
    with session_factory() as session:
        assert len(session.execute(select(PipelineRun)).scalars().all()) == 1


def test_submit_media_item_recovers_from_concurrent_insert(
    session_factory: sessionmaker[Session],
) -> None:
    """A concurrent creator of the same brand-new path forces the second
    submission's INSERT to conflict on the UNIQUE constraint. Savepoint recovery
    must adopt the winner's MediaItem and still mint this submission's own run —
    one MediaItem, two distinct runs, no error.

    The winner holds the unique-index lock (flushed, uncommitted) while the loser
    is released, so the loser's INSERT blocks then conflicts on commit: this
    drives the recovery branch deterministically rather than by lucky timing.
    """
    path = "incoming/race.wav"
    loser_released = threading.Event()
    loser_out: dict[str, object] = {}

    def loser() -> None:
        loser_released.wait(timeout=10)
        try:
            with session_factory() as session:
                loser_out["run_id"] = submit_media_item(session, path).id
                session.commit()
        except Exception as exc:  # pragma: no cover - only on a real failure
            loser_out["error"] = exc

    thread = threading.Thread(target=loser)
    thread.start()
    with session_factory() as winner:
        winner_run = submit_media_item(winner, path).id  # INSERT holds the index lock
        loser_released.set()
        time.sleep(0.5)  # let the loser reach its blocking INSERT before we commit
        winner.commit()  # release the lock -> loser conflicts -> recovery re-reads
    thread.join(timeout=20)

    assert "error" not in loser_out, f"loser raised: {loser_out.get('error')}"
    assert loser_out["run_id"] != winner_run
    with session_factory() as session:
        media = (
            session.execute(select(MediaItem).where(MediaItem.source_path == path))
            .scalars()
            .all()
        )
        assert len(media) == 1
        assert len(session.execute(select(PipelineRun)).scalars().all()) == 2


def test_requeue_missing_run_raises_not_found(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session, pytest.raises(RunNotFoundError):
        requeue_failed_run(session, uuid.uuid4())


def test_requeue_non_failed_run_raises_not_failed(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = submit_media_item(session, "incoming/q.wav").id
        session.commit()
    with session_factory() as session:
        with pytest.raises(RunNotFailedError) as exc:
            requeue_failed_run(session, run_id)
        assert exc.value.status is RunStatus.QUEUED


def test_requeue_failed_without_stage_raises_missing_stage(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = submit_media_item(session, "incoming/corrupt.wav").id
        session.commit()
        _corrupt_to_failed_without_stage(session, run_id)
    with session_factory() as session, pytest.raises(MissingStageError):
        requeue_failed_run(session, run_id)


def test_requeue_failed_run_returns_to_queued_at_same_stage(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = submit_media_item(session, "incoming/r.wav").id
        session.commit()
        failed = _drive_to_failed(session, run_id, Stage.TRANSCRIBE)
        prior_revision = failed.revision

    with session_factory() as session:
        result = requeue_failed_run(session, run_id)
        session.commit()
        assert result.status is RunStatus.QUEUED
        assert result.current_stage is Stage.TRANSCRIBE
        assert result.revision == prior_revision + 1

    with session_factory() as session:
        stored = session.get(PipelineRun, run_id)
        assert stored is not None
        assert stored.status == RunStatus.QUEUED.value
        assert stored.current_stage == Stage.TRANSCRIBE.value


def test_requeue_with_stale_expected_revision_rejects(
    session_factory: sessionmaker[Session],
) -> None:
    """A caller acting on a revision the run has moved past is rejected before any
    write — models a stale browser tab requeuing a run that changed underneath it."""
    with session_factory() as session:
        run_id = submit_media_item(session, "incoming/stale.wav").id
        session.commit()
        failed = _drive_to_failed(session, run_id, Stage.PREPARE)
        live_revision = failed.revision

    with session_factory() as session:
        with pytest.raises(StaleRevisionError):
            requeue_failed_run(session, run_id, expected_revision=live_revision - 1)
        # A rejected requeue writes nothing — the run stays FAILED.
        stored = session.get(PipelineRun, run_id)
        assert stored is not None
        assert stored.status == RunStatus.FAILED.value

    # The matching revision is accepted.
    with session_factory() as session:
        result = requeue_failed_run(session, run_id, expected_revision=live_revision)
        session.commit()
        assert result.status is RunStatus.QUEUED


def test_requeue_two_operators_second_sees_run_no_longer_failed(
    session_factory: sessionmaker[Session],
) -> None:
    """Two operators requeue the same FAILED run from the same revision: exactly
    one wins; the other is cleanly told the run is no longer FAILED (not a crash,
    not a double-enqueue). Proves the CAS serialises concurrent requeues."""
    with session_factory() as session:
        run_id = submit_media_item(session, "incoming/two.wav").id
        session.commit()
    with session_factory() as session:
        rev = _drive_to_failed(session, run_id, Stage.PREPARE).revision

    first = session_factory()
    second = session_factory()
    try:
        won = requeue_failed_run(first, run_id, expected_revision=rev)
        first.commit()
        assert won.status is RunStatus.QUEUED
        # Second operator held the same (now stale) revision; its fresh read sees
        # the run already QUEUED, so it is refused as not-failed rather than
        # requeued a second time.
        with pytest.raises(RunNotFailedError):
            requeue_failed_run(second, run_id, expected_revision=rev)
    finally:
        first.close()
        second.close()


# --- submit_url: URL ingestion registration (slice 6b) ------------------------
# submit_url stays broker-free like the rest of the service: it only registers a
# MediaItem (source_url set, source_path pre-assigned) + a QUEUED run; the worker's
# ACQUIRE stage does the yt-dlp download later (slice 6c). No file is written here.

_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_submit_url_creates_media_with_source_url_and_queued_run(
    session_factory: sessionmaker[Session],
) -> None:
    submission_id = str(uuid.uuid4())
    with session_factory() as session:
        run = submit_url(session, url=_URL, submission_id=submission_id)
        session.commit()
        run_id = run.id

    with session_factory() as session:
        media = session.execute(select(MediaItem)).scalar_one()
        assert media.source_url == _URL
        # source_path is uuid-namespaced (unique) and not yet materialized on disk.
        assert media.source_path == f"incoming/{uuid.UUID(submission_id).hex}/source"
        assert media.size_bytes is None and media.sha256 is None
        stored = session.get(PipelineRun, run_id)
        assert stored is not None
        assert stored.status == RunStatus.QUEUED.value
        assert stored.current_stage is None  # fresh run resolves to STAGE_ORDER[0] = acquire
        assert stored.media_item_id == media.id


def test_submit_url_distinct_submission_ids_mint_distinct_runs(
    session_factory: sessionmaker[Session],
) -> None:
    """source_url is non-unique: the SAME url under two submission_ids yields two
    independent immutable MediaItems + runs (distinct uuid-namespaced paths)."""
    with session_factory() as session:
        first = submit_url(session, url=_URL, submission_id=str(uuid.uuid4())).id
        second = submit_url(session, url=_URL, submission_id=str(uuid.uuid4())).id
        session.commit()

    assert first != second
    with session_factory() as session:
        assert len(session.execute(select(MediaItem)).scalars().all()) == 2
        assert len(session.execute(select(PipelineRun)).scalars().all()) == 2


def test_submit_url_same_submission_id_same_url_is_idempotent(
    session_factory: sessionmaker[Session],
) -> None:
    """A form re-POST (same server-issued submission_id, same url) returns the run
    created the first time — no duplicate MediaItem, no duplicate run."""
    submission_id = str(uuid.uuid4())
    with session_factory() as session:
        first = submit_url(session, url=_URL, submission_id=submission_id).id
        session.commit()
    with session_factory() as session:
        second = submit_url(session, url=_URL, submission_id=submission_id).id
        session.commit()

    assert first == second
    with session_factory() as session:
        assert len(session.execute(select(MediaItem)).scalars().all()) == 1
        assert len(session.execute(select(PipelineRun)).scalars().all()) == 1


def test_submit_url_same_submission_id_different_url_conflicts(
    session_factory: sessionmaker[Session],
) -> None:
    """Reusing a submission_id with a DIFFERENT url is refused (409): silently
    acquiring a URL the operator never pasted is the divergence we reject, and the
    first submission's url wins. Nothing is written on the conflicting attempt."""
    submission_id = str(uuid.uuid4())
    with session_factory() as session:
        submit_url(session, url=_URL, submission_id=submission_id)
        session.commit()
    with session_factory() as session:
        with pytest.raises(UploadConflictError):
            submit_url(session, url="https://example.com/other.mp3", submission_id=submission_id)
        session.rollback()

    with session_factory() as session:
        media = session.execute(select(MediaItem)).scalar_one()
        assert media.source_url == _URL  # unchanged — first url wins
        assert len(session.execute(select(PipelineRun)).scalars().all()) == 1


def test_submit_url_replay_heals_media_with_no_run(
    session_factory: sessionmaker[Session],
) -> None:
    """A stored MediaItem whose run never materialized (a partial first attempt)
    is healed on replay: the same url mints the missing run rather than 409-ing."""
    submission_id = str(uuid.uuid4())
    rel = f"incoming/{uuid.UUID(submission_id).hex}/source"
    with session_factory() as session:
        session.add(MediaItem(source_path=rel, source_url=_URL))  # media, but no run
        session.commit()

    with session_factory() as session:
        run = submit_url(session, url=_URL, submission_id=submission_id)
        session.commit()
        run_id = run.id

    with session_factory() as session:
        assert len(session.execute(select(MediaItem)).scalars().all()) == 1
        stored = session.get(PipelineRun, run_id)
        assert stored is not None
        assert stored.status == RunStatus.QUEUED.value


def test_submit_url_invalid_url_writes_nothing(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        with pytest.raises(UrlValidationError):
            submit_url(session, url="ftp://example.com/f.mp3", submission_id=str(uuid.uuid4()))
        session.rollback()
    with session_factory() as session:
        assert session.execute(select(MediaItem)).scalars().all() == []
        assert session.execute(select(PipelineRun)).scalars().all() == []


def test_submit_url_invalid_submission_id_writes_nothing(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        with pytest.raises(UploadValidationError):
            submit_url(session, url=_URL, submission_id="not-a-uuid")
        session.rollback()
    with session_factory() as session:
        assert session.execute(select(MediaItem)).scalars().all() == []
        assert session.execute(select(PipelineRun)).scalars().all() == []


# --- cancel_run (issue #5) ----------------------------------------------------


def _drive_to_running(session: Session, run_id: uuid.UUID, stage: Stage) -> RunSnapshot:
    """Walk a QUEUED run to RUNNING at ``stage`` through real CAS transitions."""
    held = snapshot(session.get(PipelineRun, run_id))  # type: ignore[arg-type]
    held = cas_update_run(
        session, held, status=RunStatus.RUNNING, current_stage=held.current_stage or STAGE_ORDER[0]
    )
    while held.current_stage is not stage:
        nxt = next_stage(held.current_stage)
        assert nxt is not None, f"{stage!r} not reachable in STAGE_ORDER"
        held = cas_update_run(session, held, status=RunStatus.RUNNING, current_stage=nxt)
    session.commit()
    return held


def test_cancel_missing_run_raises_not_found(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session, pytest.raises(RunNotFoundError):
        cancel_run(session, uuid.uuid4())


def test_cancel_queued_run_cancels_keeping_no_stage(
    session_factory: sessionmaker[Session],
) -> None:
    # A fresh QUEUED run carries no current_stage; cancelling it before dispatch
    # keeps that None (it never started a stage).
    with session_factory() as session:
        run_id = submit_media_item(session, "incoming/q-cancel.wav").id
        session.commit()

    with session_factory() as session:
        result = cancel_run(session, run_id)
        session.commit()
        assert result.status is RunStatus.CANCELLED
        assert result.current_stage is None

    with session_factory() as session:
        stored = session.get(PipelineRun, run_id)
        assert stored is not None
        assert stored.status == RunStatus.CANCELLED.value
        assert stored.current_stage is None


def test_cancel_running_run_keeps_stage(
    session_factory: sessionmaker[Session],
) -> None:
    # Cancelling a RUNNING run preserves current_stage so the console shows where
    # it stopped.
    with session_factory() as session:
        run_id = submit_media_item(session, "incoming/run-cancel.wav").id
        session.commit()
        _drive_to_running(session, run_id, Stage.TRANSCRIBE)

    with session_factory() as session:
        result = cancel_run(session, run_id)
        session.commit()
        assert result.status is RunStatus.CANCELLED
        assert result.current_stage is Stage.TRANSCRIBE

    with session_factory() as session:
        stored = session.get(PipelineRun, run_id)
        assert stored is not None
        assert stored.status == RunStatus.CANCELLED.value
        assert stored.current_stage == Stage.TRANSCRIBE.value


def test_cancel_awaiting_adjudication_run(
    session_factory: sessionmaker[Session],
) -> None:
    # A human-paused run (no live worker) is cancellable; stage is kept.
    with session_factory() as session:
        run_id = submit_media_item(session, "incoming/await-cancel.wav").id
        session.commit()
        held = _drive_to_running(session, run_id, Stage.DIARIZE_EMBED)
        held = cas_update_run(
            session,
            held,
            status=RunStatus.AWAITING_ADJUDICATION,
            current_stage=Stage.DIARIZE_EMBED,
        )
        session.commit()

    with session_factory() as session:
        result = cancel_run(session, run_id)
        session.commit()
        assert result.status is RunStatus.CANCELLED
        assert result.current_stage is Stage.DIARIZE_EMBED


def test_cancel_failed_run_raises_not_cancellable(
    session_factory: sessionmaker[Session],
) -> None:
    # FAILED is requeueable, not cancellable — a distinct refusal.
    with session_factory() as session:
        run_id = submit_media_item(session, "incoming/failed-cancel.wav").id
        session.commit()
        _drive_to_failed(session, run_id, Stage.TRANSCRIBE)

    with session_factory() as session:
        with pytest.raises(RunNotCancellableError) as exc:
            cancel_run(session, run_id)
        assert exc.value.status is RunStatus.FAILED
        stored = session.get(PipelineRun, run_id)
        assert stored is not None
        assert stored.status == RunStatus.FAILED.value  # untouched


def test_cancel_completed_run_raises_not_cancellable(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = submit_media_item(session, "incoming/done-cancel.wav").id
        session.commit()
        held = _drive_to_running(session, run_id, STAGE_ORDER[-1])
        cas_update_run(session, held, status=RunStatus.COMPLETED, current_stage=None)
        session.commit()

    with session_factory() as session:
        with pytest.raises(RunNotCancellableError) as exc:
            cancel_run(session, run_id)
        assert exc.value.status is RunStatus.COMPLETED


def test_cancel_already_cancelled_is_idempotent(
    session_factory: sessionmaker[Session],
) -> None:
    # A second cancel (double-click / stale tab) is a no-op success, not an
    # error and not a second revision bump.
    with session_factory() as session:
        run_id = submit_media_item(session, "incoming/idem-cancel.wav").id
        session.commit()

    with session_factory() as session:
        first = cancel_run(session, run_id)
        session.commit()
        first_revision = first.revision

    with session_factory() as session:
        again = cancel_run(session, run_id)
        session.commit()
        assert again.status is RunStatus.CANCELLED
        assert again.revision == first_revision  # no extra write

    with session_factory() as session:
        stored = session.get(PipelineRun, run_id)
        assert stored is not None
        assert stored.revision == first_revision


def test_cancel_with_stale_expected_revision_rejects(
    session_factory: sessionmaker[Session],
) -> None:
    # A stale browser tab acting on a revision the run moved past is rejected
    # before any write; the matching revision is accepted.
    with session_factory() as session:
        run_id = submit_media_item(session, "incoming/stale-cancel.wav").id
        session.commit()
        held = _drive_to_running(session, run_id, Stage.PREPARE)
        live_revision = held.revision

    with session_factory() as session:
        with pytest.raises(StaleRevisionError):
            cancel_run(session, run_id, expected_revision=live_revision - 1)
        stored = session.get(PipelineRun, run_id)
        assert stored is not None
        assert stored.status == RunStatus.RUNNING.value  # untouched

    with session_factory() as session:
        result = cancel_run(session, run_id, expected_revision=live_revision)
        session.commit()
        assert result.status is RunStatus.CANCELLED
