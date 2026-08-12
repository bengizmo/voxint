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
    RunNotFailedError,
    RunNotFoundError,
    requeue_failed_run,
    submit_media_item,
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
