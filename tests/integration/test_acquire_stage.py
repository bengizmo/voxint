"""ACQUIRE stage skeleton: the universal first stage.

Proves ACQUIRE no-ops for local media (``source_url IS NULL``) and completes +
advances to PREPARE through the engine, and that a URL run (``source_url`` set)
is refused until the downloader lands in slice 6c rather than silently passing an
un-acquired run to PREPARE.
"""

import uuid
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.fakes import FakeASR, FakeDiarizer, FakeEmbedder, FakeLLM
from voxint.db.models import (
    MediaItem,
    RunStatus,
    Stage,
    StageRun,
    StageStatus,
)
from voxint.pipeline.engine import StageFn, execute_run, submit
from voxint.pipeline.stages import acquire
from voxint.pipeline.stages.context import StageContext, StageDataError


def _ctx() -> StageContext:
    # ACQUIRE's no-op path never touches these clients; they satisfy the
    # StageContext contract so the body can be called in isolation.
    return StageContext(
        asr=FakeASR(),
        diarizer=FakeDiarizer(),
        embedder=FakeEmbedder(),
        llm=FakeLLM(),
        media_root=Path("/data/media"),
    )


def _make_run(
    session_factory: sessionmaker[Session], *, source_url: str | None
) -> uuid.UUID:
    with session_factory() as session:
        media = MediaItem(
            source_path=f"incoming/{uuid.uuid4()}/source", source_url=source_url
        )
        session.add(media)
        session.flush()
        run_id = submit(session, media.id).id
        session.commit()
    return run_id


def test_acquire_noops_for_local_media(
    session_factory: sessionmaker[Session],
) -> None:
    run_id = _make_run(session_factory, source_url=None)
    with session_factory() as session:
        acquire.run(_ctx(), session, run_id)  # returns cleanly, no effects
        session.commit()


def test_acquire_completes_and_advances_through_engine(
    session_factory: sessionmaker[Session],
) -> None:
    """The real ACQUIRE no-op runs first, completes, and hands off to PREPARE;
    downstream stages are trivial trackers so no GPU/ffmpeg is needed."""
    run_id = _make_run(session_factory, source_url=None)
    ctx = _ctx()
    executed: list[Stage] = []

    def tracker(stage: Stage) -> StageFn:
        def fn(session: Session, rid: uuid.UUID) -> None:
            executed.append(stage)

        return fn

    fns: dict[Stage, StageFn] = {s: tracker(s) for s in Stage}
    # ACQUIRE uses the real body (a no-op for this local run), not the tracker.
    fns[Stage.ACQUIRE] = lambda session, rid: acquire.run(ctx, session, rid)

    final = execute_run(session_factory, run_id, fns)
    assert final.status is RunStatus.COMPLETED
    # ACQUIRE ran its real (no-op) body, so it never appears in the tracker log;
    # PREPARE is the first tracked stage, proving ACQUIRE completed and advanced.
    assert Stage.ACQUIRE not in executed
    assert executed[0] is Stage.PREPARE

    with session_factory() as session:
        acquire_claims = (
            session.execute(
                select(StageRun).where(
                    StageRun.pipeline_run_id == run_id,
                    StageRun.stage == Stage.ACQUIRE.value,
                )
            )
            .scalars()
            .all()
        )
        assert len(acquire_claims) == 1
        assert acquire_claims[0].status == StageStatus.COMPLETED.value
        assert acquire_claims[0].attempt == 1


def test_acquire_refuses_url_run_until_downloader(
    session_factory: sessionmaker[Session],
) -> None:
    run_id = _make_run(session_factory, source_url="https://example.com/video")
    with (
        session_factory() as session,
        pytest.raises(StageDataError, match="not yet implemented"),
    ):
        acquire.run(_ctx(), session, run_id)


def test_acquire_missing_run_raises(
    session_factory: sessionmaker[Session],
) -> None:
    with (
        session_factory() as session,
        pytest.raises(StageDataError, match="no pipeline run"),
    ):
        acquire.run(_ctx(), session, uuid.uuid4())
