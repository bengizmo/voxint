"""ACQUIRE download path (slice 6c) — a FAKE downloader, no network.

The real yt-dlp subprocess wrapper is unit-tested in
``tests/unit/test_media_ytdlp.py``; here the downloader is injected via
``StageContext`` so every publish/idempotency/failure invariant runs against a
real Postgres with the stage's REAL post-download logic (exactly-one-output,
authoritative size cap, sha256, atomic publish, row population).
"""

import hashlib
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.fakes import FakeASR, FakeDiarizer, FakeEmbedder, FakeLLM
from voxint.db.models import (
    MediaItem,
    PipelineRun,
    RunStatus,
    Stage,
    StageRun,
    StageStatus,
)
from voxint.ingest.service import requeue_failed_run
from voxint.media.ytdlp import AcquisitionError, Downloader
from voxint.pipeline.engine import StageFailedError, StageFn, execute_run, submit
from voxint.pipeline.stages import acquire
from voxint.pipeline.stages.context import StageContext, StageDataError

_URL = "https://example.com/podcast"


def _ctx(
    media_root: Path, *, downloader: Downloader | None, max_bytes: int = 1024
) -> StageContext:
    return StageContext(
        asr=FakeASR(),
        diarizer=FakeDiarizer(),
        embedder=FakeEmbedder(),
        llm=FakeLLM(),
        media_root=media_root,
        downloader=downloader,
        ytdlp_max_bytes=max_bytes,
    )


def _writer(*files: tuple[str, bytes]) -> Downloader:
    """A downloader that writes the given (name, bytes) files into the temp dir."""

    def download(url: str, dest_dir: Path, max_bytes: int) -> None:
        for name, data in files:
            (dest_dir / name).write_bytes(data)

    return download


def _raiser(exc: Exception) -> Downloader:
    def download(url: str, dest_dir: Path, max_bytes: int) -> None:
        raise exc

    return download


def _forbidden() -> Downloader:
    def download(url: str, dest_dir: Path, max_bytes: int) -> None:
        raise AssertionError("downloader must not be called on the idempotent replay path")

    return download


def _make_url_run(
    session_factory: sessionmaker[Session],
    *,
    source_url: str | None = _URL,
    source_path: str | None = None,
) -> tuple[uuid.UUID, uuid.UUID, str]:
    sp = source_path or f"incoming/{uuid.uuid4().hex}/source"
    with session_factory() as session:
        media = MediaItem(source_path=sp, source_url=source_url)
        session.add(media)
        session.flush()
        media_id = media.id
        run_id = submit(session, media.id).id
        session.commit()
    return run_id, media_id, sp


def _fns(ctx: StageContext) -> dict[Stage, StageFn]:
    """ACQUIRE runs its real body; downstream stages are trivial no-op trackers."""
    fns: dict[Stage, StageFn] = {s: (lambda session, rid: None) for s in Stage}
    fns[Stage.ACQUIRE] = lambda session, rid: acquire.run(ctx, session, rid)
    return fns


def _acquire_claims(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> list[StageRun]:
    with session_factory() as session:
        return list(
            session.execute(
                select(StageRun)
                .where(
                    StageRun.pipeline_run_id == run_id,
                    StageRun.stage == Stage.ACQUIRE.value,
                )
                .order_by(StageRun.attempt)
            ).scalars()
        )


def _media_for(
    session_factory: sessionmaker[Session], media_id: uuid.UUID
) -> MediaItem:
    with session_factory() as session:
        media = session.get(MediaItem, media_id)
        assert media is not None
        return media


def test_download_publishes_source_and_records_hash(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    run_id, media_id, sp = _make_url_run(session_factory)
    data = b"the-audio-bytes"
    ctx = _ctx(tmp_path, downloader=_writer(("source.m4a", data)))

    with session_factory() as session:
        acquire.run(ctx, session, run_id)
        session.commit()

    dest = tmp_path / sp
    assert dest.is_file()
    assert dest.read_bytes() == data
    media = _media_for(session_factory, media_id)
    assert media.sha256 == hashlib.sha256(data).hexdigest()
    assert media.size_bytes == len(data)
    assert media.source_url == _URL  # provenance preserved
    # the attempt-unique temp dir is cleaned up
    assert not list(dest.parent.glob(".acquire-*"))


def test_oversize_download_is_rejected(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    run_id, media_id, sp = _make_url_run(session_factory)
    ctx = _ctx(tmp_path, downloader=_writer(("source.m4a", b"x" * 50)), max_bytes=8)

    with session_factory() as session, pytest.raises(AcquisitionError, match="exceeds"):
        acquire.run(ctx, session, run_id)

    assert not (tmp_path / sp).exists()
    media = _media_for(session_factory, media_id)
    assert media.sha256 is None and media.size_bytes is None


def test_missing_output_is_rejected(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    run_id, _media_id, sp = _make_url_run(session_factory)
    ctx = _ctx(tmp_path, downloader=_writer())  # writes nothing

    with session_factory() as session, pytest.raises(
        AcquisitionError, match="exactly one"
    ):
        acquire.run(ctx, session, run_id)
    assert not (tmp_path / sp).exists()


def test_multiple_output_is_rejected(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """A playlist / format-merge that leaves several files fails, never picks one."""
    run_id, _media_id, sp = _make_url_run(session_factory)
    ctx = _ctx(
        tmp_path,
        downloader=_writer(("a.m4a", b"one"), ("b.m4a", b"two")),
    )
    with session_factory() as session, pytest.raises(
        AcquisitionError, match="exactly one"
    ):
        acquire.run(ctx, session, run_id)
    assert not (tmp_path / sp).exists()


def test_empty_output_is_rejected(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    run_id, _media_id, _sp = _make_url_run(session_factory)
    ctx = _ctx(tmp_path, downloader=_writer(("source.m4a", b"")))
    with session_factory() as session, pytest.raises(AcquisitionError, match="empty"):
        acquire.run(ctx, session, run_id)


def test_replay_populates_row_without_redownloading(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """A crash between the atomic rename and the DB commit: the file is already at
    source_path but sha256/size are NULL. The recovered attempt hashes the
    finalized file and succeeds WITHOUT calling the downloader."""
    run_id, media_id, sp = _make_url_run(session_factory)
    dest = tmp_path / sp
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = b"already-downloaded"
    dest.write_bytes(data)

    ctx = _ctx(tmp_path, downloader=_forbidden())
    with session_factory() as session:
        acquire.run(ctx, session, run_id)  # must not raise
        session.commit()

    media = _media_for(session_factory, media_id)
    assert media.sha256 == hashlib.sha256(data).hexdigest()
    assert media.size_bytes == len(data)


def test_replay_is_a_noop_when_row_already_recorded(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """A fully-completed acquire re-run is a no-op: file present and row already
    populated ⇒ trust the stored hash, never re-download or re-hash."""
    run_id, media_id, sp = _make_url_run(session_factory)
    dest = tmp_path / sp
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"whatever-is-on-disk")
    sentinel = "sha-recorded-by-a-prior-attempt"
    with session_factory() as session:
        media = session.get(MediaItem, media_id)
        assert media is not None
        media.sha256 = sentinel
        media.size_bytes = 4321
        session.commit()

    ctx = _ctx(tmp_path, downloader=_forbidden())
    with session_factory() as session:
        acquire.run(ctx, session, run_id)
        session.commit()

    media = _media_for(session_factory, media_id)
    assert media.sha256 == sentinel  # untouched
    assert media.size_bytes == 4321


def test_url_run_without_downloader_refuses(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    run_id, _media_id, _sp = _make_url_run(session_factory)
    ctx = _ctx(tmp_path, downloader=None)
    with session_factory() as session, pytest.raises(
        StageDataError, match="no downloader configured"
    ):
        acquire.run(ctx, session, run_id)


def test_source_path_escaping_media_root_refuses(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    run_id, _media_id, _sp = _make_url_run(
        session_factory, source_path="../escapes/source"
    )
    ctx = _ctx(tmp_path, downloader=_writer(("source.m4a", b"x")))
    with session_factory() as session, pytest.raises(
        StageDataError, match="escapes media root"
    ):
        acquire.run(ctx, session, run_id)


def test_botblock_parks_run_failed_at_acquire(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """A bot-block is a deterministic AcquisitionError: the run parks FAILED @
    acquire for a manual Requeue and is NOT auto-retried (no source written)."""
    run_id, media_id, sp = _make_url_run(session_factory)
    ctx = _ctx(tmp_path, downloader=_raiser(AcquisitionError("HTTP 403: bot check")))

    with pytest.raises(StageFailedError) as excinfo:
        execute_run(session_factory, run_id, _fns(ctx))
    assert excinfo.value.stage is Stage.ACQUIRE

    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.FAILED.value
        assert run.current_stage == Stage.ACQUIRE.value
    claims = _acquire_claims(session_factory, run_id)
    assert [c.status for c in claims] == [StageStatus.FAILED.value]
    assert not (tmp_path / sp).exists()
    media = _media_for(session_factory, media_id)
    assert media.sha256 is None


def test_requeue_after_failure_creates_next_acquire_attempt(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    run_id, media_id, sp = _make_url_run(session_factory)

    # Attempt 1 bot-blocks.
    with pytest.raises(StageFailedError):
        execute_run(
            session_factory,
            run_id,
            _fns(_ctx(tmp_path, downloader=_raiser(AcquisitionError("403")))),
        )

    # Operator requeues the FAILED run at its failed stage.
    with session_factory() as session:
        requeue_failed_run(session, run_id)
        session.commit()

    # Attempt 2 succeeds and the run advances all the way through.
    data = b"second-time-lucky"
    final = execute_run(
        session_factory,
        run_id,
        _fns(_ctx(tmp_path, downloader=_writer(("source.m4a", data)))),
    )
    assert final.status is RunStatus.COMPLETED

    claims = _acquire_claims(session_factory, run_id)
    assert [c.status for c in claims] == [
        StageStatus.FAILED.value,
        StageStatus.COMPLETED.value,
    ]
    assert claims[1].attempt == 2
    assert (tmp_path / sp).read_bytes() == data
    media = _media_for(session_factory, media_id)
    assert media.sha256 == hashlib.sha256(data).hexdigest()
