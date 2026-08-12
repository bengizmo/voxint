"""ACQUIRE download path (slice 6c) — a FAKE downloader, no network.

The real yt-dlp subprocess wrapper is unit-tested in
``tests/unit/test_media_ytdlp.py``; here the downloader is injected via
``StageContext`` so every publish/idempotency/failure invariant runs against a
real Postgres with the stage's REAL post-download logic (exactly-one-output,
authoritative size cap, sha256, atomic publish, row population).
"""

import hashlib
import shlex
import socket
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
from voxint.media.netcheck import Resolver
from voxint.media.redaction import MAX_STORED_ERROR_CHARS
from voxint.media.ytdlp import AcquisitionError, Downloader, run_download_command
from voxint.pipeline.engine import StageFailedError, StageFn, execute_run, submit
from voxint.pipeline.stages import acquire
from voxint.pipeline.stages.context import StageContext, StageDataError

_URL = "https://example.com/podcast"


def _resolver_returning(*addresses: str) -> Resolver:
    """A fake ``getaddrinfo`` returning fixed address strings — no real DNS, so the
    worker-side SSRF gate is exercised deterministically and offline in CI."""

    def resolve(host: str, *args: object, **kwargs: object) -> list[
        tuple[int, int, int, str, tuple[str, int]]
    ]:
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (addr, 0))
            for addr in addresses
        ]

    return resolve


# example.com's real global unicast address — the gate sees a public answer and
# lets every existing URL-run test proceed to its FAKE downloader without network.
_PUBLIC_RESOLVER = _resolver_returning("93.184.216.34")


def _ctx(
    media_root: Path,
    *,
    downloader: Downloader | None,
    max_bytes: int = 1024,
    resolver: Resolver = _PUBLIC_RESOLVER,
) -> StageContext:
    return StageContext(
        asr=FakeASR(),
        diarizer=FakeDiarizer(),
        embedder=FakeEmbedder(),
        llm=FakeLLM(),
        media_root=media_root,
        downloader=downloader,
        resolver=resolver,
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


def test_publish_never_overwrites_a_concurrently_published_source(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """A superseded ("zombie") attempt that finishes after a live attempt already
    published must NOT overwrite the published bytes, and the committed row must
    record the hash of the bytes actually on disk — never its own stale bytes.
    The racing downloader simulates a live attempt publishing source_path while
    this (zombie) attempt is still 'downloading'."""
    run_id, media_id, sp = _make_url_run(session_factory)
    dest = tmp_path / sp
    live_bytes = b"bytes-from-the-live-attempt"
    zombie_bytes = b"bytes-from-the-zombie-attempt"

    def racing_downloader(url: str, dest_dir: Path, max_bytes: int) -> None:
        (dest_dir / "source.m4a").write_bytes(zombie_bytes)  # this attempt's output
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(live_bytes)  # a live attempt published first, mid-download

    ctx = _ctx(tmp_path, downloader=racing_downloader)
    with session_factory() as session:
        acquire.run(ctx, session, run_id)
        session.commit()

    assert dest.read_bytes() == live_bytes  # the zombie did NOT clobber it
    media = _media_for(session_factory, media_id)
    assert media.sha256 == hashlib.sha256(live_bytes).hexdigest()
    assert media.size_bytes == len(live_bytes)


def test_symlink_output_is_rejected(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """A symlink is never a valid sole output — publishing it would leave
    source_path pointing outside the temp dir, dangling after cleanup."""
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"payload")
    run_id, _media_id, sp = _make_url_run(session_factory)

    def symlink_downloader(url: str, dest_dir: Path, max_bytes: int) -> None:
        (dest_dir / "source.m4a").symlink_to(outside)

    ctx = _ctx(tmp_path, downloader=symlink_downloader)
    with session_factory() as session, pytest.raises(
        AcquisitionError, match="exactly one"
    ):
        acquire.run(ctx, session, run_id)
    assert not (tmp_path / sp).exists()


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


# A unique token planted in the fake yt-dlp stderr; the redaction property is
# asserted structurally by its ABSENCE from the persisted error columns.
_LEAK_SENTINEL = "ACQUIRE-LEAK-SENTINEL-c0ffee"
_SIGNED_URL = f"https://cdn.example.com/media.mp3?token={_LEAK_SENTINEL}&sig=deadbeef"


def _stderr_leaking_downloader(url: str, dest_dir: Path, max_bytes: int) -> None:
    """A downloader that exercises the REAL subprocess boundary (so its born-clean
    redaction runs) with a harmless stub echoing a signed URL to stderr then
    failing nonzero — exactly the shape yt-dlp produces on a blocked download."""
    blob = f"ERROR: unable to download {_SIGNED_URL}: HTTP Error 403: Forbidden"
    run_download_command(
        ["sh", "-c", f"printf '%s' {shlex.quote(blob)} 1>&2; exit 1"],
        timeout_seconds=10,
    )


def test_acquire_failure_persists_redacted_error(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """A failed download surfaces yt-dlp's stderr (which echoes the signed source
    URL). The persisted StageRun.error AND PipelineRun.error must carry NO raw
    URL / token / query, while still naming the failure and the host."""
    run_id, media_id, sp = _make_url_run(session_factory)
    ctx = _ctx(tmp_path, downloader=_stderr_leaking_downloader)

    with pytest.raises(StageFailedError) as excinfo:
        execute_run(session_factory, run_id, _fns(ctx))
    assert excinfo.value.stage is Stage.ACQUIRE

    claims = _acquire_claims(session_factory, run_id)
    assert [c.status for c in claims] == [StageStatus.FAILED.value]
    stage_error = claims[0].error
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.FAILED.value
        run_error = run.error

    for label, err in (("StageRun.error", stage_error), ("PipelineRun.error", run_error)):
        assert err is not None, label
        assert _LEAK_SENTINEL not in err, label  # the signed token is gone
        assert "token=" not in err and "sig=" not in err, label
        assert "deadbeef" not in err, label
        assert "<redacted>" in err, label  # redaction demonstrably ran
        assert "exit 1" in err, label  # the failure class is still legible
        assert "cdn.example.com" in err, label  # host kept as a diagnostic
    # A failed download publishes nothing and records no hash.
    assert not (tmp_path / sp).exists()
    media = _media_for(session_factory, media_id)
    assert media.sha256 is None


def test_persisted_stage_error_is_length_capped(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """A pathologically long error is length-capped in BOTH ledger columns by the
    engine's general persistence cap, regardless of the stage."""
    run_id, _media_id, _sp = _make_url_run(session_factory)
    huge = "E" * (MAX_STORED_ERROR_CHARS + 5000)
    ctx = _ctx(tmp_path, downloader=_raiser(AcquisitionError(huge)))

    with pytest.raises(StageFailedError):
        execute_run(session_factory, run_id, _fns(ctx))

    claims = _acquire_claims(session_factory, run_id)
    stage_error = claims[0].error
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        run_error = run.error

    for err in (stage_error, run_error):
        assert err is not None
        assert len(err) <= MAX_STORED_ERROR_CHARS
        assert err.endswith("[truncated]")


# --- worker-side DNS re-resolution SSRF gate (slice 6g) -----------------------
# validate_ingest_url gated row creation at submit but did NOT resolve DNS; the
# worker re-resolves the host here and refuses any non-public answer BEFORE the
# download. TEST-NET literals (192.0.2/198.51.100/203.0.113) stand in for a host
# that rebinds to a private address — all non-global, so the gate rejects them.

_SIGNED_SSRF_URL = "https://cdn.example.com/media.mp3?token=SSRF-REBIND-SENTINEL"


def _failing_resolver(host: str, *args: object, **kwargs: object) -> list[object]:
    raise socket.gaierror("Name or service not known")


def test_host_resolving_non_public_is_rejected_before_download(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """A host that resolves to a non-public address is refused before the download
    even runs (the FAKE downloader would AssertionError if called), and the error
    names only the host — never the URL / its signed token."""
    run_id, media_id, sp = _make_url_run(session_factory, source_url=_SIGNED_SSRF_URL)
    ctx = _ctx(
        tmp_path,
        downloader=_forbidden(),  # must NOT be called: the gate rejects first
        resolver=_resolver_returning("192.0.2.9"),  # TEST-NET-1, non-global
    )
    with session_factory() as session, pytest.raises(AcquisitionError) as exc:
        acquire.run(ctx, session, run_id)

    message = str(exc.value)
    assert "non-public" in message
    assert "cdn.example.com" in message  # host kept as a diagnostic
    assert "SSRF-REBIND-SENTINEL" not in message  # the signed token is absent
    assert "token=" not in message and _SIGNED_SSRF_URL not in message
    # Nothing published, no hash recorded (the gate rejects before any dir is made).
    assert not (tmp_path / sp).exists()
    media = _media_for(session_factory, media_id)
    assert media.sha256 is None and media.size_bytes is None


def test_any_non_public_answer_rejects(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """Rejection is on the FIRST non-public address, not "all private": a host that
    answers with a public AND a private address (a rebinding / split-horizon trick)
    is refused."""
    run_id, _media_id, sp = _make_url_run(session_factory)
    ctx = _ctx(
        tmp_path,
        downloader=_forbidden(),
        resolver=_resolver_returning("93.184.216.34", "198.51.100.7"),
    )
    with session_factory() as session, pytest.raises(AcquisitionError, match="non-public"):
        acquire.run(ctx, session, run_id)
    assert not (tmp_path / sp).exists()


def test_unresolvable_host_is_rejected(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """Fail-closed: a host that does not resolve is terminal (never proof of a
    public address), so the worker refuses it rather than hand it to yt-dlp."""
    run_id, _media_id, sp = _make_url_run(session_factory)
    ctx = _ctx(tmp_path, downloader=_forbidden(), resolver=_failing_resolver)
    with session_factory() as session, pytest.raises(
        AcquisitionError, match="could not be resolved"
    ):
        acquire.run(ctx, session, run_id)
    assert not (tmp_path / sp).exists()


def test_source_url_without_host_is_rejected(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """Defensive: the worker re-parses source_url ITSELF and refuses a stored URL
    with no host rather than resolving an empty string (don't trust a stored
    parse). The FAKE downloader must not be called."""
    run_id, _media_id, sp = _make_url_run(
        session_factory, source_url="http:///no-host/path"
    )
    ctx = _ctx(tmp_path, downloader=_forbidden())
    with session_factory() as session, pytest.raises(AcquisitionError, match="no host"):
        acquire.run(ctx, session, run_id)
    assert not (tmp_path / sp).exists()


def test_non_public_host_parks_run_failed_at_acquire(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """End to end: a non-public-resolving URL run parks FAILED @ acquire (a
    deterministic AcquisitionError, not auto-retried) with a persisted error that
    carries the host but no raw URL / token."""
    run_id, media_id, sp = _make_url_run(session_factory, source_url=_SIGNED_SSRF_URL)
    ctx = _ctx(
        tmp_path, downloader=_forbidden(), resolver=_resolver_returning("203.0.113.4")
    )

    with pytest.raises(StageFailedError) as excinfo:
        execute_run(session_factory, run_id, _fns(ctx))
    assert excinfo.value.stage is Stage.ACQUIRE

    claims = _acquire_claims(session_factory, run_id)
    assert [c.status for c in claims] == [StageStatus.FAILED.value]
    stage_error = claims[0].error
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.FAILED.value
        assert run.current_stage == Stage.ACQUIRE.value
        run_error = run.error
    for err in (stage_error, run_error):
        assert err is not None
        assert "SSRF-REBIND-SENTINEL" not in err and "token=" not in err
        assert "cdn.example.com" in err  # host survives as a diagnostic
    assert not (tmp_path / sp).exists()
    media = _media_for(session_factory, media_id)
    assert media.sha256 is None
