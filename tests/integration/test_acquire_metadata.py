"""ACQUIRE metadata capture (issue #36) — a FAKE downloader, no network.

The sanitizer itself is unit-tested in ``tests/unit/test_source_metadata.py``;
here the stage's REAL capture path runs against a real Postgres: sidecar
publish ordering, write-once row insert, hash-addressed replay repair,
best-effort degradation, and the zombie-attempt adoption semantics.
"""

import hashlib
import json
import socket
import uuid
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.fakes import FakeASR, FakeDiarizer, FakeEmbedder, FakeLLM
from voxint.db.models import MediaItem, MediaSourceMetadata
from voxint.media.netcheck import Resolver
from voxint.media.source_metadata import extract as extract_snapshot
from voxint.media.source_metadata import sidecar_filename, to_sidecar_bytes
from voxint.media.ytdlp import INFO_JSON_FILENAME, Downloader
from voxint.pipeline.engine import submit
from voxint.pipeline.stages import acquire
from voxint.pipeline.stages.context import StageContext

_URL = "https://example.com/podcast"


def _resolver_returning(*addresses: str) -> Resolver:
    def resolve(host: str, *args: object, **kwargs: object) -> list[
        tuple[int, int, int, str, tuple[str, int]]
    ]:
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (addr, 0))
            for addr in addresses
        ]

    return resolve


_PUBLIC_RESOLVER = _resolver_returning("93.184.216.34")


def _ctx(
    media_root: Path,
    *,
    downloader: Downloader | None,
    metadata_secrets: tuple[str, ...] = (),
) -> StageContext:
    return StageContext(
        asr=FakeASR(),
        diarizer=FakeDiarizer(),
        embedder=FakeEmbedder(),
        llm=FakeLLM(),
        media_root=media_root,
        downloader=downloader,
        resolver=_PUBLIC_RESOLVER,
        ytdlp_max_bytes=1024,
        metadata_secrets=metadata_secrets,
    )


def _info_bytes(overrides: "dict[str, Any] | None" = None) -> bytes:
    info: dict[str, Any] = {
        "title": "Episode 42",
        "uploader": "Example Uploader",
        "channel": "Example Channel",
        "channel_url": "https://example.com/channel/UC123",
        "description": "About microphones.",
        "upload_date": "20260214",
        "duration": 3641.5,
        "tags": ["interviews", "acoustics"],
        "webpage_url": "https://example.com/watch?v=abc123",
        "extractor": "example",
        "_version": {"version": "2026.07.04"},
    }
    if overrides:
        info.update(overrides)
    return json.dumps(info).encode()


def _writer(*files: tuple[str, bytes]) -> Downloader:
    def download(url: str, dest_dir: Path, max_bytes: int) -> None:
        for name, data in files:
            (dest_dir / name).write_bytes(data)

    return download


def _forbidden() -> Downloader:
    def download(url: str, dest_dir: Path, max_bytes: int) -> None:
        raise AssertionError("downloader must not be called on the replay path")

    return download


def _make_url_run(
    session_factory: sessionmaker[Session],
) -> tuple[uuid.UUID, uuid.UUID, str]:
    sp = f"incoming/{uuid.uuid4().hex}/source"
    with session_factory() as session:
        media = MediaItem(source_path=sp, source_url=_URL)
        session.add(media)
        session.flush()
        media_id = media.id
        run_id = submit(session, media.id).id
        session.commit()
    return run_id, media_id, sp


def _metadata_row(
    session_factory: sessionmaker[Session], media_id: uuid.UUID
) -> "MediaSourceMetadata | None":
    with session_factory() as session:
        return session.execute(
            select(MediaSourceMetadata).where(
                MediaSourceMetadata.media_item_id == media_id
            )
        ).scalar_one_or_none()


def _run_acquire(
    session_factory: sessionmaker[Session], ctx: StageContext, run_id: uuid.UUID
) -> None:
    with session_factory() as session:
        acquire.run(ctx, session, run_id)
        session.commit()


def test_download_with_info_json_persists_snapshot_and_sidecar(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    run_id, media_id, sp = _make_url_run(session_factory)
    data = b"the-audio-bytes"
    ctx = _ctx(
        tmp_path,
        downloader=_writer(("source.m4a", data), (INFO_JSON_FILENAME, _info_bytes())),
    )

    _run_acquire(session_factory, ctx, run_id)

    row = _metadata_row(session_factory, media_id)
    assert row is not None
    assert row.source_kind == "ytdlp"
    assert row.title == "Episode 42"
    assert row.uploader == "Example Uploader"
    assert row.channel == "Example Channel"
    assert row.channel_url == "https://example.com/channel/UC123"
    assert row.description == "About microphones."
    assert row.upload_date == date(2026, 2, 14)
    assert row.duration_seconds == pytest.approx(3641.5)
    assert row.tags == ["interviews", "acoustics"]
    assert row.canonical_url == "https://example.com/watch?v=abc123"
    assert row.extractor == "example"
    assert row.extractor_version == "2026.07.04"
    assert row.raw is not None and row.raw["title"] == "Episode 42"
    assert row.raw_schema_version == 1
    assert row.acquired_at is not None and row.acquired_at.tzinfo is not None

    dest = tmp_path / sp
    sha = hashlib.sha256(data).hexdigest()
    # The sanitized hash-addressed sidecar is published beside the media...
    assert (dest.parent / sidecar_filename(sha)).is_file()
    # ...but the RAW info-JSON never leaves the (cleaned-up) attempt dir.
    assert not (dest.parent / INFO_JSON_FILENAME).exists()
    assert not list(dest.parent.glob(".acquire-*"))


def test_download_without_info_json_succeeds_with_no_metadata(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """Best-effort by decision: a downloader that produced no info-JSON (or a
    FAKE writing only media, i.e. every pre-#36 test) acquires cleanly."""
    run_id, media_id, sp = _make_url_run(session_factory)
    ctx = _ctx(tmp_path, downloader=_writer(("source.m4a", b"bytes")))

    _run_acquire(session_factory, ctx, run_id)

    assert (tmp_path / sp).is_file()
    assert _metadata_row(session_factory, media_id) is None


def test_malformed_info_json_degrades_to_no_metadata(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    run_id, media_id, sp = _make_url_run(session_factory)
    ctx = _ctx(
        tmp_path,
        downloader=_writer(("source.m4a", b"bytes"), (INFO_JSON_FILENAME, b"not json")),
    )

    _run_acquire(session_factory, ctx, run_id)  # must not raise

    assert (tmp_path / sp).is_file()
    assert _metadata_row(session_factory, media_id) is None


def test_hostile_info_json_secrets_never_reach_the_row(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """End-to-end pin of the allowlist contract: signed URLs / cookies planted
    in the info-JSON must be absent from every persisted column."""
    run_id, media_id, _sp = _make_url_run(session_factory)
    hostile = _info_bytes(
        {
            "url": "https://cdn.example.com/v.mp4?token=ROW-LEAK-SENTINEL",
            "formats": [{"url": "https://cdn.example.com/f.mp4?token=ROW-LEAK-SENTINEL"}],
            "http_headers": {"Cookie": "ROW-LEAK-SENTINEL"},
            "description": "proxy http://203.0.113.9:3128 was used",
        }
    )
    ctx = _ctx(
        tmp_path,
        downloader=_writer(("source.m4a", b"bytes"), (INFO_JSON_FILENAME, hostile)),
        metadata_secrets=("http://203.0.113.9:3128",),
    )

    _run_acquire(session_factory, ctx, run_id)

    row = _metadata_row(session_factory, media_id)
    assert row is not None
    flat = json.dumps(
        {
            c.name: str(getattr(row, c.name))
            for c in MediaSourceMetadata.__table__.columns
        }
    )
    assert "ROW-LEAK-SENTINEL" not in flat
    assert "203.0.113.9" not in flat
    assert row.description is not None and "<redacted>" in row.description


def test_replay_repairs_row_from_sidecar_without_redownloading(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """Crash between publish and commit: media + sidecar are on disk, row is
    absent. The replay re-inserts the row from the hash-addressed sidecar and
    never calls the downloader (sidecar-before-media ordering guarantees the
    sidecar is present whenever metadata was captured)."""
    run_id, media_id, sp = _make_url_run(session_factory)
    dest = tmp_path / sp
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = b"already-downloaded"
    dest.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    snapshot = extract_snapshot(_info_bytes())
    acquired = datetime(2026, 8, 1, 12, 0, 0, tzinfo=UTC)
    (dest.parent / sidecar_filename(sha)).write_bytes(
        to_sidecar_bytes(snapshot, media_sha256=sha, acquired_at=acquired)
    )

    _run_acquire(session_factory, _ctx(tmp_path, downloader=_forbidden()), run_id)

    row = _metadata_row(session_factory, media_id)
    assert row is not None
    assert row.title == "Episode 42"
    assert row.acquired_at == acquired  # replay reuses the ORIGINAL capture time


def test_replay_without_sidecar_leaves_metadata_absent(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """Legacy media (published before #36) replays cleanly with no metadata."""
    run_id, media_id, sp = _make_url_run(session_factory)
    dest = tmp_path / sp
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"pre-36-bytes")

    _run_acquire(session_factory, _ctx(tmp_path, downloader=_forbidden()), run_id)

    assert _metadata_row(session_factory, media_id) is None


def test_sidecar_for_different_bytes_is_never_loaded(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """A sidecar whose embedded hash does not match the authoritative file
    (an overlapping attempt that observed different upstream bytes) is inert —
    the hash-addressed name plus the load-time hash check both refuse it."""
    run_id, media_id, sp = _make_url_run(session_factory)
    dest = tmp_path / sp
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"authoritative-bytes")
    other_sha = hashlib.sha256(b"some-other-bytes").hexdigest()
    snapshot = extract_snapshot(_info_bytes({"title": "WRONG CONTEXT"}))
    (dest.parent / sidecar_filename(other_sha)).write_bytes(
        to_sidecar_bytes(
            snapshot,
            media_sha256=other_sha,
            acquired_at=datetime(2026, 8, 1, tzinfo=UTC),
        )
    )

    _run_acquire(session_factory, _ctx(tmp_path, downloader=_forbidden()), run_id)

    assert _metadata_row(session_factory, media_id) is None


def test_existing_row_is_never_updated(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """Write-once: a replay with a (hypothetically different) sidecar on disk
    must not touch an existing row — the context an adjudication was made
    against is immutable."""
    run_id, media_id, sp = _make_url_run(session_factory)
    data = b"the-bytes"
    ctx = _ctx(
        tmp_path,
        downloader=_writer(("source.m4a", data), (INFO_JSON_FILENAME, _info_bytes())),
    )
    _run_acquire(session_factory, ctx, run_id)
    row = _metadata_row(session_factory, media_id)
    assert row is not None and row.title == "Episode 42"

    # Overwrite the published sidecar with different content (simulating any
    # future tampering/corruption), then replay: the row must be unchanged.
    dest = tmp_path / sp
    sha = hashlib.sha256(data).hexdigest()
    snapshot = extract_snapshot(_info_bytes({"title": "REWRITTEN"}))
    sidecar = dest.parent / sidecar_filename(sha)
    sidecar.unlink()
    sidecar.write_bytes(
        to_sidecar_bytes(
            snapshot,
            media_sha256=sha,
            acquired_at=datetime(2026, 8, 2, tzinfo=UTC),
        )
    )
    _run_acquire(session_factory, _ctx(tmp_path, downloader=_forbidden()), run_id)

    row = _metadata_row(session_factory, media_id)
    assert row is not None
    assert row.title == "Episode 42"  # NOT "REWRITTEN"


def test_zombie_attempt_metadata_never_describes_winner_bytes(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """The zombie-overwrite race, metadata edition: a live attempt publishes
    different bytes mid-download of this (zombie) attempt. The zombie's own
    sidecar is hash-addressed to ITS bytes, so no row is inserted for the
    winner's bytes — metadata can never describe bytes other than
    source_path's."""
    run_id, media_id, sp = _make_url_run(session_factory)
    dest = tmp_path / sp
    live_bytes = b"bytes-from-the-live-attempt"
    zombie_bytes = b"bytes-from-the-zombie-attempt"

    def racing_downloader(url: str, dest_dir: Path, max_bytes: int) -> None:
        (dest_dir / "source.m4a").write_bytes(zombie_bytes)
        (dest_dir / INFO_JSON_FILENAME).write_bytes(
            _info_bytes({"title": "ZOMBIE CONTEXT"})
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(live_bytes)  # the live attempt wins mid-download

    _run_acquire(session_factory, _ctx(tmp_path, downloader=racing_downloader), run_id)

    with session_factory() as session:
        media = session.get(MediaItem, media_id)
        assert media is not None
        assert media.sha256 == hashlib.sha256(live_bytes).hexdigest()
    # The zombie's sidecar exists under its own hash but no row was minted.
    zombie_sha = hashlib.sha256(zombie_bytes).hexdigest()
    assert (dest.parent / sidecar_filename(zombie_sha)).is_file()
    assert _metadata_row(session_factory, media_id) is None
