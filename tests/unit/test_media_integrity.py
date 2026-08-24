"""Unit tests for the sha256 backfill (issue #150, Console 2.0 P0a).

The sweep is exercised against an in-memory SQLite database holding only the
``media_items`` table, so it runs in the plain unit lane with no Postgres. The
hashing and path-guard helpers are pure and tested directly.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from voxint.db.models import Base, MediaItem
from voxint.media.integrity import (
    BackfillResult,
    backfill_sha256,
    openable_source,
    sha256_file,
)


@pytest.fixture()
def session_factory() -> Iterator[sessionmaker[Session]]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine, tables=[MediaItem.__table__])
    yield sessionmaker(engine, expire_on_commit=False)
    engine.dispose()


@pytest.fixture()
def media_root(tmp_path: Path) -> Path:
    root = tmp_path / "media"
    root.mkdir()
    return root


def _write(root: Path, rel: str, data: bytes) -> str:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return rel


# --- sha256_file ---------------------------------------------------------


def test_sha256_file_matches_hashlib(media_root: Path) -> None:
    data = b"the quick brown fox" * 100
    rel = _write(media_root, "clip.wav", data)
    assert sha256_file(media_root / rel) == hashlib.sha256(data).hexdigest()


def test_sha256_file_chunk_boundary_is_stable(media_root: Path) -> None:
    # A tiny chunk size forces many read() iterations; the digest must not
    # depend on the chunk boundary.
    data = os.urandom(4096)
    rel = _write(media_root, "big.bin", data)
    whole = sha256_file(media_root / rel)
    chunked = sha256_file(media_root / rel, chunk_bytes=7)
    assert whole == chunked == hashlib.sha256(data).hexdigest()


def test_sha256_file_empty(media_root: Path) -> None:
    rel = _write(media_root, "empty.wav", b"")
    assert sha256_file(media_root / rel) == hashlib.sha256(b"").hexdigest()


# --- openable_source -----------------------------------------------------


def test_openable_source_regular_file(media_root: Path) -> None:
    rel = _write(media_root, "sub/a.wav", b"x")
    resolved = openable_source(media_root, rel)
    assert resolved == (media_root / rel).resolve()


def test_openable_source_missing_is_none(media_root: Path) -> None:
    assert openable_source(media_root, "nope.wav") is None


def test_openable_source_escape_is_none(media_root: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"secret")
    assert openable_source(media_root, "../outside.wav") is None


def test_openable_source_directory_is_none(media_root: Path) -> None:
    (media_root / "adir").mkdir()
    assert openable_source(media_root, "adir") is None


def test_openable_source_nul_byte_is_none(media_root: Path) -> None:
    # A NUL byte in the path raises ValueError inside pathlib; the guard must
    # fail closed rather than propagate it.
    assert openable_source(media_root, "bad\x00.wav") is None


def test_openable_source_symlink_escape_is_none(media_root: Path, tmp_path: Path) -> None:
    secret = tmp_path / "secret.wav"
    secret.write_bytes(b"s")
    link = media_root / "link.wav"
    link.symlink_to(secret)
    assert openable_source(media_root, "link.wav") is None


# --- backfill_sha256 -----------------------------------------------------


def test_backfill_hashes_null_rows(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    data = b"audio-bytes"
    rel = _write(media_root, "a.wav", data)
    with session_factory() as session:
        session.add(MediaItem(source_path=rel))
        session.commit()

    with session_factory() as session:
        result = backfill_sha256(session, media_root)

    assert result.hashed == 1
    assert result.skipped_missing == ()
    assert result.scanned == 1
    with session_factory() as session:
        row = session.execute(select(MediaItem)).scalars().one()
        assert row.sha256 == hashlib.sha256(data).hexdigest()


def test_backfill_is_idempotent(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    rel = _write(media_root, "a.wav", b"bytes")
    with session_factory() as session:
        session.add(MediaItem(source_path=rel))
        session.commit()

    with session_factory() as session:
        first = backfill_sha256(session, media_root)
    with session_factory() as session:
        second = backfill_sha256(session, media_root)

    assert first.hashed == 1
    assert second.hashed == 0
    assert second.scanned == 0


def test_backfill_leaves_existing_hash_untouched(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    rel = _write(media_root, "a.wav", b"real-bytes")
    with session_factory() as session:
        # A pre-existing hash that deliberately does NOT match the file bytes:
        # the sweep must not recompute it (NULL-only), proving it never reopens
        # already-hashed rows.
        session.add(MediaItem(source_path=rel, sha256="deadbeef"))
        session.commit()

    with session_factory() as session:
        result = backfill_sha256(session, media_root)

    assert result.hashed == 0
    with session_factory() as session:
        row = session.execute(select(MediaItem)).scalars().one()
        assert row.sha256 == "deadbeef"


def test_backfill_skips_missing_bytes(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    present = _write(media_root, "here.wav", b"x")
    with session_factory() as session:
        session.add(MediaItem(source_path=present))
        session.add(MediaItem(source_path="gone.wav"))
        session.commit()

    with session_factory() as session:
        result = backfill_sha256(session, media_root)

    assert result.hashed == 1
    assert result.skipped_missing == ("gone.wav",)
    with session_factory() as session:
        rows = {
            r.source_path: r.sha256
            for r in session.execute(select(MediaItem)).scalars()
        }
    assert rows["here.wav"] is not None
    assert rows["gone.wav"] is None


def test_backfill_missing_row_retried_when_file_returns(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        session.add(MediaItem(source_path="late.wav"))
        session.commit()

    with session_factory() as session:
        first = backfill_sha256(session, media_root)
    assert first.hashed == 0
    assert first.skipped_missing == ("late.wav",)

    _write(media_root, "late.wav", b"arrived")
    with session_factory() as session:
        second = backfill_sha256(session, media_root)
    assert second.hashed == 1


def test_backfill_invokes_callback_per_hashed_row(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    _write(media_root, "a.wav", b"a")
    _write(media_root, "b.wav", b"b")
    with session_factory() as session:
        session.add(MediaItem(source_path="a.wav"))
        session.add(MediaItem(source_path="b.wav"))
        session.add(MediaItem(source_path="missing.wav"))
        session.commit()

    seen: list[str] = []
    with session_factory() as session:
        backfill_sha256(session, media_root, on_hashed=lambda m: seen.append(m.source_path))

    assert sorted(seen) == ["a.wav", "b.wav"]


def test_backfill_result_scanned_property() -> None:
    result = BackfillResult(hashed=2, skipped_missing=("x", "y", "z"))
    assert result.scanned == 5
