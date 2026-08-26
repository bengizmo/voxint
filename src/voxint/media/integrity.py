"""Content-hash backfill for existing media rows (issue #150, Console 2.0 P0a).

``media_items.sha256`` is nullable. Rows ingested before the ACQUIRE stage
began recording a content hash, and any local or uploaded media whose hash was
never computed, carry NULL. Console 2.0 wants a content hash on every row as an
INTEGRITY aid: detecting a silent byte change under an unchanged path, and
content-deduplicating in maintainer tooling. It is never identity. Identity
stays ``media_items.id``, anchored to the immutable acquisition ``source_path``;
see ``docs/adr/0001-media-identity-vs-location.md``.

The backfill resolves bytes exactly the way the pipeline does
(``media_root / source_path`` with an escape guard, regular files only) so it
hashes the same bytes a run would. It is idempotent: only rows whose ``sha256``
is NULL are read, each hashed row is committed on its own, and a row whose bytes
are absent stays NULL and is reported rather than failing the sweep, so it is
retried on a later run if the file returns.

Pure and broker-free: it takes a session and a media root, so it is testable
without Celery or a live install.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import select

from voxint.db.models import MediaItem

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.orm import Session

# Match the pipeline hasher (``pipeline.stages.acquire._sha256``): the two must
# produce the same digest for the same bytes, since a backfilled hash and a
# freshly acquired hash are compared as one integrity value.
_HASH_CHUNK_BYTES = 1024 * 1024


def sha256_file(path: Path, *, chunk_bytes: int = _HASH_CHUNK_BYTES) -> str:
    """Streamed sha256 hex digest of a file, read in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def openable_source(media_root: Path, source_path: str) -> Path | None:
    """Resolve ``media_root / source_path`` to a readable regular file, or None.

    Mirrors the acquisition/prepare byte-opener guard: the resolved path must
    stay within ``media_root`` (no symlink or ``..`` escape) and be a regular
    file. A path that escapes, is missing, is a directory, or contains a NUL
    byte fails closed to None rather than raising, so the caller can record it
    as an honest skip.

    A residual symlink-swap race (a checked path replaced with an out-of-root
    symlink before the caller opens it) is out of the single-operator threat
    model here: the media root is the operator's own directory, not an untrusted
    drop point.
    """
    root = media_root.resolve()
    candidate = media_root / source_path
    try:
        resolved = candidate.resolve()
        resolved.relative_to(root)
    except (ValueError, OSError):
        return None
    try:
        if not resolved.is_file():
            return None
    except OSError:
        return None
    return resolved


def openable_current(media_root: Path, media: MediaItem) -> Path | None:
    """Resolve the live byte location for a media item, or None.

    Uses ``current_path`` (the mutable live location, ADR 0001 / ADR 0007)
    with a fallback to ``source_path`` for rows that predate the P2a backfill
    or have a NULL ``current_path``. The resolved path must stay within
    ``media_root`` and be a regular file, same as ``openable_source``.
    """
    path = media.current_path if media.current_path is not None else media.source_path
    return openable_source(media_root, path)


@dataclass(frozen=True)
class BackfillResult:
    """Outcome of one backfill sweep.

    ``hashed`` counts rows newly given a digest; ``skipped_missing`` is the
    ``source_path`` of each NULL row whose bytes could not be read under the
    media root, whether absent, unreadable, or vanished mid-hash (retried on a
    later run if the file returns).
    """

    hashed: int
    skipped_missing: tuple[str, ...]

    @property
    def scanned(self) -> int:
        return self.hashed + len(self.skipped_missing)


def backfill_sha256(
    session: Session,
    media_root: Path,
    *,
    on_hashed: Callable[[MediaItem], None] | None = None,
) -> BackfillResult:
    """Compute and store ``sha256`` for every media row missing it.

    Reads only NULL rows (idempotent), commits each hashed row on its own so an
    interrupted sweep keeps its progress, and records rows whose bytes are
    absent without touching them.
    """
    rows = (
        session.execute(
            select(MediaItem)
            .where(MediaItem.sha256.is_(None))
            .order_by(MediaItem.created_at)
        )
        .scalars()
        .all()
    )
    hashed = 0
    missing: list[str] = []
    for media in rows:
        path = openable_current(media_root, media)
        if path is None:
            missing.append(media.source_path)
            continue
        try:
            digest = sha256_file(path)
        except OSError:
            missing.append(media.source_path)
            continue
        media.sha256 = digest
        session.commit()
        hashed += 1
        if on_hashed is not None:
            on_hashed(media)
    return BackfillResult(hashed=hashed, skipped_missing=tuple(missing))
