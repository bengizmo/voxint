"""Source recording dates at the device's offset (#741).

Only Apple's explicit-offset creation tag is read. The UTC-only creation tag
is never read because converting to UTC can change the recording's day.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

from sqlalchemy import CursorResult, select, update
from sqlalchemy.orm import Session

from voxint.db.models import MediaItem
from voxint.media.integrity import openable_current
from voxint.media.normalize import _INPUT_FORMATS, NormalizationError, _run

CREATION_DATE_TAG = "com.apple.quicktime.creationdate"
_TAG_PROBE_TIMEOUT_SECONDS = 10.0
_CREATION_DATE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?[+-][0-9]{2}:?[0-9]{2}"
)


def parse_creation_date(value: object) -> date | None:
    """Read the wall-clock date as written, requiring a numeric UTC offset."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if _CREATION_DATE.fullmatch(value) is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.date() if parsed.tzinfo is not None else None


def probe_recorded_on(
    path: Path,
    *,
    ffprobe_bin: str = "ffprobe",
    timeout_seconds: float = _TAG_PROBE_TIMEOUT_SECONDS,
) -> date | None:
    """Read source metadata without decoding; unavailable dates stay unknown."""
    try:
        result = _run(
            [ffprobe_bin, "-protocol_whitelist", "file", "-format_whitelist", _INPUT_FORMATS,
             "-v", "error", "-show_entries", f"format_tags={CREATION_DATE_TAG}",
             "-of", "json", str(path)],
            timeout_seconds=timeout_seconds,
        )
    except NormalizationError:
        return None
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
    except (ValueError, RecursionError):
        return None
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("format")
    if not isinstance(metadata, dict):
        return None
    tags = metadata.get("tags")
    if not isinstance(tags, dict):
        return None
    return parse_creation_date(tags.get(CREATION_DATE_TAG))


@dataclass(frozen=True)
class RecordedDateBackfillResult:
    """Dates found and rows left unknown during one sweep."""

    dated: tuple[tuple[uuid.UUID, date], ...]
    no_tag: int
    skipped_missing: tuple[str, ...]


def backfill_recorded_dates(
    session: Session,
    media_root: Path,
    *,
    ffprobe_bin: str,
    dry_run: bool = False,
    on_found: Callable[[uuid.UUID, date], None] | None = None,
) -> RecordedDateBackfillResult:
    """Fill unknown source dates, committing each row to preserve sweep progress."""
    rows = session.scalars(
        select(MediaItem)
        .where(MediaItem.recorded_on.is_(None), MediaItem.purged_at.is_(None))
        .order_by(MediaItem.created_at)
    ).all()
    dated: list[tuple[uuid.UUID, date]] = []
    missing: list[str] = []
    no_tag = 0
    for media in rows:
        path = openable_current(media_root, media)
        if path is None:
            missing.append(media.current_path or media.source_path)
            continue
        found = probe_recorded_on(path, ffprobe_bin=ffprobe_bin)
        if found is None:
            no_tag += 1
            continue
        media_id = media.id
        if not dry_run:
            written = cast(
                CursorResult[Any],
                session.execute(
                    update(MediaItem)
                    .where(MediaItem.id == media_id, MediaItem.recorded_on.is_(None))
                    .values(recorded_on=found)
                ),
            )
            session.commit()
            # A concurrent PREPARE stored a date first; first write wins.
            if written.rowcount == 0:
                continue
        dated.append((media_id, found))
        if on_found is not None:
            on_found(media_id, found)
    return RecordedDateBackfillResult(tuple(dated), no_tag, tuple(missing))
