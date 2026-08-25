"""Read-only query behind the media library page (Console 2.0 P2a, issue #153).

Same shape as :mod:`voxint.api.home_query` and :mod:`voxint.api.stats_query`:
a frozen dataclass plus a function that takes a :class:`~sqlalchemy.orm.Session`
and issues one bounded ``SELECT``, no HTTP and no side effects.

The library lists every media item with its folder membership and the status of
its latest run. "Latest run per file" is resolved in a single window-function
subquery (``row_number() over (partition by media_item)``), not a per-row query,
so the page cost is one round trip regardless of how many files there are.
Archived runs are excluded, matching the run listing and the Home feed, so a
file whose only runs were archived reads as "not processed yet".
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, TypeGuard

from sqlalchemy import func
from sqlalchemy import select as sa_select
from sqlalchemy.orm import Session

from voxint.db.models import (
    MediaFolder,
    MediaItem,
    MediaSourceMetadata,
    PipelineRun,
)


@dataclass(frozen=True)
class MediaLibraryRow:
    """One media file in the library, with its folder and latest-run status.

    ``source_title`` follows the run-listing display precedence's metadata leg
    (the acquisition-metadata title, ``None`` for uploads/pre-#36 media); the
    template falls back to a cleaned filename via ``friendly_media_label``.
    ``folder_path`` is the registered folder the file belongs to, or ``None``
    for uploads/URLs/unmatched media. The ``latest_run_*`` fields are ``None``
    when the file has no non-archived run yet.
    """

    id: uuid.UUID
    source_path: str
    source_title: str | None
    folder_path: str | None
    duration_seconds: float | None
    size_bytes: int | None
    added_at: datetime
    latest_run_id: uuid.UUID | None
    latest_run_status: str | None
    latest_run_at: datetime | None


# Allowlisted sort keys -> the ORDER BY columns. A ?sort= outside this map
# degrades to the default rather than 422-ing (the Home ?window= convention),
# so a bookmarked or hand-typed value never breaks the page. Every ordering
# ends with (created_at desc, id desc) as a stable, deterministic tiebreak, so
# two loads over unchanged data render identically. NULLS LAST keeps files with
# no duration/size (still-acquiring, uploads) from crowding the top.
_SORTS: Final[dict[str, tuple[Any, ...]]] = {
    "added": (MediaItem.created_at.desc(), MediaItem.id.desc()),
    "name": (
        func.lower(func.coalesce(MediaSourceMetadata.title, MediaItem.source_path)),
        MediaItem.created_at.desc(),
        MediaItem.id.desc(),
    ),
    "duration": (
        MediaItem.duration_seconds.desc().nulls_last(),
        MediaItem.created_at.desc(),
        MediaItem.id.desc(),
    ),
    "size": (
        MediaItem.size_bytes.desc().nulls_last(),
        MediaItem.created_at.desc(),
        MediaItem.id.desc(),
    ),
}
DEFAULT_SORT: Final[str] = "added"
SORT_LABELS: Final[tuple[tuple[str, str], ...]] = (
    ("added", "Newest"),
    ("name", "Name"),
    ("duration", "Longest"),
    ("size", "Largest"),
)

# The listing is bounded (the read doctrine's "no unbounded SELECT"). The target
# audience is a single operator with a modest library, so a flat cap is honest
# and needs no config knob; the page says so when it truncates.
MEDIA_LIBRARY_LIMIT: Final[int] = 500


def sort_is_known(sort: str | None) -> TypeGuard[str]:
    return sort in _SORTS


def media_library(
    session: Session, *, sort: str = DEFAULT_SORT, limit: int = MEDIA_LIBRARY_LIMIT
) -> list[MediaLibraryRow]:
    """The media library rows, newest-first by default.

    ``sort`` is coerced to :data:`DEFAULT_SORT` when not in the allowlist. At most
    ``limit`` rows are returned; the caller surfaces truncation.
    """
    order_by = _SORTS.get(sort, _SORTS[DEFAULT_SORT])

    # Latest non-archived run per media item, picked with a window rather than a
    # correlated per-row subquery so the whole page is one round trip.
    ranked = (
        sa_select(
            PipelineRun.media_item_id.label("media_item_id"),
            PipelineRun.id.label("run_id"),
            PipelineRun.status.label("status"),
            PipelineRun.created_at.label("run_created_at"),
            func.row_number()
            .over(
                partition_by=PipelineRun.media_item_id,
                order_by=(PipelineRun.created_at.desc(), PipelineRun.id.desc()),
            )
            .label("rn"),
        )
        .where(PipelineRun.archived_at.is_(None))
        .subquery()
    )
    latest = sa_select(ranked).where(ranked.c.rn == 1).subquery()

    stmt = (
        sa_select(
            MediaItem.id,
            MediaItem.source_path,
            MediaItem.duration_seconds,
            MediaItem.size_bytes,
            MediaItem.created_at,
            MediaSourceMetadata.title.label("source_title"),
            MediaFolder.path.label("folder_path"),
            latest.c.run_id,
            latest.c.status,
            latest.c.run_created_at,
        )
        # Outer: most media has no metadata snapshot (uploads, pre-#36 runs),
        # sits outside a registered folder, or has never been run.
        .outerjoin(
            MediaSourceMetadata, MediaSourceMetadata.media_item_id == MediaItem.id
        )
        .outerjoin(MediaFolder, MediaFolder.id == MediaItem.media_folder_id)
        .outerjoin(latest, latest.c.media_item_id == MediaItem.id)
        .order_by(*order_by)
        .limit(limit)
    )

    rows: list[MediaLibraryRow] = []
    for row in session.execute(stmt):
        rows.append(
            MediaLibraryRow(
                id=row.id,
                source_path=row.source_path,
                source_title=row.source_title,
                folder_path=row.folder_path,
                duration_seconds=row.duration_seconds,
                size_bytes=row.size_bytes,
                # TIMESTAMPTZ comes back in the session timezone; normalize so
                # the template's "... UTC" title labels are always true.
                added_at=row.created_at.astimezone(UTC),
                latest_run_id=row.run_id,
                latest_run_status=row.status,
                latest_run_at=(
                    row.run_created_at.astimezone(UTC)
                    if row.run_created_at is not None
                    else None
                ),
            )
        )
    return rows
