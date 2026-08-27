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

from __future__ import annotations

import math
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
    Project,
)


@dataclass(frozen=True)
class MediaLibraryRow:
    """One media file in the library, with its folder and latest-run status.

    ``source_title`` follows the run-listing display precedence's metadata leg
    (the acquisition-metadata title, ``None`` for uploads/pre-#36 media); the
    template falls back to a cleaned filename via ``friendly_media_label``.
    ``folder_path``/``project_name`` are the file's settings folder and that
    folder's project, or ``None`` for uploads/URLs/unmatched media; ``folder_path``
    is the folder whose settings apply, which may differ from where the bytes sit
    (ADR 0002 addendum). ``media_folder_id``/``project_id`` back the bulk-assign and
    grouping controls. The ``latest_run_*`` fields are ``None`` when the file has no
    run in the current view (no non-archived run by default; no archived run in the
    ``archived`` view, where such files are omitted entirely).
    """

    id: uuid.UUID
    source_path: str
    source_title: str | None
    media_folder_id: uuid.UUID | None
    folder_path: str | None
    project_id: uuid.UUID | None
    project_name: str | None
    duration_seconds: float | None
    size_bytes: int | None
    added_at: datetime
    latest_run_id: uuid.UUID | None
    latest_run_status: str | None
    latest_run_at: datetime | None


@dataclass(frozen=True)
class FolderGroup:
    """A registered folder and its media items, with aggregate stats.

    Used by the R3 media overview to render expandable folder rows in the
    grid table. Review counts let the template pick the right chip.
    """

    folder_id: uuid.UUID
    folder_path: str
    project_name: str | None
    item_count: int
    total_duration_seconds: float
    failed_count: int
    running_count: int
    completed_count: int
    latest_date: datetime | None
    items: list[MediaLibraryRow]


@dataclass(frozen=True)
class MediaSummary:
    """Aggregate counts for the command-bar summary line."""

    folder_count: int
    file_count: int
    total_hours: str


def group_by_folder(
    rows: list[MediaLibraryRow],
) -> tuple[list[FolderGroup], list[MediaLibraryRow]]:
    """Group library rows by folder, computing per-folder aggregates.

    Returns ``(folder_groups, ungrouped)`` where ``ungrouped`` is media items
    with no settings folder. Folder groups are sorted by path (case-insensitive).
    """
    by_folder: dict[uuid.UUID, list[MediaLibraryRow]] = {}
    ungrouped: list[MediaLibraryRow] = []
    for row in rows:
        if row.media_folder_id is not None:
            by_folder.setdefault(row.media_folder_id, []).append(row)
        else:
            ungrouped.append(row)

    groups: list[FolderGroup] = []
    for folder_id, items in by_folder.items():
        first = items[0]
        total_dur = sum(
            r.duration_seconds for r in items
            if r.duration_seconds is not None and math.isfinite(r.duration_seconds)
        )
        latest = max(
            (r.added_at for r in items), default=None
        )
        groups.append(FolderGroup(
            folder_id=folder_id,
            folder_path=first.folder_path or "unknown",
            project_name=first.project_name,
            item_count=len(items),
            total_duration_seconds=total_dur,
            failed_count=sum(
                1 for r in items if r.latest_run_status == "failed"
            ),
            running_count=sum(
                1 for r in items
                if r.latest_run_status in ("running", "queued")
            ),
            completed_count=sum(
                1 for r in items if r.latest_run_status == "completed"
            ),
            latest_date=latest,
            items=items,
        ))
    groups.sort(key=lambda g: g.folder_path.lower())
    return groups, ungrouped


def media_summary(
    folder_groups: list[FolderGroup],
    ungrouped: list[MediaLibraryRow],
) -> MediaSummary:
    """Compute the command-bar summary: folder count, file count, total hours."""
    folder_count = len(folder_groups)
    file_count = sum(g.item_count for g in folder_groups) + len(ungrouped)
    total_seconds = sum(g.total_duration_seconds for g in folder_groups) + sum(
        r.duration_seconds for r in ungrouped
        if r.duration_seconds is not None and math.isfinite(r.duration_seconds)
    )
    hours = total_seconds / 3600
    if hours >= 10:
        total_hours = f"{hours:.0f} hrs"
    elif hours >= 1:
        total_hours = f"{hours:.1f} hrs"
    else:
        minutes = total_seconds / 60
        total_hours = f"{minutes:.0f} min"
    return MediaSummary(
        folder_count=folder_count,
        file_count=file_count,
        total_hours=total_hours,
    )


@dataclass(frozen=True)
class FolderOption:
    """One registered folder for the upload/assign settings-folder picker.

    ``project_name`` labels the folder by the project it joins (``None`` for a
    folder in no project). The picker sets a settings SCOPE, not a project
    membership, so it renders whether or not the projects area is enabled.
    """

    id: uuid.UUID
    path: str
    project_name: str | None


def folder_options(session: Session) -> list[FolderOption]:
    """Registered folders for the picker, path-sorted, each labelled by project."""
    stmt = (
        sa_select(
            MediaFolder.id,
            MediaFolder.path,
            Project.name.label("project_name"),
        )
        .outerjoin(Project, Project.id == MediaFolder.project_id)
        .order_by(func.lower(MediaFolder.path))
    )
    return [
        FolderOption(id=row.id, path=row.path, project_name=row.project_name)
        for row in session.execute(stmt)
    ]


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
    session: Session,
    *,
    sort: str = DEFAULT_SORT,
    limit: int = MEDIA_LIBRARY_LIMIT,
    archived: bool = False,
) -> list[MediaLibraryRow]:
    """The media library rows, newest-first by default.

    ``sort`` is coerced to :data:`DEFAULT_SORT` when not in the allowlist. At most
    ``limit`` rows are returned; the caller surfaces truncation.

    ``archived`` picks the view. Default (``False``): every media item, its latest
    NON-archived run, matching the run listing and the Home feed — a file whose only
    runs were archived reads as "not processed yet". Archived view (``True``): ONLY
    items whose most-recent ARCHIVED run exists (inner join), showing that run — the
    discoverable set for bulk unarchive, since the default view hides archived runs
    and so gives unarchive no target (Console 2.0 P2b).
    """
    order_by = _SORTS.get(sort, _SORTS[DEFAULT_SORT])

    # Latest run per media item within the chosen view (non-archived by default,
    # archived-only in the archived view), picked with a window rather than a
    # correlated per-row subquery so the whole page is one round trip.
    run_view = (
        PipelineRun.archived_at.is_not(None)
        if archived
        else PipelineRun.archived_at.is_(None)
    )
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
        .where(run_view)
        .subquery()
    )
    latest = sa_select(ranked).where(ranked.c.rn == 1).subquery()

    stmt = sa_select(
        MediaItem.id,
        MediaItem.source_path,
        MediaItem.media_folder_id,
        MediaItem.duration_seconds,
        MediaItem.size_bytes,
        MediaItem.created_at,
        MediaSourceMetadata.title.label("source_title"),
        MediaFolder.path.label("folder_path"),
        Project.id.label("project_id"),
        Project.name.label("project_name"),
        latest.c.run_id,
        latest.c.status,
        latest.c.run_created_at,
    )
    # Outer: most media has no metadata snapshot (uploads, pre-#36 runs), sits under
    # no settings folder, or belongs to a folder with no project.
    stmt = stmt.outerjoin(
        MediaSourceMetadata, MediaSourceMetadata.media_item_id == MediaItem.id
    ).outerjoin(MediaFolder, MediaFolder.id == MediaItem.media_folder_id).outerjoin(
        Project, Project.id == MediaFolder.project_id
    )
    if archived:
        # Archived view: only items that actually have an archived run — an INNER
        # join drops never-run and only-live media, which have nothing to unarchive.
        stmt = stmt.join(latest, latest.c.media_item_id == MediaItem.id)
    else:
        stmt = stmt.outerjoin(latest, latest.c.media_item_id == MediaItem.id)
    stmt = stmt.order_by(*order_by).limit(limit)

    rows: list[MediaLibraryRow] = []
    for row in session.execute(stmt):
        rows.append(
            MediaLibraryRow(
                id=row.id,
                source_path=row.source_path,
                source_title=row.source_title,
                media_folder_id=row.media_folder_id,
                folder_path=row.folder_path,
                project_id=row.project_id,
                project_name=row.project_name,
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
