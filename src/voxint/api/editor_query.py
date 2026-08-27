"""Read-only queries for the media editor detail page (Console 2.0 P3a, issue #156).

Run selection: the editor opens the latest completed run by default and accepts
an explicit ``?run=`` override validated against the media item. The chooser
lists every run for the file so the operator can switch between them.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.db.models import MediaItem, PipelineRun, RunStatus


@dataclass(frozen=True)
class MediaIdentity:
    id: uuid.UUID
    source_path: str
    current_path: str | None
    duration_seconds: float | None
    size_bytes: int | None


@dataclass(frozen=True)
class SelectedRun:
    id: uuid.UUID
    status: str
    created_at: datetime
    updated_at: datetime
    revision: int
    archived_at: datetime | None


@dataclass(frozen=True)
class ChooserEntry:
    id: uuid.UUID
    status: str
    created_at: datetime
    archived: bool


@dataclass(frozen=True)
class MediaDetail:
    media: MediaIdentity
    selected_run: SelectedRun | None
    chooser: list[ChooserEntry]


def media_detail(
    session: Session,
    media_id: uuid.UUID,
    *,
    run_override: uuid.UUID | None = None,
) -> MediaDetail | None:
    """Load the media item and resolve the selected run.

    Returns ``None`` when the media item does not exist. When ``run_override``
    is given and the run does not exist or belongs to a different media item,
    also returns ``None`` (generic 404 -- never reveal whether the run exists
    elsewhere).
    """
    media = session.execute(
        select(MediaItem).where(MediaItem.id == media_id)
    ).scalar_one_or_none()
    if media is None:
        return None

    identity = MediaIdentity(
        id=media.id,
        source_path=media.source_path,
        current_path=media.current_path,
        duration_seconds=media.duration_seconds,
        size_bytes=media.size_bytes,
    )

    runs = (
        session.execute(
            select(PipelineRun)
            .where(PipelineRun.media_item_id == media_id)
            .order_by(PipelineRun.created_at.desc(), PipelineRun.id.desc())
        )
        .scalars()
        .all()
    )

    chooser = [
        ChooserEntry(
            id=r.id,
            status=r.status,
            created_at=r.created_at,
            archived=r.archived_at is not None,
        )
        for r in runs
    ]

    selected: SelectedRun | None = None

    if run_override is not None:
        for r in runs:
            if r.id == run_override:
                selected = _to_selected(r)
                break
        else:
            return None
    else:
        for r in runs:
            if r.status == RunStatus.COMPLETED.value:
                selected = _to_selected(r)
                break

    return MediaDetail(media=identity, selected_run=selected, chooser=chooser)


def _to_selected(run: PipelineRun) -> SelectedRun:
    return SelectedRun(
        id=run.id,
        status=run.status,
        created_at=run.created_at,
        updated_at=run.updated_at,
        revision=run.revision,
        archived_at=run.archived_at,
    )
