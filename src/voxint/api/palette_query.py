"""Entity search for the command palette (issue #162, slice 2).

Three entity kinds -- media, speakers, projects -- searched via ILIKE with
proper escaping. No trigram or FTS indexes: at single-operator scale (hundreds
of entities), substring matching with no index is sub-5ms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from voxint.api.media_query import _escape_like
from voxint.api.presentation import friendly_media_label
from voxint.db.models import MediaItem, MediaSourceMetadata, Project, Speaker

if TYPE_CHECKING:
    from voxint.config import Settings

MIN_QUERY_CHARS = 2
MAX_QUERY_CHARS = 200
PER_KIND_CAP = 5


@dataclass(frozen=True)
class EntityItem:
    kind: str  # "media", "speaker", "project"
    id: str
    label: str
    sublabel: str | None
    href: str
    archived: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "id": self.id,
            "label": self.label,
            "sublabel": self.sublabel,
            "href": self.href,
            "archived": self.archived,
        }


@dataclass(frozen=True)
class EntityResults:
    state: str  # "ok" | "short_query"
    query: str
    items: list[EntityItem]

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "query": self.query,
            "items": [i.as_dict() for i in self.items],
        }


def search_entities(
    session: Session,
    *,
    query: str,
    settings: Settings,
    app_state: Any,
) -> EntityResults:
    """Search media, speakers, and projects by ILIKE substring match."""
    q = query.strip()
    if len(q) < MIN_QUERY_CHARS:
        return EntityResults(state="short_query", query=q, items=[])

    # Truncate excessively long queries.
    q = q[:MAX_QUERY_CHARS]
    pattern = f"%{_escape_like(q)}%"
    items: list[EntityItem] = []

    # ---- Media items ----
    media_enabled = settings.console_media_enabled and getattr(
        app_state, "media_routed", False
    )
    if media_enabled:
        media_stmt = (
            select(
                MediaItem.id,
                MediaItem.source_path,
                MediaSourceMetadata.title,
            )
            .outerjoin(
                MediaSourceMetadata,
                MediaSourceMetadata.media_item_id == MediaItem.id,
            )
            .where(
                MediaItem.trashed_at.is_(None),
                MediaItem.purged_at.is_(None),
                or_(
                    MediaItem.source_path.ilike(pattern, escape="\\"),
                    MediaSourceMetadata.title.ilike(pattern, escape="\\"),
                ),
            )
            .order_by(func.lower(func.coalesce(MediaSourceMetadata.title, MediaItem.source_path)))
            .limit(PER_KIND_CAP)
        )
        for m_row in session.execute(media_stmt):
            label = friendly_media_label(m_row.title, m_row.source_path)
            items.append(
                EntityItem(
                    kind="media",
                    id=str(m_row.id),
                    label=label,
                    sublabel=None,
                    href=f"/media/{m_row.id}/editor",
                    archived=False,
                )
            )

    # ---- Speakers ----
    speaker_stmt = (
        select(Speaker.id, Speaker.display_name)
        .where(
            Speaker.merged_into_id.is_(None),
            Speaker.deleted_at.is_(None),
            Speaker.display_name.ilike(pattern, escape="\\"),
        )
        .order_by(func.lower(Speaker.display_name))
        .limit(PER_KIND_CAP)
    )
    speakers_enabled = getattr(settings, "console_speakers_enabled", True)
    for s_row in session.execute(speaker_stmt):
        href = f"/speakers/{s_row.id}" if speakers_enabled else "/speakers"
        items.append(
            EntityItem(
                kind="speaker",
                id=str(s_row.id),
                label=s_row.display_name,
                sublabel=None,
                href=href,
                archived=False,
            )
        )

    # ---- Projects ----
    projects_enabled = settings.console_projects_enabled and getattr(
        app_state, "projects_routed", False
    )
    if projects_enabled:
        project_stmt = (
            select(Project.id, Project.name, Project.archived_at)
            .where(Project.name.ilike(pattern, escape="\\"))
            .order_by(func.lower(Project.name))
            .limit(PER_KIND_CAP)
        )
        for p_row in session.execute(project_stmt):
            archived = p_row.archived_at is not None
            sublabel = "(archived)" if archived else None
            items.append(
                EntityItem(
                    kind="project",
                    id=str(p_row.id),
                    label=p_row.name,
                    sublabel=sublabel,
                    href=f"/projects/{p_row.id}",
                    archived=archived,
                )
            )

    return EntityResults(state="ok", query=q, items=items)
