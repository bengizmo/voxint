"""The media library area (Console 2.0 P2a, #153): the file listing at /media.

One read-only page listing every media file with its folder membership and the
status of its latest run (the query lives in :mod:`voxint.api.media_query`).
The route is always registered so the console route inventory is stable across
the dark-ship flip; access is gated by :func:`require_media_enabled`, which 404s
until ``console_media_enabled`` is on. The page is reachable only by URL for now
(the sidebar's Media link is repointed in a later phase).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from voxint.api.media_query import (
    DEFAULT_SORT,
    MEDIA_LIBRARY_LIMIT,
    SORT_LABELS,
    media_library,
    sort_is_known,
)
from voxint.api.routers.deps import (
    OperatorDep,
    SessionDep,
    require_media_enabled,
    require_onboarded,
    templates,
)

# require_onboarded first (an un-onboarded operator is sent to setup), then the
# area gate (404 when the flag is off) — the same order the module docstring and
# the projects area will follow.
router = APIRouter(
    dependencies=[Depends(require_onboarded), Depends(require_media_enabled)]
)

# The layout toggle: cards for scanning, a table for dense comparison. A ?view=
# outside this set degrades to the default rather than 422-ing (the Home ?window=
# convention), so a bookmarked value never breaks the page.
_VIEWS: Final[tuple[str, ...]] = ("cards", "table")
_DEFAULT_VIEW: Final[str] = "cards"


@router.get("/media", name="media_library")
def media_library_page(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    sort: str | None = None,
    view: str | None = None,
) -> Response:
    selected_sort = sort if sort_is_known(sort) else DEFAULT_SORT
    selected_view = view if view in _VIEWS else _DEFAULT_VIEW
    rows = media_library(session, sort=selected_sort)
    context = {
        "request": request,
        "active_nav": "media",
        "now": datetime.now(UTC),
        "rows": rows,
        "sort": selected_sort,
        "sorts": SORT_LABELS,
        "view": selected_view,
        "views": _VIEWS,
        # The listing is capped; say so honestly when it is full rather than
        # implying the library ends here.
        "truncated": len(rows) >= MEDIA_LIBRARY_LIMIT,
        "limit": MEDIA_LIBRARY_LIMIT,
    }
    return templates.TemplateResponse(request, "media/media.html", context)
