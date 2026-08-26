"""The console activity poll endpoint: ``GET /activity/events`` (issue #162).

The browser's activity indicator polls this every few seconds for new
``activity_events`` rows (completion toasts) and the live-jobs badge count. A
read-only JSON endpoint keyed on the monotonic ``id`` cursor:

* ``?since=<id>`` returns the rows with a larger id, oldest first, capped at one
  page; ``next_cursor`` is the last id returned and ``has_more`` is true when the
  page filled, so the client drains ascending pages without skipping a row.
* No ``since`` is the **bootstrap** call: a fresh browser (or one whose stored
  cursor fell outside the retained range) gets no events and the current
  high-water mark, so it baselines without toasting the retained backlog.

Dark-shipped behind ``console_activity_enabled``: operator auth runs first (the
``require_onboarded`` router dependency + ``OperatorDep``), then a disabled
install answers 404 like an unrouted page. GET only, so there is no CSRF surface.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from voxint.activity import ACTIVITY_POLL_LIMIT, events_since, high_water
from voxint.api.jobs_query import jobs_badge_count
from voxint.api.routers.deps import OperatorDep, SessionDep, require_onboarded

router = APIRouter(dependencies=[Depends(require_onboarded)])


@router.get("/activity/events", name="activity_events")
def activity_events(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    since: int | None = None,
) -> JSONResponse:
    if not request.app.state.settings.console_activity_enabled:
        raise HTTPException(status_code=404, detail="not found")

    badge = jobs_badge_count(session)
    hw = high_water(session)
    if since is None:
        # Bootstrap: baseline at the high-water mark, no historical toasts.
        return JSONResponse(
            {"events": [], "next_cursor": hw, "has_more": False, "badge": badge, "high_water": hw}
        )

    rows = events_since(session, after_id=since, limit=ACTIVITY_POLL_LIMIT)
    events = [
        {"id": row.id, "kind": row.kind, "title": row.title, "href": row.href} for row in rows
    ]
    next_cursor = rows[-1].id if rows else since
    # ``high_water`` lets the client detect a cursor that fell outside the
    # retained range (a DB reset that lowered max(id), or a prune) and rebaseline
    # to it without a stale flood; a normal cursor is always <= high_water.
    return JSONResponse(
        {
            "events": events,
            "next_cursor": next_cursor,
            "has_more": len(rows) == ACTIVITY_POLL_LIMIT,
            "badge": badge,
            "high_water": hw,
        }
    )
