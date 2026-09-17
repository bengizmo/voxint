"""Console activity polling and SSE endpoints (issues #162 and #499).

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

import asyncio
import json as json_mod
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session, sessionmaker
from starlette.concurrency import run_in_threadpool

from voxint.activity import ACTIVITY_POLL_LIMIT, events_since, high_water, retained_floor
from voxint.api.jobs_query import jobs_badge_count
from voxint.api.routers.deps import (
    OperatorDep,
    get_session_factory,
    require_onboarded,
)

router = APIRouter(dependencies=[Depends(require_onboarded)])


@router.get("/activity/events", name="activity_events")
def activity_events(
    request: Request,
    operator: OperatorDep,
    since: int | None = Query(default=None, ge=0),
) -> JSONResponse:
    if not request.app.state.settings.console_activity_enabled:
        raise HTTPException(status_code=404, detail="not found")

    return JSONResponse(_read_snapshot(get_session_factory(request), since))


def _read_snapshot(factory: sessionmaker[Session], cursor: int | None) -> dict[str, Any]:
    """Read and materialize a poll using a short-lived database session."""
    with factory() as session:
        badge = jobs_badge_count(session)
        hw = high_water(session)
        floor = retained_floor(session)
        rows = (
            events_since(session, after_id=cursor, limit=ACTIVITY_POLL_LIMIT)
            if cursor is not None
            else []
        )
        events = [
            {"id": row.id, "kind": row.kind, "title": row.title, "href": row.href} for row in rows
        ]
        next_cursor = hw if cursor is None else rows[-1].id if rows else cursor
        # Retention bounds let clients detect stale cursors and database resets.
        return {
            "events": events,
            "next_cursor": next_cursor,
            "has_more": len(rows) == ACTIVITY_POLL_LIMIT,
            "badge": badge,
            "high_water": hw,
            "floor": floor,
        }


@router.get("/activity/stream", name="activity_stream")
async def activity_stream(
    request: Request,
    operator: OperatorDep,
    since: int | None = Query(default=None, ge=0),
) -> StreamingResponse:
    if not request.app.state.settings.console_activity_enabled:
        raise HTTPException(status_code=404, detail="not found")

    cursor = since
    last_event_id = request.headers.get("Last-Event-ID")
    if last_event_id is not None:
        try:
            cursor = int(last_event_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="invalid Last-Event-ID") from exc
        if cursor < 0:
            raise HTTPException(status_code=422, detail="invalid Last-Event-ID")

    factory = get_session_factory(request)

    async def stream() -> AsyncIterator[str]:
        nonlocal cursor
        last_badge: int | None = None
        poll_cycles = 0
        try:
            yield "retry: 3000\n\n"
            while True:
                if await request.is_disconnected():
                    break
                snapshot = await run_in_threadpool(_read_snapshot, factory, cursor)
                if cursor is None:
                    # Baseline once so later polls can see newly arriving events.
                    cursor = snapshot["next_cursor"]
                if snapshot["events"]:
                    yield (
                        f"event: activity\nid: {snapshot['next_cursor']}\n"
                        f"data: {json_mod.dumps(snapshot)}\n\n"
                    )
                    cursor = snapshot["next_cursor"]
                    last_badge = snapshot["badge"]
                    if snapshot["has_more"]:
                        continue
                elif snapshot["badge"] != last_badge:
                    yield f"event: badge\ndata: {json_mod.dumps({'badge': snapshot['badge']})}\n\n"
                    last_badge = snapshot["badge"]

                await asyncio.sleep(2)
                poll_cycles += 1
                if poll_cycles == 10:
                    yield ": keepalive\n\n"
                    poll_cycles = 0
        finally:
            # Each snapshot already closed its session; no stream resources remain.
            pass

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
