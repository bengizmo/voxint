"""Console activity endpoints: JSON bootstrap + SSE stream (issues #162, #499).

The browser baselines once via ``GET /activity/events`` (JSON, no ``?since=``),
then opens ``GET /activity/stream?since=<cursor>`` (SSE) for live delivery.
Both read from the monotonic ``activity_events`` outbox. The SSE stream polls
the DB every 2 seconds, emits ``activity`` / ``badge`` / ``init`` / ``reset``
events, and sends keepalive comments for proxy liveness.

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


# The router-level require_onboarded SessionDep stays open (idle) for the
# stream's lifetime. Stream reads use short-lived sessions via _read_snapshot.
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

        yield "retry: 3000\n\n"
        while True:
            if await request.is_disconnected():
                break
            snapshot = await run_in_threadpool(_read_snapshot, factory, cursor)

            if cursor is None:
                cursor = snapshot["next_cursor"]
                init = {"next_cursor": cursor, "badge": snapshot["badge"]}
                yield (
                    f"event: init\nid: {cursor}\n"
                    f"data: {json_mod.dumps(init)}\n\n"
                )
                last_badge = snapshot["badge"]
                await asyncio.sleep(2)
                poll_cycles += 1
                continue

            hw = snapshot["high_water"]
            floor = snapshot["floor"]
            if cursor > hw or (floor > 0 and cursor < floor - 1):
                cursor = hw
                rst = {"next_cursor": cursor, "badge": snapshot["badge"]}
                yield (
                    f"event: reset\nid: {cursor}\n"
                    f"data: {json_mod.dumps(rst)}\n\n"
                )
                last_badge = snapshot["badge"]
                await asyncio.sleep(2)
                poll_cycles += 1
                continue

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

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
