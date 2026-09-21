"""Console activity endpoints: JSON bootstrap + SSE stream (issues #162, #499).

The browser baselines once via ``GET /activity/events`` (JSON, no ``?since=``),
then opens ``GET /activity/stream?since=<cursor>`` (SSE) for live delivery.
Both read from the monotonic ``activity_events`` outbox. The SSE stream polls
the DB every 2 seconds, emits ``activity`` / ``badge`` / ``init`` / ``reset``
events, and sends keepalive comments for proxy liveness.

Dark-shipped behind ``console_activity_enabled``: operator auth runs first (the
``require_onboarded`` router dependency + ``OperatorDep``), then a disabled
install answers 404 like an unrouted page. GET only, so there is no CSRF surface.

The SSE stream endpoint lives on ``stream_router``, a separate router without
``require_onboarded`` as a router-level dependency, to avoid holding a database
connection from ``SessionDep`` for the stream's lifetime (N5). Authentication
and onboarding are checked with a short-lived session at the start of the
handler, then released before entering the polling loop.
"""

from __future__ import annotations

import asyncio
import json as json_mod
import logging
from collections import defaultdict
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

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_onboarded)])

_MAX_STREAMS_PER_USER = 4
_MAX_STREAMS_GLOBAL = 20

_active_streams: dict[str, int] = defaultdict(int)
_active_streams_total = 0
_streams_lock = asyncio.Lock()

_SESSION_REVALIDATION_CYCLES = 15


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


def _authenticate_stream(request: Request, factory: sessionmaker[Session]) -> str:
    """Short-lived auth + onboarding check for the SSE stream.

    Returns the authenticated username. Raises HTTPException on failure.
    Releases the database session before returning.
    """
    from voxint.api.auth import (
        SESSION_COOKIE,
        load_session_user,
        verify_basic_credentials,
    )
    from voxint.app_settings import is_onboarded

    settings = request.app.state.settings

    with factory() as session:
        if settings.voxint_multi_user:
            token = request.cookies.get(SESSION_COOKIE)
            if not token:
                raise HTTPException(status_code=401, detail="authentication required")
            user = load_session_user(session, token)
            if user is None:
                raise HTTPException(status_code=401, detail="authentication required")
            username = user.username
        else:
            import base64

            auth = request.headers.get("authorization", "")
            if not auth.lower().startswith("basic "):
                raise HTTPException(
                    status_code=401,
                    detail="invalid credentials",
                    headers={"WWW-Authenticate": 'Basic realm="voxint-review"'},
                )
            try:
                decoded = base64.b64decode(auth[6:]).decode("utf-8")
                username, password = decoded.split(":", 1)
            except Exception:
                raise HTTPException(
                    status_code=401,
                    detail="invalid credentials",
                    headers={"WWW-Authenticate": 'Basic realm="voxint-review"'},
                ) from None
            if not verify_basic_credentials(settings, username, password):
                raise HTTPException(
                    status_code=401,
                    detail="invalid credentials",
                    headers={"WWW-Authenticate": 'Basic realm="voxint-review"'},
                )

        if not is_onboarded(session):
            raise HTTPException(status_code=404, detail="not found")

    return username


def _revalidate_session(request: Request, factory: sessionmaker[Session]) -> bool:
    """Check that the stream's session/credentials are still valid."""
    from voxint.api.auth import SESSION_COOKIE, load_session_user

    settings = request.app.state.settings

    with factory() as session:
        if settings.voxint_multi_user:
            token = request.cookies.get(SESSION_COOKIE)
            if not token:
                return False
            return load_session_user(session, token) is not None
        else:
            from voxint.api.auth import verify_basic_credentials

            auth = request.headers.get("authorization", "")
            if not auth.lower().startswith("basic "):
                return False
            import base64

            try:
                decoded = base64.b64decode(auth[6:]).decode("utf-8")
                username, password = decoded.split(":", 1)
            except Exception:
                return False
            return verify_basic_credentials(settings, username, password)


stream_router = APIRouter()


@stream_router.get("/activity/stream", name="activity_stream")
async def activity_stream(
    request: Request,
    since: int | None = Query(default=None, ge=0),
) -> StreamingResponse:
    if not request.app.state.settings.console_activity_enabled:
        raise HTTPException(status_code=404, detail="not found")

    factory = get_session_factory(request)
    username = await run_in_threadpool(_authenticate_stream, request, factory)

    global _active_streams_total
    async with _streams_lock:
        if _active_streams_total >= _MAX_STREAMS_GLOBAL:
            raise HTTPException(status_code=429, detail="too many active streams")
        if _active_streams[username] >= _MAX_STREAMS_PER_USER:
            raise HTTPException(status_code=429, detail="too many active streams")
        _active_streams[username] += 1
        _active_streams_total += 1

    cursor = since
    last_event_id = request.headers.get("Last-Event-ID")
    if last_event_id is not None:
        try:
            cursor = int(last_event_id)
        except ValueError as exc:
            async with _streams_lock:
                _active_streams[username] -= 1
                _active_streams_total -= 1
            raise HTTPException(status_code=422, detail="invalid Last-Event-ID") from exc
        if cursor < 0:
            async with _streams_lock:
                _active_streams[username] -= 1
                _active_streams_total -= 1
            raise HTTPException(status_code=422, detail="invalid Last-Event-ID")

    async def stream() -> AsyncIterator[str]:
        global _active_streams_total
        nonlocal cursor
        last_badge: int | None = None
        poll_cycles = 0
        revalidation_cycles = 0

        try:
            yield "retry: 3000\n\n"
            while True:
                if await request.is_disconnected():
                    break

                revalidation_cycles += 1
                if revalidation_cycles >= _SESSION_REVALIDATION_CYCLES:
                    revalidation_cycles = 0
                    valid = await run_in_threadpool(
                        _revalidate_session, request, factory
                    )
                    if not valid:
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
        finally:
            async with _streams_lock:
                _active_streams[username] -= 1
                if _active_streams[username] <= 0:
                    del _active_streams[username]
                _active_streams_total -= 1

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
