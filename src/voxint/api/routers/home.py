"""The Home area: the console landing page at ``/`` (Console 2.0 R1, #210).

Home answers "what needs my attention and how do I add a recording": the
needs-attention cards, the quick actions, the windowed activity counts, and a
recent-activity feed. Read-only; the queries live in
:mod:`voxint.api.home_query` and :mod:`voxint.api.stats_query` (the latter
shared with ``voxint stats``, so the two surfaces agree for the same window).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Final

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from voxint.adjudication.resolver import adjudication_queue
from voxint.api.home_query import recent_activity
from voxint.api.routers.deps import (
    OperatorDep,
    SessionDep,
    require_onboarded,
    templates,
)
from voxint.api.stats_query import run_status_counts, windowed_counts
from voxint.db.models import RunStatus

router = APIRouter(dependencies=[Depends(require_onboarded)])

_WINDOWS: Final[tuple[tuple[str, str, timedelta | None], ...]] = (
    ("hour", "1h", timedelta(hours=1)),
    ("day", "24h", timedelta(hours=24)),
    ("week", "7d", timedelta(days=7)),
    ("all", "All time", None),
)
_DEFAULT_WINDOW: Final[str] = "day"

_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def _date_summary(now: datetime) -> str:
    weekday = _WEEKDAYS[now.weekday()]
    return f"{weekday}, {now.strftime('%b')} {now.day}"


@router.get("/", name="home")
def home(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    window: str | None = None,
) -> Response:
    now = datetime.now(UTC)
    known = {value for value, _, _ in _WINDOWS}
    window_invalid = window is not None and window not in known
    selected = window if window in known else _DEFAULT_WINDOW
    span = next(s for value, _, s in _WINDOWS if value == selected)
    since = None if span is None else now - span

    queue = adjudication_queue(session)
    status_counts = run_status_counts(session)

    context = {
        "request": request,
        "active_nav": "home",
        "now": now,
        "date_summary": _date_summary(now),
        "window": selected,
        "window_invalid": window_invalid,
        "windows": [(value, label) for value, label, _ in _WINDOWS],
        "counts": windowed_counts(session, since=since),
        "review_backlog": len(queue),
        "unresolved_voices": sum(entry.unresolved_labels for entry in queue),
        "failed_runs": status_counts.get(RunStatus.FAILED.value, 0),
        "activity": recent_activity(session),
    }
    return templates.TemplateResponse(request, "home/home.html", context)
