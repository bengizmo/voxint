"""The Home area: the console landing page at ``/`` (Console 2.0 P1, #152).

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

# The stat switcher's windows: query value, label, span (None = all time).
# "hour" and "day" and "week" resolve to the same cutoffs the CLI's --since
# accepts as 1h / 24h / 7d, so "stats match voxint stats" is checkable by hand.
_WINDOWS: Final[tuple[tuple[str, str, timedelta | None], ...]] = (
    ("hour", "Hour", timedelta(hours=1)),
    ("day", "Day", timedelta(hours=24)),
    ("week", "Week", timedelta(days=7)),
    ("all", "All time", None),
)
_DEFAULT_WINDOW: Final[str] = "day"


@router.get("/", name="home")
def home(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    window: str | None = None,
) -> Response:
    now = datetime.now(UTC)
    # A bad/bookmarked ?window= degrades to the default rather than 422-ing,
    # and the page says so - it must never silently show a different window
    # than was asked for (the old dashboard's ?since= convention).
    known = {value for value, _, _ in _WINDOWS}
    window_invalid = window is not None and window not in known
    selected = window if window in known else _DEFAULT_WINDOW
    span = next(s for value, _, s in _WINDOWS if value == selected)
    since = None if span is None else now - span

    # One adjudication_queue call feeds BOTH attention counts (backlog runs and
    # unresolved voices), and the failed count comes from the same status query
    # /metrics uses - the cards cannot drift from the pages they link to.
    queue = adjudication_queue(session)
    status_counts = run_status_counts(session)

    context = {
        "request": request,
        "active_nav": "home",
        "now": now,
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
