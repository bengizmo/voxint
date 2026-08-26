"""The Jobs area: ``/jobs`` and ``/jobs/{run_id}`` (Console 2.0 P5, #160).

``/jobs`` answers "what is the pipeline doing right now": a per-stage activity
strip (queued/active), the per-service hardware-health strip, a compact recent-
runs table, and a normalized list of the auxiliary job families (asset,
translation, embedding, research). ``/jobs/{run_id}`` renders the same run-detail
sections as the legacy ``/runs/{id}`` page — it shares the builder and the body
partial, and its forms post to the existing ``/runs/{id}/...`` action endpoints
rather than minting ``/jobs/{id}/...`` aliases.

Dark-ship (#160): both pages are registered unconditionally and reachable
directly, carrying NO area gate — the flag is rollout control, not authorization
(codex-ratified). ``console_jobs_enabled`` gates only the sidebar's Jobs link
(``shell.jobs_enabled``); the ``/runs`` retirement is a later coordinated slice.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Final

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from voxint.api.jobs_query import jobs_badge_count, recent_aux_jobs, stage_activity
from voxint.api.resource_status import (
    build_resource_strip,
    collect_resource_status_or_empty,
)
from voxint.api.routers.deps import (
    OperatorDep,
    SessionDep,
    require_onboarded,
    templates,
)
from voxint.api.routers.legacy_runs import build_run_detail_context
from voxint.api.runs_query import list_runs
from voxint.api.stats_query import run_status_counts

router = APIRouter(dependencies=[Depends(require_onboarded)])

# Compact dashboard bounds (single-operator scale): the recent-runs table and
# the auxiliary-job list are "what happened lately" glances, not paginated
# browsers — the full runs browser stays at /runs.
_RECENT_RUNS: Final[int] = 10
_RECENT_AUX_JOBS: Final[int] = 20


@router.get("/jobs", name="jobs")
def jobs(request: Request, operator: OperatorDep, session: SessionDep) -> Response:
    page = list_runs(
        session,
        status=None,
        review=None,
        cursor=None,
        page_size=_RECENT_RUNS,
    )
    context = {
        "request": request,
        "active_nav": "jobs",
        "now": datetime.now(UTC),
        # Per-stage queued/active strip; its totals reconcile with the run
        # status counts below (the voxint stats source), so the page cannot
        # silently disagree with the CLI.
        "stage_activity": stage_activity(session),
        "status_counts": run_status_counts(session),
        "resource_strip": build_resource_strip(
            collect_resource_status_or_empty(request.app.state.settings)
        ),
        "runs": page.items,
        "aux_jobs": recent_aux_jobs(session, limit=_RECENT_AUX_JOBS),
        # The live-jobs count shared with the shell activity badge (#162): the
        # same query drives both, so the badge equals this page by construction.
        "jobs_badge_count": jobs_badge_count(session),
    }
    return templates.TemplateResponse(request, "jobs/jobs.html", context)


@router.get("/jobs/{run_id}", name="job_detail")
def job_detail(
    run_id: uuid.UUID, request: Request, operator: OperatorDep, session: SessionDep
) -> Response:
    # Absorbs the /runs/{id} run-detail page: same sections, same forms (which
    # post to the legacy /runs/{id}/... endpoints). The tutorial banner is off —
    # /jobs/{id} is dark-shipped and not in the tutorial's route map yet.
    context = build_run_detail_context(
        run_id, request, session, active_nav="jobs", tutorial=False
    )
    return templates.TemplateResponse(request, "jobs/job_detail.html", context)
