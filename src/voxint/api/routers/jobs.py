"""The Jobs area: ``/jobs`` and ``/jobs/{run_id}`` (Console 2.0 R2, #211).

``/jobs`` answers "what is the pipeline doing right now": a 5-cell pipeline
board (mapping the 6 internal stages to 5 operator-facing stages), the
degraded-state banner, a filterable recent-runs table, and auxiliary jobs.
``/jobs/{run_id}`` renders the same run-detail sections as the legacy
``/runs/{id}`` page.

Dark-ship (#160): both pages are registered unconditionally and reachable
directly, carrying NO area gate.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from voxint.api.health_probe import probe_services
from voxint.api.jobs_query import (
    StageActivity,
    jobs_badge_count,
    recent_aux_jobs,
    stage_activity,
)
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
from voxint.api.runs_query import (
    ReviewFilter,
    list_runs,
)
from voxint.api.stats_query import run_status_counts
from voxint.db.models import RunStatus

router = APIRouter(dependencies=[Depends(require_onboarded)])

_RECENT_RUNS: Final[int] = 10
_RECENT_AUX_JOBS: Final[int] = 20


@dataclass(frozen=True)
class DisplayStage:
    """One cell of the 5-stage pipeline board."""

    key: str
    label: str
    queued: int
    active: int


_STAGE_MAP: Final[dict[str, str]] = {
    "acquire": "add_media",
    "prepare": "add_media",
    "transcribe": "transcribe",
    "diarize_embed": "separate_voices",
    "enhance_match": "match_speakers",
    "finalize": "enrich",
}

_DISPLAY_STAGES: Final[tuple[tuple[str, str], ...]] = (
    ("add_media", "ADD MEDIA"),
    ("transcribe", "TRANSCRIBE"),
    ("separate_voices", "SEPARATE VOICES"),
    ("match_speakers", "MATCH SPEAKERS"),
    ("enrich", "ENRICH"),
)


def _collapse_stages(raw: list[StageActivity]) -> list[DisplayStage]:
    """Collapse 6 pipeline stages into 5 display stages."""
    buckets: dict[str, list[int]] = {key: [0, 0] for key, _ in _DISPLAY_STAGES}
    for sa in raw:
        display_key = _STAGE_MAP.get(sa.stage)
        if display_key and display_key in buckets:
            buckets[display_key][0] += sa.queued
            buckets[display_key][1] += sa.active
    return [
        DisplayStage(key=key, label=label, queued=b[0], active=b[1])
        for (key, label) in _DISPLAY_STAGES
        for b in [buckets[key]]
    ]


@dataclass(frozen=True)
class DegradedService:
    """One degraded pipeline component for the banner."""

    name: str
    consequence: str
    detail: str


_SERVICE_CONSEQUENCES: Final[dict[str, tuple[str, str]]] = {
    "transcription": (
        "Transcription is paused.",
        "New recordings will queue but not process until the transcriber is back.",
    ),
    "diarization": (
        "Voice separation is paused.",
        "Runs still transcribe, but speakers will not be identified.",
    ),
    "speaker embedding": (
        "Speaker matching is paused.",
        "Runs still finish, but new voices will not be matched to known speakers.",
    ),
}


def _detect_degraded(request: Request) -> list[DegradedService]:
    settings = request.app.state.settings
    try:
        probes = probe_services(settings)
    except Exception:
        return []
    degraded = []
    for probe in probes:
        if not probe.up and probe.name in _SERVICE_CONSEQUENCES:
            headline, detail = _SERVICE_CONSEQUENCES[probe.name]
            degraded.append(DegradedService(
                name=probe.name, consequence=headline, detail=detail,
            ))
    if settings.llm_enabled is False:
        degraded.append(DegradedService(
            name="enrichment",
            consequence="Enrichment is paused.",
            detail="Runs still finish — speaker profiles just won't fill in automatically.",
        ))
    return degraded


_FILTER_MAP: Final[dict[str, tuple[RunStatus | None, ReviewFilter | None]]] = {
    "all": (None, None),
    "active": (RunStatus.RUNNING, None),
    "needs_review": (None, ReviewFilter.NEEDED),
    "failed": (RunStatus.FAILED, None),
}

_FILTER_LABELS: Final[tuple[tuple[str, str], ...]] = (
    ("all", "All"),
    ("active", "Active"),
    ("needs_review", "Needs review"),
    ("failed", "Failed"),
)


def _pipeline_summary(status_counts: dict[str, int], badge: int) -> str:
    running = status_counts.get(RunStatus.RUNNING.value, 0)
    queued = status_counts.get(RunStatus.QUEUED.value, 0)
    parts: list[str] = []
    if running:
        parts.append(f"{running} running")
    if queued:
        parts.append(f"{queued} queued")
    if not parts:
        parts.append("idle")
    return " · ".join(parts)


@router.get("/jobs", name="jobs")
def jobs(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    filter: str | None = None,
) -> Response:
    active_filter = filter if filter in _FILTER_MAP else "all"
    status_f, review_f = _FILTER_MAP[active_filter]

    page = list_runs(
        session,
        status=status_f,
        review=review_f,
        cursor=None,
        page_size=_RECENT_RUNS,
    )

    status_counts = run_status_counts(session)
    badge = jobs_badge_count(session)

    context = {
        "request": request,
        "active_nav": "jobs",
        "now": datetime.now(UTC),
        "display_stages": _collapse_stages(stage_activity(session)),
        "status_counts": status_counts,
        "pipeline_summary": _pipeline_summary(status_counts, badge),
        "degraded": _detect_degraded(request),
        "resource_strip": build_resource_strip(
            collect_resource_status_or_empty(request.app.state.settings)
        ),
        "runs": page.items,
        "aux_jobs": recent_aux_jobs(session, limit=_RECENT_AUX_JOBS),
        "jobs_badge_count": badge,
        "active_filter": active_filter,
        "filter_labels": _FILTER_LABELS,
        "stage_activity": stage_activity(session),
    }
    return templates.TemplateResponse(request, "jobs/jobs.html", context)


@router.get("/jobs/{run_id}", name="job_detail")
def job_detail(
    run_id: uuid.UUID, request: Request, operator: OperatorDep, session: SessionDep
) -> Response:
    context = build_run_detail_context(
        run_id, request, session, active_nav="jobs", tutorial=False
    )
    return templates.TemplateResponse(request, "jobs/job_detail.html", context)
