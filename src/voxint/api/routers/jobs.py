"""Compatibility redirects from the legacy Jobs URLs to canonical Runs URLs."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Final

from fastapi import APIRouter, Depends, Request
from starlette.responses import RedirectResponse

from voxint.api.health_probe import probe_services
from voxint.api.resource_status import ResourceStripView
from voxint.api.routers.deps import require_onboarded
from voxint.db.models import RunStatus

router = APIRouter(dependencies=[Depends(require_onboarded)])

_FILTER_TO_RUNS: Final[dict[str, str]] = {
    "active": "/runs?view=active",
    "needs_review": "/runs?view=needs_attention",
    "failed": "/runs?view=failed",
}


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
            degraded.append(
                DegradedService(
                    name=probe.name,
                    consequence=headline,
                    detail=detail,
                )
            )
    if settings.llm_enabled is False:
        degraded.append(
            DegradedService(
                name="enrichment",
                consequence="Enrichment is paused.",
                detail="Runs still finish — speaker profiles just won't fill in automatically.",
            )
        )
    return degraded


def _pipeline_summary(
    status_counts: dict[str, int],
    resource_strip: ResourceStripView | None = None,
) -> str:
    running = status_counts.get(RunStatus.RUNNING.value, 0)
    queued = status_counts.get(RunStatus.QUEUED.value, 0)
    parts: list[str] = []
    if running:
        parts.append(f"{running} running")
    if queued:
        parts.append(f"{queued} queued")
    if (
        resource_strip
        and resource_strip.telemetry_present
        and any(g.state == "busy" for g in resource_strip.gpus)
    ):
        parts.append("GPU busy")
    if not parts:
        parts.append("idle")
    return " · ".join(parts)


@router.get("/jobs", name="jobs")
def jobs(filter: str | None = None) -> RedirectResponse:
    destination = _FILTER_TO_RUNS.get(filter or "", "/runs")
    return RedirectResponse(destination, status_code=303)


@router.get("/jobs/{run_id}", name="job_detail")
def job_detail(run_id: uuid.UUID) -> RedirectResponse:
    return RedirectResponse(f"/runs/{run_id}", status_code=303)
