"""Synthdetect plugin routes: report page, manual score, settings POST (#145).

No ``from __future__ import annotations`` here: the route handlers are defined
inside the ``build_synthdetect_router`` closure, and FastAPI's dependency
resolution requires eagerly-evaluated annotations to see ``Depends()`` calls.
Deferred (string) annotations break this — do not re-add the future import.
"""

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.api.csrf import CSRF_PLUGIN, CSRF_SETTINGS, verify_csrf_token
from voxint.api.routers.deps import AdminDep
from voxint.app_settings import get_app_settings, get_or_create
from voxint.config import Settings
from voxint.db.models import (
    DiarizationTurn,
    PipelineRun,
    RunStatus,
    SynthdetectJob,
    SynthdetectScore,
)
from voxint.plugins.deps import PluginRouteDeps

logger = logging.getLogger(__name__)

_FEATURE_FLAG_CHOICES = ("on", "off", "inherit")


def build_synthdetect_router(deps: PluginRouteDeps) -> APIRouter:
    router = APIRouter()

    @router.get("/synthdetect/report/{run_id}")
    def synthdetect_report(
        run_id: uuid.UUID,
        request: Request,
        session: Session = Depends(deps.get_session),  # noqa: B008
    ) -> Response:
        run = session.get(PipelineRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        job = session.execute(
            select(SynthdetectJob)
            .where(SynthdetectJob.pipeline_run_id == run_id)
            .order_by(SynthdetectJob.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()

        all_scores: list[dict[str, object]] = []
        high_risk_scores: list[dict[str, object]] = []
        skipped_count = 0

        if job is not None and job.status == "succeeded":
            scores_with_turns = session.execute(
                select(SynthdetectScore, DiarizationTurn)
                .outerjoin(
                    DiarizationTurn,
                    SynthdetectScore.diarization_turn_id == DiarizationTurn.id,
                )
                .where(SynthdetectScore.synthdetect_job_id == job.id)
                .order_by(DiarizationTurn.turn_index)
            ).all()

            for score, turn in scores_with_turns:
                row = {
                    "speaker_label": score.speaker_label,
                    "start_seconds": turn.start_seconds if turn else None,
                    "end_seconds": turn.end_seconds if turn else None,
                    "calibrated_score": score.calibrated_score,
                    "raw_logit": score.raw_logit,
                    "window_count": score.window_count,
                    "skip_reason": score.skip_reason,
                }
                all_scores.append(row)
                if score.skip_reason:
                    skipped_count += 1
                elif score.calibrated_score is not None and score.calibrated_score >= 0.1:
                    high_risk_scores.append(row)

            high_risk_scores.sort(
                key=lambda r: float(r["calibrated_score"] or 0.0), reverse=True  # type: ignore[arg-type]
            )
            high_risk_scores = high_risk_scores[:20]

        return deps.templates.TemplateResponse(
            request,
            "synthdetect/report.html",
            {
                "request": request,
                "run": run,
                "synthdetect_job": job,
                "all_scores": all_scores,
                "high_risk_scores": high_risk_scores,
                "skipped_count": skipped_count,
                "active_nav": "runs",
            },
        )

    @router.post("/synthdetect/score/{run_id}")
    def synthdetect_score(
        run_id: uuid.UUID,
        request: Request,
        admin: AdminDep,
        csrf_token: Annotated[str | None, Form()] = None,
        session: Session = Depends(deps.get_session),  # noqa: B008
    ) -> Response:
        if not verify_csrf_token(
            request.app.state.csrf_secret, CSRF_PLUGIN, csrf_token
        ):
            raise HTTPException(status_code=403, detail="Invalid CSRF token")

        run = session.get(PipelineRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if run.status != RunStatus.COMPLETED.value:
            raise HTTPException(
                status_code=409, detail="Run is not completed"
            )
        if run.archived_at is not None:
            raise HTTPException(
                status_code=409, detail="Run is archived"
            )

        settings: Settings = request.app.state.settings
        row = get_app_settings(session)

        from voxint.app_settings import resolve_effective_synthdetect_enabled

        if not resolve_effective_synthdetect_enabled(row, settings):
            raise HTTPException(
                status_code=409,
                detail="Synthetic speech detection is off",
            )

        from voxint.plugins.synthdetect.jobs import create_job

        job, _already = create_job(session, run_id, settings=settings)
        session.commit()

        if job is not None:
            from celery.exceptions import OperationalError

            from voxint.plugins.synthdetect import TASK_NAME
            from voxint.worker.app import app as celery_app

            try:
                celery_app.send_task(
                    TASK_NAME, args=(str(job.id),), ignore_result=True
                )
            except OperationalError:
                logger.warning(
                    "broker unavailable — synthdetect job %s stays QUEUED",
                    job.id,
                    exc_info=True,
                )

        return RedirectResponse(f"/runs/{run_id}", status_code=303)

    @router.post("/synthdetect/settings")
    def synthdetect_settings(
        request: Request,
        admin: AdminDep,
        synthdetect_enabled: Annotated[str, Form()] = "inherit",
        synthdetect_autogenerate: Annotated[str, Form()] = "inherit",
        csrf_token: Annotated[str | None, Form()] = None,
        session: Session = Depends(deps.get_session),  # noqa: B008
    ) -> Response:
        if not verify_csrf_token(
            request.app.state.csrf_secret, CSRF_SETTINGS, csrf_token
        ):
            raise HTTPException(status_code=403, detail="Invalid CSRF token")

        settings: Settings = request.app.state.settings
        submitted = {
            "synthdetect_enabled": synthdetect_enabled,
            "synthdetect_autogenerate": synthdetect_autogenerate,
        }

        for name in ("synthdetect_enabled", "synthdetect_autogenerate"):
            if submitted[name] not in _FEATURE_FLAG_CHOICES:
                return deps.render_settings_page(
                    request,
                    session,
                    synthdetect_errors=[
                        "Unrecognized setting value. Choose On, Off, or Use"
                        " installation setting."
                    ],
                    synthdetect_submitted=submitted,
                )

        candidates: dict[str, bool | None] = {}
        for name in ("synthdetect_enabled", "synthdetect_autogenerate"):
            choice = submitted[name]
            candidates[name] = None if choice == "inherit" else (choice == "on")

        def _effective(name: str) -> bool:
            c = candidates[name]
            return bool(getattr(settings, name)) if c is None else c

        if _effective("synthdetect_autogenerate") and not _effective(
            "synthdetect_enabled"
        ):
            return deps.render_settings_page(
                request,
                session,
                synthdetect_errors=[
                    "Turn synthetic speech detection on before enabling"
                    " automatic scoring. The automatic step only runs the"
                    " feature it rides on."
                ],
                synthdetect_submitted=submitted,
            )

        row = get_or_create(session, llm_enabled_default=settings.llm_enabled)
        for name, value in candidates.items():
            setattr(row, name, value)
        session.commit()
        return RedirectResponse("/settings#synthdetect", status_code=303)

    return router
