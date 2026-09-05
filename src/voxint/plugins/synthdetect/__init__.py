"""Synthdetect plugin: synthetic speech detection for pipeline runs (#145)."""

from __future__ import annotations

import argparse
import logging
import uuid
from typing import TYPE_CHECKING, Any, ClassVar

from voxint.plugins.base import (
    JobLaneSpec,
    PanelContribution,
    PluginManifest,
    RunCompletedEvent,
    SettingsSection,
    VoxintPlugin,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from fastapi import APIRouter
    from sqlalchemy.orm import Session

    from voxint.config import Settings
    from voxint.db.models import AppSettings
    from voxint.plugins.deps import PluginRouteDeps

logger = logging.getLogger(__name__)

TASK_NAME = "voxint.plugin.synthdetect.score_run"


class SynthdetectPlugin(VoxintPlugin):
    manifest: ClassVar[PluginManifest] = PluginManifest(
        id="synthdetect",
        name="Synthetic Speech Detection",
        description="Scores audio turns for synthetic speech risk",
        settings_prefixes=("synthdetect_",),
        task_names=(TASK_NAME,),
    )

    def add_cli_commands(self, subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
        from voxint.plugins.synthdetect.cli import register_commands

        register_commands(subparsers)

    def enabled(self, row: AppSettings | None, settings: Settings) -> bool:
        from voxint.app_settings import resolve_effective_synthdetect_enabled

        return resolve_effective_synthdetect_enabled(row, settings)

    def invariant_errors(
        self, row: AppSettings | None, settings: Settings
    ) -> list[str]:
        from voxint.app_settings import (
            resolve_effective_synthdetect_autogenerate,
            resolve_effective_synthdetect_enabled,
        )

        if resolve_effective_synthdetect_autogenerate(
            row, settings
        ) and not resolve_effective_synthdetect_enabled(row, settings):
            return [
                "synthdetect_autogenerate requires synthdetect_enabled=true"
            ]
        return []

    def settings_section(self) -> SettingsSection:
        return SettingsSection(
            section_id="synthdetect",
            title="Synthetic speech detection",
            template="synthdetect/settings.html",
            order=200,
        )

    def run_detail_panels(self) -> Sequence[PanelContribution]:
        return [
            PanelContribution(
                slot="run-detail-end",
                template="synthdetect/panel.html",
                order=200,
            ),
        ]

    def run_detail_context(
        self,
        run_id: uuid.UUID,
        session: Session,
        settings: Settings,
    ) -> dict[str, Any]:
        from sqlalchemy import case, select

        from voxint.app_settings import get_app_settings
        from voxint.db.models import SynthdetectJob, SynthdetectJobStatus

        row = get_app_settings(session)
        # Prefer the latest SUCCEEDED job so a fresh re-score does not hide
        # completed results.
        job = session.execute(
            select(SynthdetectJob)
            .where(
                SynthdetectJob.pipeline_run_id == run_id,
                SynthdetectJob.status == SynthdetectJobStatus.SUCCEEDED.value,
            )
            .order_by(SynthdetectJob.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        if job is None:
            # Fall back to the latest non-succeeded job (active first, then
            # terminal) so FAILED error info is still visible.
            job = session.execute(
                select(SynthdetectJob)
                .where(SynthdetectJob.pipeline_run_id == run_id)
                .order_by(
                    case(
                        (SynthdetectJob.status == SynthdetectJobStatus.RUNNING.value, 0),
                        (SynthdetectJob.status == SynthdetectJobStatus.QUEUED.value, 1),
                        else_=2,
                    ),
                    SynthdetectJob.created_at.desc(),
                )
                .limit(1)
            ).scalar_one_or_none()
        return {
            "synthdetect_job": job,
            "synthdetect_plugin_enabled": self.enabled(row, settings),
        }

    def build_router(self, deps: PluginRouteDeps) -> APIRouter:
        from voxint.plugins.synthdetect.routes import build_synthdetect_router

        return build_synthdetect_router(deps)

    def task_modules(self) -> Sequence[str]:
        return ["voxint.plugins.synthdetect.tasks"]

    def task_routes(self) -> Mapping[str, Mapping[str, str]]:
        return {TASK_NAME: {"queue": "post"}}

    def job_lanes(self) -> Sequence[JobLaneSpec]:
        from voxint.plugins.synthdetect.jobs import stale_queued_job_ids

        return [
            JobLaneSpec(
                stale_queued_job_ids=stale_queued_job_ids,
                redispatch_task_name=TASK_NAME,
                limit=50,
            ),
        ]

    def on_run_completed(self, event: RunCompletedEvent) -> None:
        from voxint.app_settings import (
            get_app_settings,
            resolve_effective_synthdetect_autogenerate,
            resolve_effective_synthdetect_enabled,
        )
        from voxint.plugins.synthdetect.jobs import create_job

        try:
            with event.session_factory() as session:
                row = get_app_settings(session)
                if not resolve_effective_synthdetect_enabled(row, event.settings):
                    return
                if not resolve_effective_synthdetect_autogenerate(row, event.settings):
                    return
                job, already = create_job(
                    session, event.run_id, settings=event.settings
                )
                session.commit()
                job_id = str(job.id) if job else None

            if job_id:
                from voxint.worker.app import app

                app.send_task(TASK_NAME, args=(job_id,), ignore_result=True)
                logger.info("synthdetect enqueued for run %s (job %s)", event.run_id, job_id)
            elif already:
                logger.debug("synthdetect already active for run %s", event.run_id)
        except Exception:
            logger.exception("synthdetect enqueue failed for run %s", event.run_id)
