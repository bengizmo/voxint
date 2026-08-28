"""Synthdetect plugin: synthetic speech detection for pipeline runs (#145)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from voxint.plugins.base import JobLaneSpec, PluginManifest, RunCompletedEvent, VoxintPlugin

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from voxint.config import Settings
    from voxint.db.models import AppSettings

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

    def enabled(self, row: AppSettings | None, settings: Settings) -> bool:
        from voxint.app_settings import resolve_effective_synthdetect_enabled

        return resolve_effective_synthdetect_enabled(row, settings)

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
