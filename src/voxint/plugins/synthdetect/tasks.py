"""Celery task for synthdetect scoring."""

from __future__ import annotations

import uuid

from voxint.worker.app import app


@app.task(name="voxint.plugin.synthdetect.score_run", ignore_result=True)  # type: ignore[misc, untyped-decorator, unused-ignore]
def score_run(job_id_str: str) -> None:
    from voxint.config import get_settings
    from voxint.plugins.synthdetect.jobs import execute_job
    from voxint.worker.tasks import _runtime

    factory, _ = _runtime()
    execute_job(factory, uuid.UUID(job_id_str), settings=get_settings())
