"""Acquire stage: the universal first stage that materializes the source file.

For local or uploaded media (``source_url IS NULL``) the file already sits at
``source_path``, so this stage is a **no-op success** — it exists only so every
run starts at ``STAGE_ORDER[0]`` and the transition machine stays uniform.

For URL runs (``source_url`` set) it downloads the media with yt-dlp and
atomically publishes it to ``source_path``. That download path lands in slice
6c; this skeleton owns the no-op branch and refuses to silently pass a URL run
through un-acquired (which would leave PREPARE with no file to read).
"""

import uuid

from sqlalchemy.orm import Session

from voxint.db.models import MediaItem, PipelineRun
from voxint.pipeline.stages.context import StageContext, StageDataError


def run(ctx: StageContext, session: Session, run_id: uuid.UUID) -> None:
    pipeline_run = session.get(PipelineRun, run_id)
    if pipeline_run is None:
        raise StageDataError(f"no pipeline run {run_id}")
    media = session.get(MediaItem, pipeline_run.media_item_id)
    if media is None:
        raise StageDataError(f"run {run_id}: media item missing")

    if media.source_url is None:
        # Local/uploaded media: the bytes are already at source_path. PREPARE is
        # the sole containment/decodability gate, so there is nothing to do here.
        return

    # URL acquisition (yt-dlp download → atomic publish to source_path) is wired
    # in slice 6c. Until then no submission path sets source_url, so this branch
    # is unreachable in normal operation; refuse loudly rather than no-op past a
    # URL run and let PREPARE fail on a missing file.
    raise StageDataError(
        f"run {run_id}: URL acquisition is not yet implemented (slice 6c)"
    )
