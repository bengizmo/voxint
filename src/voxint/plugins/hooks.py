"""Core-side dispatch of the plugin hooks that fire from the worker (issue #137).

Two dispatchers the worker seams (#138) call:

* :func:`dispatch_run_completed` runs each active plugin's ``on_run_completed``
  after a run's finalize commit, isolating failures per plugin so a completed run
  stays COMPLETED whatever a handler does. Handlers are enqueue-only and
  idempotent, and each opens its own session from the event, so an exception in
  one plugin cannot corrupt another's work — the isolation here is exception
  containment plus a plugin-scoped logger.
* :func:`redispatch_stale_lane_jobs` is the generic form of #130's embedding
  stale-QUEUED recovery: for every plugin :class:`~voxint.plugins.base.JobLaneSpec`
  it re-publishes each stranded job id to the lane's task by name, bounded per
  sweep, with no row mutation (the guarded claim CAS collapses duplicates).

Both take their Celery entry point as a ``send_task`` callable rather than
importing the worker app, so ``voxint.plugins`` never imports the worker (import
direction is law).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING

from voxint.plugins.base import JobLaneSpec, RunCompletedEvent, VoxintPlugin

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)


def dispatch_run_completed(
    plugins: Sequence[VoxintPlugin], event: RunCompletedEvent
) -> None:
    """Fire ``on_run_completed`` for each plugin, containing per-plugin failures.

    A handler raising is logged under ``voxint.plugin.<id>`` and swallowed: the
    run is already COMPLETED and the next plugin still runs. Order is registry
    order (by ``manifest.id``), so the fan-out is deterministic.
    """
    for plugin in plugins:
        try:
            plugin.on_run_completed(event)
        except Exception:
            logging.getLogger(f"voxint.plugin.{plugin.manifest.id}").exception(
                "on_run_completed failed for run %s; the run stays COMPLETED",
                event.run_id,
            )


def redispatch_stale_lane_jobs(
    lanes: Sequence[JobLaneSpec],
    *,
    session_factory: sessionmaker[Session],
    send_task: Callable[..., object],
    cutoff: datetime,
) -> dict[str, int]:
    """Re-publish stranded QUEUED jobs for every lane; return per-task counts.

    Mirrors the embedding recovery in ``worker.tasks`` (#130): read each lane's
    stale-QUEUED ids (oldest first, capped at ``lane.limit``) in a short session,
    then send each id to ``lane.redispatch_task_name``. No dispatch lease and no
    row mutation — a job stays QUEUED until a worker claims it, and the claim CAS
    makes a duplicate delivery (or a live worker about to claim it) a no-op. A
    broker outage defers the lane's remaining jobs to a later sweep rather than
    failing the sweep.
    """
    from celery.exceptions import OperationalError

    dispatched: dict[str, int] = {}
    for lane in lanes:
        with session_factory() as session:
            stale_ids = list(
                lane.stale_queued_job_ids(session, cutoff=cutoff, limit=lane.limit)
            )
        count = 0
        for job_id in stale_ids:
            try:
                send_task(
                    lane.redispatch_task_name, args=(str(job_id),), ignore_result=True
                )
            except OperationalError:
                logger.warning(
                    "lane %s recovery enqueue deferred (broker unavailable); job "
                    "%s stays QUEUED for a later sweep",
                    lane.redispatch_task_name,
                    job_id,
                    exc_info=True,
                )
                break  # broker is down; stop this lane, retry next sweep
            count += 1
        if count:
            dispatched[lane.redispatch_task_name] = (
                dispatched.get(lane.redispatch_task_name, 0) + count
            )
    return dispatched
