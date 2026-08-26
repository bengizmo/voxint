"""Read models for the Console 2.0 Jobs area (#160).

Pure, session-in / dataclass-out queries feeding the ``/jobs`` dashboard: a
per-stage pipeline-activity strip and a normalized recent list of the auxiliary
job families. The recent-runs table on ``/jobs`` reuses
:func:`voxint.api.runs_query.list_runs` directly, so it is not rebuilt here.

Two truth rules keep the strip honest (codex-ratified, #160):

* **Queued per stage** buckets every non-archived queued run by its
  ``current_stage`` (a run with no started stage counts under the first stage).
  The buckets partition the queued runs, so their sum equals the queued count
  ``stats_query.run_status_counts`` reports — the strip cannot silently disagree
  with ``voxint stats``.
* **Active per stage** counts DISTINCT live runs that have a *running* stage
  attempt **at the run's current stage**, grouped by that stage. Anchoring on
  ``current_stage`` (a run's single authoritative position) makes active a true
  per-run partition: a run with several running attempt rows at its current
  stage (a stale lease plus a fresh claim) folds to one, and a stale ``running``
  row left at a *previous* stage the run has already moved past cannot make the
  same run appear active at two stages at once. Attempts belonging to archived
  or terminal runs are excluded, so a crashed ``running`` row left behind by a
  finished run never inflates the strip. Because a run contributes to at most
  one stage, the active total never exceeds the running run count; it is below
  it while a running run sits between stages with no worker on it yet, so the
  page describes active as "a worker in that stage now", not as a figure that
  reconciles with Running.

The auxiliary families are heterogeneous: research jobs are speaker-scoped with
a nullable run link, the other three are run-scoped. They are normalized into
one :class:`AuxJob` read model with nullable run *and* speaker targets. All four
share the ``{queued, running, succeeded, failed, cancelled}`` vocabulary — note
``succeeded``, not the ``completed`` that ``RunStatus`` uses. Each family is
read newest-first up to a bound, merged in Python and truncated (no cross-family
SQL union — unnecessary at single-operator scale).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from voxint.db.models import (
    STAGE_ORDER,
    EmbeddingJob,
    PipelineRun,
    ResearchJob,
    RunAssetJob,
    RunStatus,
    StageRun,
    StageStatus,
    TranslationJob,
)

# A run is "live" while it is not in a terminal state; only live runs can have a
# genuinely-running stage attempt, so a stale ``running`` StageRow attached to a
# completed/failed/cancelled run is excluded from the active strip.
_LIVE_RUN_STATUSES: tuple[str, ...] = (
    RunStatus.QUEUED.value,
    RunStatus.RUNNING.value,
    RunStatus.AWAITING_ADJUDICATION.value,
)


@dataclass(frozen=True)
class StageActivity:
    """Queued + active run counts for one pipeline stage."""

    stage: str
    queued: int
    active: int


@dataclass(frozen=True)
class AuxJob:
    """One auxiliary job, normalized across the four families.

    ``pipeline_run_id`` is None for a speaker-only research job; ``speaker_id``
    is set only for research. ``detail`` is the family's most useful one-line
    qualifier (asset kind, target language, embedding space) or empty.
    """

    id: uuid.UUID
    family: str
    status: str
    created_at: datetime
    pipeline_run_id: uuid.UUID | None
    speaker_id: uuid.UUID | None
    detail: str
    error: str | None


def stage_activity(session: Session) -> list[StageActivity]:
    """Queued + active counts per stage, in canonical pipeline order.

    See the module docstring for the truth rules. Returns one entry per stage in
    ``STAGE_ORDER`` (zeros included) so the strip always renders every stage.
    """
    first_stage = STAGE_ORDER[0].value

    queued_by_stage: dict[str, int] = {}
    queued_rows = session.execute(
        select(PipelineRun.current_stage, func.count())
        .where(
            PipelineRun.status == RunStatus.QUEUED.value,
            PipelineRun.archived_at.is_(None),
        )
        .group_by(PipelineRun.current_stage)
    ).all()
    for stage_value, count in queued_rows:
        # A queued run that has not started any stage yet (current_stage NULL)
        # is waiting on the first stage.
        key = stage_value or first_stage
        queued_by_stage[key] = queued_by_stage.get(key, 0) + count

    active_rows = session.execute(
        select(StageRun.stage, func.count(func.distinct(StageRun.pipeline_run_id)))
        .join(PipelineRun, PipelineRun.id == StageRun.pipeline_run_id)
        .where(
            StageRun.status == StageStatus.RUNNING.value,
            PipelineRun.archived_at.is_(None),
            PipelineRun.status.in_(_LIVE_RUN_STATUSES),
            # Anchor on the run's current stage so a stale running row left at a
            # stage the run has already moved past cannot count it active there:
            # a live run contributes to at most one stage (a true partition).
            StageRun.stage == PipelineRun.current_stage,
        )
        .group_by(StageRun.stage)
    ).all()
    active_by_stage = {stage_value: count for stage_value, count in active_rows}

    return [
        StageActivity(
            stage=stage.value,
            queued=queued_by_stage.get(stage.value, 0),
            active=active_by_stage.get(stage.value, 0),
        )
        for stage in STAGE_ORDER
    ]


def jobs_badge_count(session: Session) -> int:
    """Count of live jobs for the shell's Jobs-nav badge (issue #162).

    Non-archived pipeline runs that are queued or running, plus queued/running
    auxiliary jobs across the four families. Shared by the ``/jobs`` page and the
    activity poll endpoint so the badge equals what the page reports by
    construction. Excludes ``awaiting_adjudication`` (paused ON the operator, not
    active work) and every terminal state; ``archived`` runs are already terminal.
    """
    active_aux = ("queued", "running")
    runs = session.execute(
        select(func.count())
        .select_from(PipelineRun)
        .where(
            PipelineRun.status.in_((RunStatus.QUEUED.value, RunStatus.RUNNING.value)),
            PipelineRun.archived_at.is_(None),
        )
    ).scalar_one()
    aux = 0
    for model in (RunAssetJob, TranslationJob, EmbeddingJob, ResearchJob):
        aux += session.execute(
            select(func.count()).select_from(model).where(model.status.in_(active_aux))
        ).scalar_one()
    return int(runs + aux)


def recent_aux_jobs(
    session: Session, *, limit: int = 20, per_family: int | None = None
) -> list[AuxJob]:
    """The newest auxiliary jobs across all four families, merged by recency.

    Each family is read newest-first up to ``per_family`` (defaults to ``limit``)
    rows, normalized to :class:`AuxJob`, then the union is sorted by
    ``created_at`` descending and truncated to ``limit``. Bounding per family
    first keeps a single busy family from starving the others out of the list.
    """
    bound = per_family if per_family is not None else limit
    jobs: list[AuxJob] = []

    for asset in session.execute(
        select(RunAssetJob)
        .order_by(RunAssetJob.created_at.desc(), RunAssetJob.id.desc())
        .limit(bound)
    ).scalars():
        jobs.append(
            AuxJob(
                id=asset.id,
                family="asset",
                status=asset.status,
                created_at=asset.created_at,
                pipeline_run_id=asset.pipeline_run_id,
                speaker_id=None,
                detail=asset.asset_kind,
                error=asset.error,
            )
        )

    for translation in session.execute(
        select(TranslationJob)
        .order_by(TranslationJob.created_at.desc(), TranslationJob.id.desc())
        .limit(bound)
    ).scalars():
        jobs.append(
            AuxJob(
                id=translation.id,
                family="translation",
                status=translation.status,
                created_at=translation.created_at,
                pipeline_run_id=translation.pipeline_run_id,
                speaker_id=None,
                detail=translation.target_language,
                error=translation.error,
            )
        )

    for embedding in session.execute(
        select(EmbeddingJob)
        .order_by(EmbeddingJob.created_at.desc(), EmbeddingJob.id.desc())
        .limit(bound)
    ).scalars():
        jobs.append(
            AuxJob(
                id=embedding.id,
                family="embedding",
                status=embedding.status,
                created_at=embedding.created_at,
                pipeline_run_id=embedding.pipeline_run_id,
                speaker_id=None,
                detail=embedding.embedding_space,
                error=embedding.error,
            )
        )

    for research in session.execute(
        select(ResearchJob)
        .order_by(ResearchJob.created_at.desc(), ResearchJob.id.desc())
        .limit(bound)
    ).scalars():
        # Research is speaker-scoped: the run link is provenance only and may be
        # NULL, so it is never assumed present.
        jobs.append(
            AuxJob(
                id=research.id,
                family="research",
                status=research.status,
                created_at=research.created_at,
                pipeline_run_id=research.pipeline_run_id,
                speaker_id=research.speaker_id,
                detail="",
                error=research.error,
            )
        )

    # Sort by (created_at, id) so jobs sharing a commit timestamp order
    # deterministically, matching the per-family SQL tie-break above.
    jobs.sort(key=lambda job: (job.created_at, job.id), reverse=True)
    return jobs[:limit]
