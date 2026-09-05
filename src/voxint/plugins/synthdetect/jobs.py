"""Synthdetect job lifecycle: create, claim, execute, stale-QUEUED recovery."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from contextlib import suppress
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import case, func, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError

from voxint.app_settings import (
    get_app_settings,
    resolve_effective_synthdetect_enabled,
)
from voxint.db.models import (
    AppSettings,
    DiarizationTurn,
    MediaItem,
    PipelineRun,
    SynthdetectJob,
    SynthdetectJobStatus,
    SynthdetectScore,
)
from voxint.plugins.synthdetect.calibration import (
    DEFAULT_INFERENCE_SPACE,
    DEFAULT_POLICY_ID,
    apply_calibration,
)
from voxint.plugins.synthdetect.client import HttpSynthdetectClient, SynthdetectServiceError

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker

    from voxint.config import Settings

logger = logging.getLogger(__name__)


class SynthdetectHashError(Exception):
    """Raised when a run lacks inputs required for its synthdetect source hash."""


# Bump when client-side window planning changes in a way that can alter scores for the same audio.
WINDOW_PLAN_VERSION = 1


def create_job(
    session: Session,
    pipeline_run_id: uuid.UUID,
    *,
    settings: Settings,
) -> tuple[SynthdetectJob | None, bool]:
    """Create a QUEUED job. Returns (job, already_existed).

    Idempotent: if an active job already exists for this run, returns
    (None, True) without creating a duplicate.
    """
    job = SynthdetectJob(
        pipeline_run_id=pipeline_run_id,
        inference_space=DEFAULT_INFERENCE_SPACE,
        calibration_policy_id=DEFAULT_POLICY_ID,
    )
    try:
        job.source_content_hash = synthdetect_source_hash(session, pipeline_run_id)
    except SynthdetectHashError:
        job.source_content_hash = None
    try:
        with session.begin_nested():
            session.add(job)
            session.flush()
    except IntegrityError as exc:
        constraint = getattr(getattr(exc.orig, "diag", None), "constraint_name", None)
        if constraint != "synthdetect_jobs_one_active_per_run":
            raise
        return None, True
    return job, False


def synthdetect_source_hash(session: Session, pipeline_run_id: uuid.UUID) -> str:
    """Hash the media identity and ordered diarization windows for a run."""
    media_sha256 = session.execute(
        select(MediaItem.sha256)
        .select_from(PipelineRun)
        .join(MediaItem, PipelineRun.media_item_id == MediaItem.id)
        .where(PipelineRun.id == pipeline_run_id)
    ).scalar_one_or_none()
    if not media_sha256:
        raise SynthdetectHashError("media sha256 is unavailable")

    turns = list(
        session.execute(
            select(DiarizationTurn)
            .where(DiarizationTurn.pipeline_run_id == pipeline_run_id)
            .order_by(DiarizationTurn.turn_index)
        ).scalars()
    )
    if not turns:
        raise SynthdetectHashError("no diarization turns")

    canonical = json.dumps(
        {
            "source_schema_version": 1,
            "media_sha256": media_sha256,
            "turns": [
                [int(turn.start_seconds * 1000), int(turn.end_seconds * 1000), turn.label]
                for turn in turns
            ],
            "window_plan_version": WINDOW_PLAN_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _last_succeeded_job_columns(
    session: Session, pipeline_run_id: uuid.UUID
) -> tuple[str | None, str | None, str | None]:
    """Return staleness columns from the run's latest succeeded job."""
    row = session.execute(
        select(
            SynthdetectJob.source_content_hash,
            SynthdetectJob.inference_space,
            SynthdetectJob.calibration_policy_id,
        )
        .where(
            SynthdetectJob.pipeline_run_id == pipeline_run_id,
            SynthdetectJob.status == SynthdetectJobStatus.SUCCEEDED.value,
        )
        .order_by(SynthdetectJob.created_at.desc())
        .limit(1)
    ).one_or_none()
    if row is None:
        return None, None, None
    return row.tuple()


def active_synthdetect_job(
    session: Session, pipeline_run_id: uuid.UUID
) -> SynthdetectJob | None:
    """Return the active (QUEUED or RUNNING) synthdetect job for a run, if any."""
    return session.execute(
        select(SynthdetectJob)
        .where(
            SynthdetectJob.pipeline_run_id == pipeline_run_id,
            SynthdetectJob.status.in_(
                (
                    SynthdetectJobStatus.QUEUED.value,
                    SynthdetectJobStatus.RUNNING.value,
                )
            ),
        )
        .limit(1)
    ).scalar_one_or_none()


def runs_needing_synthdetect(
    session: Session, *, settings: Settings
) -> list[uuid.UUID]:
    """Return completed runs with missing or stale synthdetect results."""
    del settings  # Reserved for audio-availability policy in the next hardening step.
    run_ids = list(
        session.execute(
            select(PipelineRun.id)
            .where(PipelineRun.status == "completed")
            .order_by(PipelineRun.created_at)
        ).scalars()
    )
    needing: list[uuid.UUID] = []
    for run_id in run_ids:
        try:
            source_hash = synthdetect_source_hash(session, run_id)
        except SynthdetectHashError:
            continue
        previous_hash, inference_space, policy_id = _last_succeeded_job_columns(
            session, run_id
        )
        if (
            previous_hash is None
            or previous_hash != source_hash
            or inference_space != DEFAULT_INFERENCE_SPACE
            or policy_id != DEFAULT_POLICY_ID
        ):
            needing.append(run_id)
    return needing


def synthdetect_gates_open(settings: Settings, row: AppSettings | None) -> bool:
    """Whether synthdetect is effectively enabled for this installation."""
    return resolve_effective_synthdetect_enabled(row, settings)


def request_cancel(session: Session, job_id: uuid.UUID) -> bool:
    """Force-cancel an active job outright (caller commits)."""
    cancelled = cast(
        CursorResult[Any],
        session.execute(
            update(SynthdetectJob)
            .where(
                SynthdetectJob.id == job_id,
                SynthdetectJob.status.in_(
                    (
                        SynthdetectJobStatus.QUEUED.value,
                        SynthdetectJobStatus.RUNNING.value,
                    )
                ),
            )
            .values(
                cancel_requested=True,
                status=SynthdetectJobStatus.CANCELLED.value,
                finished_at=case(
                    (SynthdetectJob.started_at.isnot(None), func.now()),
                    else_=None,
                ),
            )
        ),
    )
    return cancelled.rowcount == 1


def _cancel_pending(session: Session, job_id: uuid.UUID) -> bool:
    """Read the cancel flag directly, bypassing the ORM identity map."""
    return bool(
        session.execute(
            select(SynthdetectJob.cancel_requested).where(SynthdetectJob.id == job_id)
        ).scalar_one()
    )


def _finish(
    session_factory: sessionmaker[Session],
    job_id: uuid.UUID,
    *,
    status: SynthdetectJobStatus,
    error: str | None = None,
) -> None:
    """Resolve an active job to a terminal state without overwriting a winner."""
    resolved: Any = status.value
    if status is SynthdetectJobStatus.FAILED:
        resolved = case(
            (
                SynthdetectJob.cancel_requested.is_(True),
                SynthdetectJobStatus.CANCELLED.value,
            ),
            else_=status.value,
        )

    with session_factory() as session:
        finished = cast(
            CursorResult[Any],
            session.execute(
                update(SynthdetectJob)
                .where(
                    SynthdetectJob.id == job_id,
                    SynthdetectJob.status.in_(
                        (
                            SynthdetectJobStatus.QUEUED.value,
                            SynthdetectJobStatus.RUNNING.value,
                        )
                    ),
                )
                .values(
                    status=resolved,
                    error=error[:500] if error else None,
                    finished_at=case(
                        (SynthdetectJob.started_at.isnot(None), func.now()),
                        else_=None,
                    ),
                )
            ),
        )
        if finished.rowcount == 0:
            session.rollback()
            logger.info("synthdetect job %s: already terminal", job_id)
            return
        outcome = session.execute(
            select(SynthdetectJob.status).where(SynthdetectJob.id == job_id)
        ).scalar_one()
        session.commit()

    if outcome == SynthdetectJobStatus.FAILED.value:
        logger.warning("synthdetect job %s: failed — %s", job_id, error)
    else:
        logger.info("synthdetect job %s: %s", job_id, outcome)


def claim_job(session: Session, job_id: uuid.UUID) -> SynthdetectJob | None:
    """queued -> running, exactly once (duplicate delivery no-ops)."""
    claimed = cast(
        CursorResult[Any],
        session.execute(
            update(SynthdetectJob)
            .where(
                SynthdetectJob.id == job_id,
                SynthdetectJob.status == SynthdetectJobStatus.QUEUED.value,
                SynthdetectJob.cancel_requested.is_(False),
            )
            .values(status=SynthdetectJobStatus.RUNNING.value, started_at=func.now())
        ),
    )
    if claimed.rowcount != 1:
        return None
    session.commit()
    return session.get(SynthdetectJob, job_id)


def stale_queued_job_ids(
    session: Session, *, cutoff: datetime, limit: int | None = None
) -> list[uuid.UUID]:
    """Ids of jobs stuck in QUEUED since before ``cutoff``."""
    query = (
        select(SynthdetectJob.id)
        .where(
            SynthdetectJob.status == SynthdetectJobStatus.QUEUED.value,
            SynthdetectJob.created_at < cutoff,
        )
        .order_by(SynthdetectJob.created_at)
    )
    if limit is not None:
        query = query.limit(limit)
    return list(session.execute(query).scalars())


def execute_job(
    session_factory: sessionmaker[Session],
    job_id: uuid.UUID,
    *,
    settings: Settings,
) -> None:
    """Claim the job, score all turns, persist results."""
    from voxint.plugins.media import RunAudioUnavailable, run_audio_descriptor

    with session_factory() as session:
        job = claim_job(session, job_id)
        if job is None:
            logger.info("synthdetect job %s: claim failed (already claimed or cancelled)", job_id)
            return

    try:
        with session_factory() as session:
            row = get_app_settings(session)
            if not synthdetect_gates_open(settings, row):
                _finish(
                    session_factory,
                    job_id,
                    status=SynthdetectJobStatus.FAILED,
                    error="synthdetect was disabled after this job was queued",
                )
                return

        with session_factory() as session:
            if _cancel_pending(session, job_id):
                _finish(
                    session_factory,
                    job_id,
                    status=SynthdetectJobStatus.CANCELLED,
                )
                return

        with session_factory() as session:
            audio = run_audio_descriptor(
                session, job.pipeline_run_id, media_root=settings.media_root
            )

        if audio.reclaimed:
            _finish(
                session_factory,
                job_id,
                status=SynthdetectJobStatus.FAILED,
                error="audio reclaimed",
            )
            return

        with session_factory() as session:
            turns = list(
                session.execute(
                    select(DiarizationTurn)
                    .where(DiarizationTurn.pipeline_run_id == job.pipeline_run_id)
                    .order_by(DiarizationTurn.turn_index)
                ).scalars()
            )

        if not turns:
            _finish(
                session_factory,
                job_id,
                status=SynthdetectJobStatus.FAILED,
                error="no diarization turns",
            )
            return

        client = HttpSynthdetectClient(
            settings.synthdetect_url,
            timeout=settings.synthdetect_http_timeout_seconds,
        )

        intervals = [
            {"start_seconds": t.start_seconds, "end_seconds": t.end_seconds}
            for t in turns
        ]

        response = client.score(audio.media_relative_path, intervals)
        results = response.get("results", [])

        if len(results) != len(turns):
            _finish(
                session_factory,
                job_id,
                status=SynthdetectJobStatus.FAILED,
                error=(
                    f"result count mismatch: {len(results)} results for "
                    f"{len(turns)} turns"
                ),
            )
            return

        scores: list[SynthdetectScore] = []
        scored = 0
        skipped = 0
        risk_values: list[float] = []

        for turn, result in zip(turns, results, strict=True):
            raw = result.get("raw_score")
            skip = result.get("skip_reason")
            wc = result.get("window_count", 0)

            calibrated = None
            if raw is not None:
                calibrated = apply_calibration(raw, job.calibration_policy_id)
                risk_values.append(calibrated)
                scored += 1
            else:
                skipped += 1

            scores.append(
                SynthdetectScore(
                    synthdetect_job_id=job.id,
                    pipeline_run_id=job.pipeline_run_id,
                    diarization_turn_id=turn.id,
                    speaker_label=turn.label,
                    raw_logit=raw,
                    calibrated_score=calibrated,
                    window_count=wc,
                    skip_reason=skip,
                    inference_space=job.inference_space,
                    calibration_policy_id=job.calibration_policy_id,
                )
            )

        with session_factory() as session:
            if _cancel_pending(session, job_id):
                _finish(
                    session_factory,
                    job_id,
                    status=SynthdetectJobStatus.CANCELLED,
                )
                return

        fresh_hash = None
        with session_factory() as session:
            with suppress(SynthdetectHashError):
                fresh_hash = synthdetect_source_hash(session, job.pipeline_run_id)

            session.add_all(scores)
            published = cast(
                CursorResult[Any],
                session.execute(
                    update(SynthdetectJob)
                    .where(
                        SynthdetectJob.id == job_id,
                        SynthdetectJob.status == SynthdetectJobStatus.RUNNING.value,
                        SynthdetectJob.cancel_requested.is_(False),
                    )
                    .values(
                        status=SynthdetectJobStatus.SUCCEEDED.value,
                        total_turns=len(turns),
                        scored_turns=scored,
                        skipped_turns=skipped,
                        mean_risk=(
                            sum(risk_values) / len(risk_values) if risk_values else None
                        ),
                        max_risk=max(risk_values) if risk_values else None,
                        finished_at=func.now(),
                        source_content_hash=fresh_hash,
                    )
                ),
            )
            if published.rowcount != 1:
                session.rollback()
                _finish(
                    session_factory,
                    job_id,
                    status=SynthdetectJobStatus.CANCELLED,
                )
                return
            session.commit()

        logger.info(
            "synthdetect job %s: succeeded (%d scored, %d skipped)",
            job_id, scored, skipped,
        )

    except SynthdetectServiceError as exc:
        _finish(
            session_factory,
            job_id,
            status=SynthdetectJobStatus.FAILED,
            error=f"service error: {exc}",
        )
    except RunAudioUnavailable as exc:
        _finish(
            session_factory,
            job_id,
            status=SynthdetectJobStatus.FAILED,
            error=f"audio unavailable: {exc.code}",
        )
    except Exception:
        logger.exception("synthdetect job %s: unexpected error", job_id)
        _finish(
            session_factory,
            job_id,
            status=SynthdetectJobStatus.FAILED,
            error="unexpected error",
        )
