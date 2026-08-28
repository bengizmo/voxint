"""Synthdetect job lifecycle: create, claim, execute, stale-QUEUED recovery."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import func, select, update
from sqlalchemy.engine import CursorResult

from voxint.db.models import (
    DiarizationTurn,
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
    existing = session.execute(
        select(SynthdetectJob.id).where(
            SynthdetectJob.pipeline_run_id == pipeline_run_id,
            SynthdetectJob.status.in_((
                SynthdetectJobStatus.QUEUED.value,
                SynthdetectJobStatus.RUNNING.value,
            )),
        )
    ).scalar_one_or_none()
    if existing is not None:
        return None, True

    job = SynthdetectJob(
        pipeline_run_id=pipeline_run_id,
        inference_space=DEFAULT_INFERENCE_SPACE,
        calibration_policy_id=DEFAULT_POLICY_ID,
    )
    session.add(job)
    session.flush()
    return job, False


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
            audio = run_audio_descriptor(
                session, job.pipeline_run_id, media_root=settings.media_root
            )

        if audio.reclaimed:
            _fail_job(session_factory, job_id, "audio reclaimed")
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
            _fail_job(session_factory, job_id, "no diarization turns")
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
            _fail_job(
                session_factory, job_id,
                f"result count mismatch: {len(results)} results for {len(turns)} turns",
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

            scores.append(SynthdetectScore(
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
            ))

        with session_factory() as session:
            session.add_all(scores)
            session.execute(
                update(SynthdetectJob)
                .where(
                    SynthdetectJob.id == job_id,
                    SynthdetectJob.status == SynthdetectJobStatus.RUNNING.value,
                )
                .values(
                    status=SynthdetectJobStatus.SUCCEEDED.value,
                    total_turns=len(turns),
                    scored_turns=scored,
                    skipped_turns=skipped,
                    mean_risk=sum(risk_values) / len(risk_values) if risk_values else None,
                    max_risk=max(risk_values) if risk_values else None,
                    finished_at=func.now(),
                )
            )
            session.commit()

        logger.info(
            "synthdetect job %s: succeeded (%d scored, %d skipped)",
            job_id, scored, skipped,
        )

    except SynthdetectServiceError as exc:
        _fail_job(session_factory, job_id, f"service error: {exc}")
    except RunAudioUnavailable as exc:
        _fail_job(session_factory, job_id, f"audio unavailable: {exc.code}")
    except Exception:
        logger.exception("synthdetect job %s: unexpected error", job_id)
        _fail_job(session_factory, job_id, "unexpected error")


def _fail_job(
    session_factory: sessionmaker[Session],
    job_id: uuid.UUID,
    error: str,
) -> None:
    with session_factory() as session:
        session.execute(
            update(SynthdetectJob)
            .where(
                SynthdetectJob.id == job_id,
                SynthdetectJob.status == SynthdetectJobStatus.RUNNING.value,
            )
            .values(
                status=SynthdetectJobStatus.FAILED.value,
                error=error[:500],
                finished_at=func.now(),
            )
        )
        session.commit()
    logger.warning("synthdetect job %s: failed — %s", job_id, error)
