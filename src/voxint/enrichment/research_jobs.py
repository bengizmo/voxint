"""Research-job lifecycle (issue #40): durable state for one operator-initiated
web-research execution.

The ``research_jobs`` row is orchestration state only — queued → running
(guarded claim, so a duplicate Celery delivery no-ops) → succeeded | failed |
cancelled — plus progress counters the console polls and the cooperative
``cancel_requested`` flag the loop re-reads between rounds. Results live in
the immutable #37 draft tables; ``producer_run_id`` links the two, stamped in
the same transaction that records the producer run (atomic finalization).

Deliberate v1 cuts: no automatic retries and no recovery sweep — a worker
crash leaves an honest ``running`` row whose age the console shows, and the
operator cancels and starts a fresh job. Hidden re-execution of a
non-deterministic web loop is worse than a visible stall.
"""

import uuid
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import CursorResult, select, update
from sqlalchemy.orm import Session, sessionmaker

from voxint.clients.llm import HttpLLMClient
from voxint.config import Settings
from voxint.db.models import ResearchJob, ResearchJobStatus
from voxint.enrichment.producers.web_researcher import (
    WebResearcherError,
    build_seed,
    load_research_speaker,
    make_roster_lookup,
    record_research_outcome,
)
from voxint.research.agent import (
    ChatJsonClient,
    ProgressCounters,
    ResearchAgentError,
    ResearchCancelled,
    run_research_loop,
)
from voxint.research.fetch import ClientFactory
from voxint.research.search import SearchProvider

MAX_ERROR_CHARS = 500


class ResearchJobError(Exception):
    """A job cannot be created or started — gates off, bad target, unknown id."""


def research_gates_open(settings: Settings) -> bool:
    """All three capability gates, together — checked at job creation AND again
    in the worker, so queued work cannot outlive a capability shutdown."""
    return (
        settings.enrichment_web_research_enabled
        and settings.voxint_web_research
        and settings.llm_enabled
    )


def budget_snapshot(settings: Settings) -> dict[str, object]:
    """The budgets the operator's start preview showed — frozen onto the job."""
    return {
        "max_searches": settings.research_max_searches,
        "max_reads": settings.research_max_reads,
        "max_rounds": settings.research_max_rounds,
        "max_actions_per_round": settings.research_max_actions_per_round,
        "deadline_seconds": settings.research_deadline_seconds,
    }


def create_job(
    session: Session,
    *,
    speaker_id: uuid.UUID,
    settings: Settings,
    operator_note: str | None = None,
    pipeline_run_id: uuid.UUID | None = None,
) -> ResearchJob:
    """Validate and insert a QUEUED job (the caller commits, then publishes)."""
    if not research_gates_open(settings):
        raise ResearchJobError(
            "web research is disabled — it needs ENRICHMENT_WEB_RESEARCH_ENABLED,"
            " VOXINT_WEB_RESEARCH, and LLM_ENABLED all true"
        )
    try:
        speaker = load_research_speaker(session, speaker_id)
    except WebResearcherError as exc:
        raise ResearchJobError(str(exc)) from exc
    job = ResearchJob(
        speaker_id=speaker.id,
        pipeline_run_id=pipeline_run_id,
        status=ResearchJobStatus.QUEUED.value,
        budget=budget_snapshot(settings),
        operator_note=(operator_note or None),
    )
    session.add(job)
    session.flush()
    return job


def claim_job(session: Session, job_id: uuid.UUID) -> ResearchJob | None:
    """queued → running, exactly once. None means someone else has (or had) it —
    a duplicate delivery no-ops instead of double-running the loop."""
    claimed = cast(
        CursorResult[Any],
        session.execute(
            update(ResearchJob)
            .where(
                ResearchJob.id == job_id,
                ResearchJob.status == ResearchJobStatus.QUEUED.value,
            )
            .values(
                status=ResearchJobStatus.RUNNING.value,
                started_at=datetime.now(tz=UTC),
            )
        ),
    )
    if claimed.rowcount != 1:
        return None
    session.commit()
    return session.get(ResearchJob, job_id)


def request_cancel(session: Session, job_id: uuid.UUID) -> bool:
    """Set the cooperative flag; a still-QUEUED job is cancelled outright
    (its delivery will fail the claim and no-op). The caller commits."""
    job = session.get(ResearchJob, job_id)
    if job is None or job.status not in (
        ResearchJobStatus.QUEUED.value,
        ResearchJobStatus.RUNNING.value,
    ):
        return False
    job.cancel_requested = True
    if job.status == ResearchJobStatus.QUEUED.value:
        job.status = ResearchJobStatus.CANCELLED.value
    return True


def _finish(
    session: Session,
    job_id: uuid.UUID,
    *,
    status: ResearchJobStatus,
    error: str | None = None,
) -> None:
    session.execute(
        update(ResearchJob)
        .where(ResearchJob.id == job_id)
        .values(
            status=status.value,
            error=error[:MAX_ERROR_CHARS] if error else None,
            finished_at=datetime.now(tz=UTC),
        )
    )
    session.commit()


def execute_job(
    session_factory: sessionmaker[Session],
    job_id: uuid.UUID,
    *,
    settings: Settings,
    llm: ChatJsonClient | None = None,
    search_provider: SearchProvider | None = None,
    read_client_factory: ClientFactory | None = None,
    read_resolver: object | None = None,
) -> None:
    """The worker body: claim, run the loop, finalize. Never raises for job
    outcomes — failures land on the row as bounded, honest ``error`` text.
    ``llm`` / ``search_provider`` / ``read_client_factory`` / ``read_resolver``
    are injection seams (tests; the CLI's inline mode)."""
    with session_factory() as session:
        job = claim_job(session, job_id)
        if job is None:
            return
        if not research_gates_open(settings):
            _finish(
                session,
                job_id,
                status=ResearchJobStatus.FAILED,
                error="web research was disabled after this job was queued",
            )
            return
        try:
            speaker = load_research_speaker(session, job.speaker_id)
            seed = build_seed(session, speaker=speaker, operator_note=job.operator_note)
        except WebResearcherError as exc:
            _finish(session, job_id, status=ResearchJobStatus.FAILED, error=str(exc))
            return

        def should_cancel() -> bool:
            return bool(
                session.execute(
                    select(ResearchJob.cancel_requested).where(ResearchJob.id == job_id)
                ).scalar_one()
            )

        def on_progress(counters: ProgressCounters) -> None:
            session.execute(
                update(ResearchJob)
                .where(ResearchJob.id == job_id)
                .values(
                    searches_used=counters.searches_used,
                    reads_used=counters.reads_used,
                    rounds_used=counters.rounds_used,
                )
            )
            session.commit()

        started_at = datetime.now(tz=UTC)
        owned_client: HttpLLMClient | None = None
        client: ChatJsonClient
        if llm is None:
            owned_client = HttpLLMClient(
                settings.llm_base_url,
                settings.llm_model,
                settings.llm_api_key,
                settings.llm_timeout_seconds,
            )
            client = owned_client
        else:
            client = llm
        try:
            conclusion = run_research_loop(
                llm=client,
                settings=settings,
                seed=seed,
                roster_lookup=make_roster_lookup(session, target_speaker_id=speaker.id),
                should_cancel=should_cancel,
                on_progress=on_progress,
                search_provider=search_provider,
                read_client_factory=read_client_factory,
                read_resolver=read_resolver,  # type: ignore[arg-type]
            )
        except ResearchCancelled:
            _finish(session, job_id, status=ResearchJobStatus.CANCELLED)
            return
        except (ResearchAgentError, WebResearcherError) as exc:
            _finish(session, job_id, status=ResearchJobStatus.FAILED, error=str(exc))
            return
        finally:
            if owned_client is not None:
                owned_client.close()

        # Atomic finalization: the producer run and the job stamp commit together.
        producer_run = record_research_outcome(
            session,
            job_id=job_id,
            speaker_id=speaker.id,
            settings=settings,
            conclusion=conclusion,
            started_at=started_at,
        )
        session.flush()
        session.execute(
            update(ResearchJob)
            .where(ResearchJob.id == job_id)
            .values(
                status=ResearchJobStatus.SUCCEEDED.value,
                producer_run_id=producer_run.id,
                searches_used=conclusion.searches_used,
                reads_used=conclusion.reads_used,
                rounds_used=conclusion.rounds_used,
                finished_at=datetime.now(tz=UTC),
            )
        )
        session.commit()
