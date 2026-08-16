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

import logging
import uuid
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import CursorResult, case, func, select, update
from sqlalchemy.orm import Session, sessionmaker

from voxint.clients.llm import HttpLLMClient
from voxint.config import DEFAULT_LLM_TIMEOUT_SECONDS, Settings
from voxint.db.models import ResearchJob, ResearchJobStatus
from voxint.enrichment.producers.web_researcher import (
    WebResearcherError,
    build_seed,
    load_research_speaker,
    make_roster_lookup,
    record_research_outcome,
)
from voxint.media.netcheck import Resolver
from voxint.research.agent import (
    ChatJsonClient,
    ProgressCounters,
    ResearchAgentError,
    ResearchCancelled,
    run_research_loop,
)
from voxint.research.fetch import ClientFactory
from voxint.research.search import SearchProvider

logger = logging.getLogger(__name__)

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


# snapshot key → the Settings field the loop actually reads. The worker
# reconstructs its execution settings from the job's snapshot through this
# map, so the preview the operator approved is the budget that runs — a
# settings change between enqueue and execution never silently applies.
_BUDGET_FIELDS: dict[str, str] = {
    "max_searches": "research_max_searches",
    "max_reads": "research_max_reads",
    "max_rounds": "research_max_rounds",
    "max_actions_per_round": "research_max_actions_per_round",
    "deadline_seconds": "research_deadline_seconds",
}


def budget_snapshot(settings: Settings) -> dict[str, object]:
    """The budgets the operator's start preview showed — frozen onto the job.

    ``llm_timeout_seconds`` rides along for the stale-RUNNING bound in
    :func:`request_cancel`; it is not an operator-approved budget."""
    snapshot: dict[str, object] = {
        key: getattr(settings, field) for key, field in _BUDGET_FIELDS.items()
    }
    snapshot["llm_timeout_seconds"] = settings.llm_timeout_seconds
    return snapshot


def _settings_from_snapshot(settings: Settings, budget: dict[str, object]) -> Settings:
    """Execution settings honoring the job's approved budget snapshot; a key
    missing from an older snapshot falls back to the live setting."""
    update_fields = {
        field: budget[key]
        for key, field in _BUDGET_FIELDS.items()
        if isinstance(budget.get(key), (int, float))
    }
    return settings.model_copy(update=update_fields) if update_fields else settings


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
    a duplicate delivery no-ops instead of double-running the loop. A job whose
    cancel flag is already set is refused too: a cancel that lands between
    enqueue and delivery must win even if the status write raced."""
    claimed = cast(
        CursorResult[Any],
        session.execute(
            update(ResearchJob)
            .where(
                ResearchJob.id == job_id,
                ResearchJob.status == ResearchJobStatus.QUEUED.value,
                ResearchJob.cancel_requested.is_(False),
            )
            .values(
                status=ResearchJobStatus.RUNNING.value,
                # DB clock, like created_at — an app-clock value could trip the
                # started_at >= created_at CHECK under clock skew.
                started_at=func.now(),
            )
        ),
    )
    if claimed.rowcount != 1:
        return None
    session.commit()
    return session.get(ResearchJob, job_id)


# Grace a provably-dead RUNNING job gets past its deadline before the operator
# may force-cancel it. The deadline is checked between rounds, and the forced
# conclude gets its own single repair attempt (research.agent._round_reply),
# so two post-deadline LLM calls are legitimate — the stale bound allows both.
# A round already in flight when the deadline trips can stretch further still;
# a misfire on such a job resolves safely (the finalize CAS turns its late
# outcome into CANCELLED — the state the operator asked for anyway).
STALE_RUNNING_GRACE_SECONDS = 60.0


def _snapshot_llm_timeout(budget: dict[str, Any]) -> float:
    """The per-attempt LLM timeout frozen onto the job at enqueue.

    Both the stale-RUNNING bound and the worker's client construction read it
    through here, so cancellation always reasons about the timeout the job
    actually runs under — never live settings changed since enqueue. The
    fallback covers pre-0.11 snapshots written before the key existed."""
    raw = budget.get("llm_timeout_seconds")
    return float(raw) if isinstance(raw, (int, float)) else DEFAULT_LLM_TIMEOUT_SECONDS


def request_cancel(session: Session, job_id: uuid.UUID) -> bool:
    """Cancel cooperatively — atomically, never clobbering a terminal state.

    One guarded UPDATE sets the flag and resolves QUEUED outright (its delivery
    will fail the claim and no-op); a RUNNING loop observes the flag between
    rounds. A RUNNING job that provably outlived its own wall-clock budget
    (deadline + two LLM timeouts + grace — the forced conclude plus its single
    repair attempt) has no live loop left to observe anything — cancel it
    outright so a worker crash cannot block the speaker forever. The caller
    commits."""
    flagged = cast(
        CursorResult[Any],
        session.execute(
            update(ResearchJob)
            .where(
                ResearchJob.id == job_id,
                ResearchJob.status.in_(
                    (ResearchJobStatus.QUEUED.value, ResearchJobStatus.RUNNING.value)
                ),
            )
            .values(
                cancel_requested=True,
                status=case(
                    (
                        ResearchJob.status == ResearchJobStatus.QUEUED.value,
                        ResearchJobStatus.CANCELLED.value,
                    ),
                    else_=ResearchJob.status,
                ),
            )
        ),
    )
    if flagged.rowcount != 1:
        return False
    # Column select (not session.get) so the identity map cannot serve a
    # pre-UPDATE snapshot of the row just mutated through Core.
    status, started_at, budget = session.execute(
        select(ResearchJob.status, ResearchJob.started_at, ResearchJob.budget).where(
            ResearchJob.id == job_id
        )
    ).one()
    if status == ResearchJobStatus.RUNNING.value and started_at is not None:
        deadline = budget.get("deadline_seconds")
        bound = (
            float(deadline if isinstance(deadline, (int, float)) else 300.0)
            + 2 * _snapshot_llm_timeout(budget)
            + STALE_RUNNING_GRACE_SECONDS
        )
        # DB clock on BOTH sides: started_at was stamped with now() at claim,
        # so an app-clock cutoff would reintroduce exactly the skew the claim
        # path avoided (make_interval's 7th positional argument is seconds).
        session.execute(
            update(ResearchJob)
            .where(
                ResearchJob.id == job_id,
                ResearchJob.status == ResearchJobStatus.RUNNING.value,
                ResearchJob.started_at < func.now() - func.make_interval(0, 0, 0, 0, 0, 0, bound),
            )
            .values(
                status=ResearchJobStatus.CANCELLED.value,
                finished_at=func.now(),
            )
        )
    return True


def _finish(
    session: Session,
    job_id: uuid.UUID,
    *,
    status: ResearchJobStatus,
    error: str | None = None,
) -> None:
    """Guarded active→terminal CAS — a terminal row is never mutated again
    (a force-cancel that already resolved the job must not be overwritten by
    a late worker failure), and a FAILED verdict racing an operator cancel
    resolves to CANCELLED: the operator asked for exactly that outcome."""
    resolved: Any = status.value
    if status is ResearchJobStatus.FAILED:
        resolved = case(
            (ResearchJob.cancel_requested.is_(True), ResearchJobStatus.CANCELLED.value),
            else_=status.value,
        )
    session.execute(
        update(ResearchJob)
        .where(
            ResearchJob.id == job_id,
            ResearchJob.status.in_(
                (ResearchJobStatus.QUEUED.value, ResearchJobStatus.RUNNING.value)
            ),
        )
        .values(
            status=resolved,
            error=error[:MAX_ERROR_CHARS] if error else None,
            finished_at=func.now(),
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
    read_resolver: Resolver | None = None,
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

        # The loop runs under the budgets the operator approved at start —
        # never under settings changed since (the snapshot is the contract).
        exec_settings = _settings_from_snapshot(settings, job.budget)
        started_at = job.started_at or datetime.now(tz=UTC)
        owned_client: HttpLLMClient | None = None
        client: ChatJsonClient
        if llm is None:
            # The snapshotted timeout, not the live one: request_cancel's
            # stale-RUNNING bound is computed from the snapshot, so the client
            # must run under the same value or a settings change between
            # enqueue and execution could force-cancel a still-live request.
            owned_client = HttpLLMClient(
                settings.llm_base_url,
                settings.llm_model,
                settings.llm_api_key,
                _snapshot_llm_timeout(job.budget),
            )
            client = owned_client
        else:
            client = llm
        try:
            conclusion = run_research_loop(
                llm=client,
                settings=exec_settings,
                seed=seed,
                roster_lookup=make_roster_lookup(session, target_speaker_id=speaker.id),
                should_cancel=should_cancel,
                on_progress=on_progress,
                search_provider=search_provider,
                read_client_factory=read_client_factory,
                read_resolver=read_resolver,
            )
        except ResearchCancelled:
            _finish(session, job_id, status=ResearchJobStatus.CANCELLED)
            return
        except (ResearchAgentError, WebResearcherError) as exc:
            _finish(session, job_id, status=ResearchJobStatus.FAILED, error=str(exc))
            return
        except Exception as exc:
            # Last-resort honesty: an unexpected failure must never leave the
            # job RUNNING forever. Closed-vocabulary error only — arbitrary
            # exception text is not persist-safe; detail goes to the log.
            logger.exception("research job %s failed unexpectedly", job_id)
            session.rollback()
            _finish(
                session,
                job_id,
                status=ResearchJobStatus.FAILED,
                error=f"unexpected error ({type(exc).__name__}) — see worker logs",
            )
            return
        finally:
            if owned_client is not None:
                owned_client.close()

        # Atomic finalization: the producer run and the job stamp commit
        # together, and only while the row is still RUNNING with no cancel
        # pending — a cancel that lands between the check below and this stamp
        # must win (the drafts roll back with the missed stamp), and so must a
        # force-cancel (a force-cancelled job must not be stamped SUCCEEDED nor
        # keep its drafts). The whole block — the final cancel read included —
        # sits under the same failure umbrella as the loop: a DB error here
        # must land as an honest FAILED row, never a forever-RUNNING job
        # (there is no recovery sweep).
        try:
            # A cancel that raced the loop's final round wins: check the flag
            # before persisting anything (the loop's between-round checks
            # cannot see a flag set after its last read).
            if should_cancel():
                _finish(session, job_id, status=ResearchJobStatus.CANCELLED)
                return
            producer_run = record_research_outcome(
                session,
                job_id=job_id,
                speaker_id=speaker.id,
                settings=exec_settings,
                conclusion=conclusion,
                started_at=started_at,
            )
            session.flush()
            stamped = cast(
                CursorResult[Any],
                session.execute(
                    update(ResearchJob)
                    .where(
                        ResearchJob.id == job_id,
                        ResearchJob.status == ResearchJobStatus.RUNNING.value,
                        ResearchJob.cancel_requested.is_(False),
                    )
                    .values(
                        status=ResearchJobStatus.SUCCEEDED.value,
                        producer_run_id=producer_run.id,
                        searches_used=conclusion.searches_used,
                        reads_used=conclusion.reads_used,
                        rounds_used=conclusion.rounds_used,
                        finished_at=func.now(),
                    )
                ),
            )
            if stamped.rowcount != 1:
                # Cancel won the race; the drafts roll back with us and the
                # cooperative flag (if the row is still RUNNING) resolves.
                session.rollback()
                _finish(session, job_id, status=ResearchJobStatus.CANCELLED)
                return
            session.commit()
        except WebResearcherError as exc:
            session.rollback()
            _finish(session, job_id, status=ResearchJobStatus.FAILED, error=str(exc))
        except Exception as exc:
            logger.exception("research job %s failed during finalization", job_id)
            session.rollback()
            _finish(
                session,
                job_id,
                status=ResearchJobStatus.FAILED,
                error=f"unexpected error ({type(exc).__name__}) — see worker logs",
            )
