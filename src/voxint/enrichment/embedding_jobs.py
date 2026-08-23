"""Embedding-index job lifecycle (#121): one build attempt per (run, space).

A dedicated lane, deliberately NOT the LLM-coupled run-asset job family — the
embedder is a local ONNX graph with no LLM, no ``llm_enabled`` gate, and no
egress. It reuses the proven *patterns* only:

- ``status`` moves queued → running (guarded claim, so a duplicate Celery
  delivery no-ops) → succeeded | failed | cancelled;
- a partial unique index allows one active job per (run, space);
- ``source_content_hash`` is the staleness detector.

The write model is a **whole-run atomic rebuild** (paragraph boundaries shift on
any correction/split/speaker change, so per-chunk surgery is fragile): under a
per-(run, space) transaction-scoped advisory lock the executor allocates the next
``generation``, inserts that generation's chunks, deletes every prior
generation, and stamps the job SUCCEEDED — all in one transaction, so a reader
sees either the whole old index or the whole new one, never a mix.

Two concurrency subtleties, both from a pre-implementation review:

- **Consistent-snapshot read.** Resolving the transcript spans several
  statements; under READ COMMITTED a correction committing mid-read could
  persist a *hybrid* vector index (worse than the run-asset lane's tolerated
  torn snapshot, which self-heals). So :func:`_snapshot_source` reads and hashes
  in its own short REPEATABLE READ transaction and commits it BEFORE the
  CPU-bound embed, holding no snapshot or lock during it.
- **Force-cancel is fencing.** Embedding has no external side effect, so a
  crashed RUNNING job (which would otherwise hold the one-active slot forever
  with no natural deadline) is force-cancellable outright. The guarded terminal
  CAS (``WHERE status='running' AND NOT cancel_requested``) fences a still-
  computing worker from publishing after a cancel admitted a replacement; the
  advisory lock serializes any overlapping finalization. No age-based sweep in
  v1 — that needs heartbeat/lease state to tell a slow job from a dead one.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, cast

from sqlalchemy import (
    CursorResult,
    case,
    delete,
    func,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from voxint.app_settings import (
    get_app_settings,
    resolve_effective_semantic_index_enabled,
)
from voxint.config import Settings
from voxint.db.models import (
    AppSettings,
    EmbeddingJob,
    EmbeddingJobStatus,
    PipelineRun,
    RunStatus,
    SegmentEmbedding,
)
from voxint.embeddings.onnx_embedder import EMBEDDING_SPACE, TextEmbedder, get_text_embedder
from voxint.enrichment.producers.segment_embeddings import (
    ChunkEmbedding,
    EmbeddingError,
    EmbeddingSource,
    embedding_source_hash,
    load_embedding_source,
    produce_segment_embeddings,
)

logger = logging.getLogger(__name__)

MAX_ERROR_CHARS = 500


class EmbeddingJobError(Exception):
    """A job cannot be created — the feature is off, or the run has no transcript."""


def embedding_gates_open(settings: Settings, row: AppSettings | None) -> bool:
    """Whether the semantic-index feature is effectively enabled.

    Independent of ``llm_enabled`` — the embedder needs no LLM. Resolved
    row-over-env so a UI toggle applies with no restart; checked at job creation
    AND again in the worker, so queued work cannot outlive a capability shutdown.
    """
    return resolve_effective_semantic_index_enabled(row, settings)


def create_jobs(
    session: Session,
    *,
    pipeline_run_id: uuid.UUID,
    settings: Settings,
) -> tuple[EmbeddingJob | None, bool]:
    """Validate and insert one QUEUED job for the active space (caller commits).

    Returns ``(job, already_active)`` — a run whose (run, space) slot already has
    an active job yields ``(None, True)`` (a skip, not an error), so a "reindex
    all" action degrades per run. The friendly pre-check is the partial unique
    index itself: the insert runs in a savepoint and its IntegrityError maps to
    the skip (check-then-insert would race). Raises :class:`EmbeddingJobError`
    when the feature is disabled or the run has no transcript to embed.

    The enqueue-time ``source_content_hash`` is provisional: the executor
    recomputes it authoritatively under a consistent snapshot at run time and
    restamps the succeeded job, so a transcript that changes between enqueue and
    execution is embedded and recorded as what was actually read.
    """
    row = get_app_settings(session)
    if not embedding_gates_open(settings, row):
        raise EmbeddingJobError(
            "semantic search is disabled — enable it with SEMANTIC_INDEX_ENABLED"
            " (or the in-UI toggle)"
        )
    try:
        source = load_embedding_source(session, pipeline_run_id)
    except EmbeddingError as exc:
        raise EmbeddingJobError(str(exc)) from exc
    job = EmbeddingJob(
        pipeline_run_id=pipeline_run_id,
        embedding_space=EMBEDDING_SPACE,
        status=EmbeddingJobStatus.QUEUED.value,
        cancel_requested=False,
        source_content_hash=embedding_source_hash(source),
    )
    try:
        with session.begin_nested():
            session.add(job)
            session.flush()
    except IntegrityError as exc:
        constraint = getattr(getattr(exc.orig, "diag", None), "constraint_name", None)
        if constraint != "embedding_jobs_one_active_per_run_space":
            raise
        return None, True
    return job, False


def claim_job(session: Session, job_id: uuid.UUID) -> EmbeddingJob | None:
    """queued → running, exactly once (duplicate delivery no-ops). A job whose
    cancel flag already landed is refused so a cancel that raced the enqueue
    wins."""
    claimed = cast(
        CursorResult[Any],
        session.execute(
            update(EmbeddingJob)
            .where(
                EmbeddingJob.id == job_id,
                EmbeddingJob.status == EmbeddingJobStatus.QUEUED.value,
                EmbeddingJob.cancel_requested.is_(False),
            )
            # DB clock, like created_at — an app-clock value could trip the
            # started_at >= created_at CHECK under clock skew.
            .values(status=EmbeddingJobStatus.RUNNING.value, started_at=func.now())
        ),
    )
    if claimed.rowcount != 1:
        return None
    session.commit()
    return session.get(EmbeddingJob, job_id)


def request_cancel(session: Session, job_id: uuid.UUID) -> bool:
    """Force-cancel an active job outright (caller commits).

    Embedding has no in-flight external effect, so unlike the run-asset lane this
    needs no deadline: a RUNNING job is cancelled immediately, freeing the
    one-active slot for a replacement. A still-computing worker is fenced off by
    the executor's guarded success CAS (it will roll back). ``finished_at`` is
    stamped only for RUNNING (a QUEUED job has no ``started_at``, and the
    finished_at⇒started_at CHECK forbids it there)."""
    cancelled = cast(
        CursorResult[Any],
        session.execute(
            update(EmbeddingJob)
            .where(
                EmbeddingJob.id == job_id,
                EmbeddingJob.status.in_(
                    (EmbeddingJobStatus.QUEUED.value, EmbeddingJobStatus.RUNNING.value)
                ),
            )
            .values(
                cancel_requested=True,
                status=EmbeddingJobStatus.CANCELLED.value,
                finished_at=case(
                    (
                        EmbeddingJob.status == EmbeddingJobStatus.RUNNING.value,
                        func.now(),
                    ),
                    else_=None,
                ),
            )
        ),
    )
    return cancelled.rowcount == 1


def _cancel_pending(session: Session, job_id: uuid.UUID) -> bool:
    """Whether a cancel has been requested — a column select (not the identity
    map) so a cancel committed by another session after claim is seen."""
    return bool(
        session.execute(
            select(EmbeddingJob.cancel_requested).where(EmbeddingJob.id == job_id)
        ).scalar_one()
    )


def _finish(
    session: Session,
    job_id: uuid.UUID,
    *,
    status: EmbeddingJobStatus,
    error: str | None = None,
) -> None:
    """Guarded active→terminal CAS for the FAILED/CANCELLED outcomes — a terminal
    row is never mutated again (a force-cancel that already resolved the job must
    not be overwritten by a late worker failure), and a FAILED verdict racing an
    operator cancel resolves to CANCELLED: the operator asked for exactly that.

    ``finished_at`` is stamped only when ``started_at`` is set (the job was
    claimed), so failing a never-started job cannot trip the finished_at⇒
    started_at CHECK."""
    resolved: Any = status.value
    if status is EmbeddingJobStatus.FAILED:
        resolved = case(
            (EmbeddingJob.cancel_requested.is_(True), EmbeddingJobStatus.CANCELLED.value),
            else_=status.value,
        )
    session.execute(
        update(EmbeddingJob)
        .where(
            EmbeddingJob.id == job_id,
            EmbeddingJob.status.in_(
                (EmbeddingJobStatus.QUEUED.value, EmbeddingJobStatus.RUNNING.value)
            ),
        )
        .values(
            status=resolved,
            error=error[:MAX_ERROR_CHARS] if error else None,
            finished_at=case(
                (EmbeddingJob.started_at.isnot(None), func.now()), else_=None
            ),
        )
    )
    session.commit()


def _snapshot_source(
    session_factory: sessionmaker[Session], pipeline_run_id: uuid.UUID
) -> tuple[EmbeddingSource, str]:
    """Resolve and hash the transcript under one consistent snapshot.

    Its own short REPEATABLE READ transaction, committed before the caller does
    any CPU work, so the multi-statement resolution can never produce a hybrid
    index and no snapshot/lock is held during the embed. A dedicated session
    keeps the isolation level from leaking into the publish transaction."""
    with session_factory() as read_session:
        read_session.connection(
            execution_options={"isolation_level": "REPEATABLE READ"}
        )
        source = load_embedding_source(read_session, pipeline_run_id)
        source_hash = embedding_source_hash(source)
        read_session.commit()
        return source, source_hash


def _publish(
    session: Session,
    job_id: uuid.UUID,
    *,
    pipeline_run_id: uuid.UUID,
    embedding_space: str,
    chunks: list[ChunkEmbedding],
    source_hash: str,
) -> None:
    """Atomically replace the run's index and stamp the job SUCCEEDED.

    Under a per-(run, space) advisory lock: allocate the next generation from the
    run's succeeded jobs (NOT from the vector rows — an intentionally empty
    generation or manual cleanup must not reset the counter), insert this
    generation's chunks, delete every prior generation, then a guarded RUNNING→
    SUCCEEDED CAS. If the CAS matches no row a force-cancel won the race: roll the
    whole rebuild back and resolve CANCELLED."""
    session.execute(
        text("SELECT pg_advisory_xact_lock(hashtext(:producer), hashtext(:scope))"),
        {"producer": f"segment_embeddings:{embedding_space}", "scope": str(pipeline_run_id)},
    )
    previous = session.execute(
        select(func.coalesce(func.max(EmbeddingJob.generation), 0)).where(
            EmbeddingJob.pipeline_run_id == pipeline_run_id,
            EmbeddingJob.embedding_space == embedding_space,
            EmbeddingJob.status == EmbeddingJobStatus.SUCCEEDED.value,
        )
    ).scalar_one()
    generation = (previous or 0) + 1
    if chunks:
        session.execute(
            insert(SegmentEmbedding),
            [
                {
                    "pipeline_run_id": pipeline_run_id,
                    "embedding_space": embedding_space,
                    "generation": generation,
                    "chunk_index": chunk.chunk_index,
                    "start_seconds": chunk.start_seconds,
                    "end_seconds": chunk.end_seconds,
                    "speaker_label": chunk.speaker_label,
                    "text_rendering": chunk.text_rendering,
                    "chunk_text": chunk.chunk_text,
                    "content_hash": chunk.content_hash,
                    "embedding": chunk.embedding,
                }
                for chunk in chunks
            ],
        )
    session.execute(
        delete(SegmentEmbedding).where(
            SegmentEmbedding.pipeline_run_id == pipeline_run_id,
            SegmentEmbedding.embedding_space == embedding_space,
            SegmentEmbedding.generation < generation,
        )
    )
    session.flush()
    stamped = cast(
        CursorResult[Any],
        session.execute(
            update(EmbeddingJob)
            .where(
                EmbeddingJob.id == job_id,
                EmbeddingJob.status == EmbeddingJobStatus.RUNNING.value,
                EmbeddingJob.cancel_requested.is_(False),
            )
            .values(
                status=EmbeddingJobStatus.SUCCEEDED.value,
                generation=generation,
                source_content_hash=source_hash,
                finished_at=func.now(),
            )
        ),
    )
    if stamped.rowcount != 1:
        session.rollback()
        _finish(session, job_id, status=EmbeddingJobStatus.CANCELLED)
        return
    session.commit()


def execute_job(
    session_factory: sessionmaker[Session],
    job_id: uuid.UUID,
    *,
    settings: Settings,
    embedder: TextEmbedder | None = None,
) -> None:
    """The worker body: claim, resolve, embed, publish. Never raises for job
    outcomes — failures land on the row as bounded, honest ``error`` text; the
    whole post-claim lifecycle is under one rollback-and-finish boundary so an
    ordinary DB error cannot strand a RUNNING row. ``embedder`` is an injection
    seam (tests; the CLI's inline mode)."""
    with session_factory() as session:
        job = claim_job(session, job_id)
        if job is None:
            return
        pipeline_run_id = job.pipeline_run_id
        embedding_space = job.embedding_space
        try:
            row = get_app_settings(session)
            if not embedding_gates_open(settings, row):
                _finish(
                    session,
                    job_id,
                    status=EmbeddingJobStatus.FAILED,
                    error="semantic search was disabled after this job was queued",
                )
                return
            if _cancel_pending(session, job_id):
                _finish(session, job_id, status=EmbeddingJobStatus.CANCELLED)
                return
            source, source_hash = _snapshot_source(session_factory, pipeline_run_id)
            if _cancel_pending(session, job_id):
                _finish(session, job_id, status=EmbeddingJobStatus.CANCELLED)
                return
            chunks = produce_segment_embeddings(
                source, embedder if embedder is not None else get_text_embedder()
            )
            if _cancel_pending(session, job_id):
                _finish(session, job_id, status=EmbeddingJobStatus.CANCELLED)
                return
            _publish(
                session,
                job_id,
                pipeline_run_id=pipeline_run_id,
                embedding_space=embedding_space,
                chunks=chunks,
                source_hash=source_hash,
            )
        except EmbeddingError as exc:
            session.rollback()
            _finish(session, job_id, status=EmbeddingJobStatus.FAILED, error=str(exc))
        except FileNotFoundError as exc:
            # The embedder raises this with an actionable message when the
            # vendored MiniLM weights are absent (e.g. a native install run
            # via `voxint embed backfill` before the minilm-onnx-v1 asset was
            # fetched). Preserve it verbatim so the operator sees "weights not
            # found", not a generic "unexpected error".
            session.rollback()
            _finish(session, job_id, status=EmbeddingJobStatus.FAILED, error=str(exc))
        except Exception as exc:
            logger.exception("embedding job %s failed unexpectedly", job_id)
            session.rollback()
            _finish(
                session,
                job_id,
                status=EmbeddingJobStatus.FAILED,
                error=f"unexpected error ({type(exc).__name__}) — see worker logs",
            )


def _last_succeeded_hash(
    session: Session, pipeline_run_id: uuid.UUID, embedding_space: str
) -> str | None:
    """The ``source_content_hash`` of the run's current generation (its latest
    succeeded job for the space), or ``None`` when it has never been indexed."""
    return session.execute(
        select(EmbeddingJob.source_content_hash)
        .where(
            EmbeddingJob.pipeline_run_id == pipeline_run_id,
            EmbeddingJob.embedding_space == embedding_space,
            EmbeddingJob.status == EmbeddingJobStatus.SUCCEEDED.value,
        )
        .order_by(EmbeddingJob.generation.desc())
        .limit(1)
    ).scalar_one_or_none()


def active_job(
    session: Session,
    pipeline_run_id: uuid.UUID,
    *,
    embedding_space: str = EMBEDDING_SPACE,
) -> EmbeddingJob | None:
    """The one active (queued|running) job for the (run, space) slot, or None.

    The partial unique index ``embedding_jobs_one_active_per_run_space`` guarantees
    at most one, so this is what ``create_jobs`` collided with when it returned
    ``already_active``. Callers that recover a stranded job (the CLI backfill lever
    for #130) use it to tell a re-runnable QUEUED job from a live RUNNING one."""
    return session.execute(
        select(EmbeddingJob).where(
            EmbeddingJob.pipeline_run_id == pipeline_run_id,
            EmbeddingJob.embedding_space == embedding_space,
            EmbeddingJob.status.in_(
                (EmbeddingJobStatus.QUEUED.value, EmbeddingJobStatus.RUNNING.value)
            ),
        )
    ).scalar_one_or_none()


def stale_queued_job_ids(session: Session, *, cutoff: datetime) -> list[uuid.UUID]:
    """Ids of jobs stuck in QUEUED since before ``cutoff`` (#130 recovery).

    A job committed QUEUED whose Celery dispatch evaporated (process/broker died
    between the row commit and ``.delay()``) sits QUEUED forever, holding the
    one-active slot so no new job can be created for the run. ``created_at`` is the
    only age signal — a QUEUED job has no ``started_at``, and its only legitimate
    next transition is a guarded claim or a direct cancel, so it is never touched
    before the cutoff. The recovery sweep re-dispatches these by id (no row
    mutation); the guarded claim CAS makes a duplicate delivery a no-op."""
    return list(
        session.execute(
            select(EmbeddingJob.id).where(
                EmbeddingJob.status == EmbeddingJobStatus.QUEUED.value,
                EmbeddingJob.created_at < cutoff,
            )
        ).scalars()
    )


def runs_needing_embeddings(
    session: Session, *, embedding_space: str = EMBEDDING_SPACE
) -> list[uuid.UUID]:
    """Completed runs with no current index, or whose transcript has changed
    since it was last indexed (staleness).

    A run needs (re)embedding when it has no succeeded job for the space, or when
    its freshly-resolved ``source_content_hash`` differs from that job's stamped
    hash. The fresh hash is read under READ COMMITTED (a torn snapshot here only
    risks one spurious re-embed, which the executor's authoritative RR recompute
    self-heals). Runs whose transcript cannot be resolved are skipped."""
    run_ids = (
        session.execute(
            select(PipelineRun.id)
            .where(PipelineRun.status == RunStatus.COMPLETED.value)
            .order_by(PipelineRun.created_at)
        )
        .scalars()
        .all()
    )
    needing: list[uuid.UUID] = []
    for run_id in run_ids:
        try:
            source = load_embedding_source(session, run_id)
        except EmbeddingError:
            continue
        current = _last_succeeded_hash(session, run_id, embedding_space)
        if current is None or current != embedding_source_hash(source):
            needing.append(run_id)
    return needing


__all__ = [
    "EmbeddingJobError",
    "active_job",
    "claim_job",
    "create_jobs",
    "embedding_gates_open",
    "execute_job",
    "request_cancel",
    "runs_needing_embeddings",
    "stale_queued_job_ids",
]
