"""Clean-up job lifecycle (#758): durable state for one generation attempt.

The ``cleanup_jobs`` row is orchestration state only, cloned from
``translation_jobs``: queued, then running (a guarded claim, so a duplicate
Celery delivery no-ops), then succeeded, failed or cancelled. The *result* is
an immutable ``run_cleanups`` row written by ``cleanups.record_cleanup`` and
linked through ``cleanup_id`` in the same UPDATE that stamps the job
SUCCEEDED (the table requires the two together). A failed or cancelled job
records nothing and consumes no generation.

Guards beyond the translation template:

- **English only (D6)**: a run detected as another language is refused at
  creation and again at claim. A run with no detected language is allowed;
  the console says the language was not verified.
- **Cancel** is checked before and after every LLM call and immediately
  before persisting, so a reply that lands after a cancel is discarded.
- **Source-changed race**: the executor freezes one source at job start and
  recomputes the current transcript hash immediately before persisting; a
  mismatch fails the job, so an edit made while the model was working never
  retires a still-valid generation. Best-effort, as in translation: an edit
  committing between that check and the insert leaves a generation that
  every reader sees as stale.
- **Snapshot validation**: the job executes from the settings snapshot taken
  at enqueue; a snapshot of the wrong shape fails the job instead of being
  silently replaced by current settings.

There is no auto-generation after finalize and no Celery retry. The recovery
sweep only republishes stale QUEUED jobs; cancel is deadline-aware like
translation's, so a crashed RUNNING row can be cleared by the operator.
"""

import logging
import uuid
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Protocol, cast

from sqlalchemy import CursorResult, case, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from voxint.api.languages import language_label
from voxint.app_settings import (
    get_app_settings,
    llm_bundled_active,
    resolve_effective_filler_list,
    resolve_effective_llm_api_key,
    resolve_effective_llm_enabled,
    resolve_effective_llm_endpoint,
)
from voxint.clients.llm import HttpLLMClient, SamplingProfile
from voxint.config import DEFAULT_LLM_TIMEOUT_SECONDS, Settings
from voxint.db.models import POST_SEGMENT, AppSettings, CleanupJob, CleanupJobStatus
from voxint.enrichment.cleanups import (
    CleanupError,
    filler_list_snapshot,
    load_cleanup_source,
    record_cleanup,
)
from voxint.enrichment.producers.cleanup_llm import (
    CLEANUP_PROMPT_VERSION,
    PRODUCER_NAME,
    PRODUCER_VERSION,
    CleanupCancelled,
    CleanupProducerError,
    propose_cleanup,
)
from voxint.enrichment.producers.cleanup_llm import ChatJsonLLM as ProducerChatJsonLLM
from voxint.enrichment.translation_jobs import normalized_language
from voxint.enrichment.translations import (
    TranslationError,
    load_translation_source,
    translation_source_hash,
)
from voxint.export.filler_lists import FillerListError
from voxint.gpu_phase.state import admit_lane

logger = logging.getLogger(__name__)

MAX_ERROR_CHARS = 500
# Same arithmetic and grace as translation's force-cancel of a dead executor.
STALE_RUNNING_GRACE_SECONDS = 60.0

ENGLISH = "en"
SOURCE_CHANGED_ERROR = (
    "the transcript changed while it was being cleaned up; generate the clean-up again"
)
SNAPSHOT_INVALID_ERROR = (
    "this clean-up job's saved settings are not valid; generate the clean-up again"
)
_ACTIVE = (CleanupJobStatus.QUEUED.value, CleanupJobStatus.RUNNING.value)


class ChatJsonLLM(Protocol):
    """The only capability the executor needs from a client (injection seam)."""

    def chat_json(self, messages: Any) -> dict[str, object]: ...


class CleanupJobError(Exception):
    """A job cannot be created: gates off, wrong language, no transcript."""


def cleanup_gates_open(settings: Settings, row: AppSettings | None) -> bool:
    """Checked at creation and again in the worker. Clean-up rides on the
    configured LLM and has no feature flag of its own."""
    return resolve_effective_llm_enabled(row, settings)


def language_refusal(detected_language: str | None) -> str | None:
    """Why a run's detected language rules clean-up out (D6), or None.

    ``None`` (not detected) is allowed; the console says it was not verified.
    """
    code = normalized_language(detected_language)
    if code is None or code == ENGLISH:
        return None
    return (
        "clean-up works on English transcripts only, and this run was detected as"
        f" {language_label(code)}"
    )


# Snapshot key: (Settings field, validator). The executor runs from these
# values, never from settings changed since enqueue (the #40 doctrine).
def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _positive_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and value > 0


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _bool(value: object) -> bool:
    return isinstance(value, bool)


_KNOBS = ("llm_attempts_per_batch", "llm_batch_max_segments", "llm_batch_max_chars")
_CONFIG_FIELDS: dict[str, tuple[str, Any]] = {
    "model": ("llm_model", _text),
    "base_url": ("llm_base_url", _text),
    "llm_timeout_seconds": ("llm_timeout_seconds", _positive_number),
    "llm_disable_thinking": ("llm_disable_thinking", _bool),
    **{knob: (knob, _positive_int) for knob in _KNOBS},
}


def job_config_snapshot(settings: Settings, filler_list: Mapping[str, Any]) -> dict[str, object]:
    snapshot: dict[str, object] = {
        key: getattr(settings, field) for key, (field, _check) in _CONFIG_FIELDS.items()
    }
    snapshot["filler_list"] = dict(filler_list)
    return snapshot


def _valid_filler_list(value: object) -> bool:
    if not isinstance(value, Mapping) or not isinstance(value.get("preset_version"), str):
        return False
    return all(
        isinstance(value.get(key), list) and all(isinstance(e, str) for e in value[key])
        for key in ("words", "phrases", "kept")
    )


def _settings_from_snapshot(
    settings: Settings, config: Mapping[str, Any]
) -> tuple[Settings, dict[str, Any]]:
    """The executed settings and filler list, or :class:`CleanupJobError`."""
    if not isinstance(config, Mapping):
        raise CleanupJobError(SNAPSHOT_INVALID_ERROR)
    update_fields: dict[str, object] = {}
    for key, (field, check) in _CONFIG_FIELDS.items():
        if not check(config.get(key)):
            raise CleanupJobError(SNAPSHOT_INVALID_ERROR)
        update_fields[field] = config[key]
    fillers = config.get("filler_list")
    if not _valid_filler_list(fillers):
        raise CleanupJobError(SNAPSHOT_INVALID_ERROR)
    return settings.model_copy(update=update_fields), dict(cast(Mapping[str, Any], fillers))


def create_job(
    session: Session, *, pipeline_run_id: uuid.UUID, settings: Settings
) -> tuple[CleanupJob | None, bool]:
    """Validate and insert one QUEUED job (the caller commits, then publishes).

    Returns ``(job, already_active)``. An active job for the run is a skip,
    mapped from the partial unique index itself (check-then-insert would race).
    """
    row = get_app_settings(session)
    if not cleanup_gates_open(settings, row):
        raise CleanupJobError(
            "clean-up is off: it needs the language model enabled"
            " (env LLM_ENABLED or the in-UI toggle)"
        )
    try:
        source = load_cleanup_source(session, pipeline_run_id)
    except CleanupError as exc:
        raise CleanupJobError(str(exc)) from exc
    refusal = language_refusal(source.translation.source_language)
    if refusal is not None:
        raise CleanupJobError(refusal)
    try:
        fillers = filler_list_snapshot(resolve_effective_filler_list(row, settings))
    except FillerListError as exc:
        raise CleanupJobError(str(exc)) from exc
    base_url, model = resolve_effective_llm_endpoint(row, settings)
    snapshot = job_config_snapshot(
        settings.model_copy(update={"llm_base_url": base_url, "llm_model": model}), fillers
    )
    job = CleanupJob(
        pipeline_run_id=pipeline_run_id,
        status=CleanupJobStatus.QUEUED.value,
        config=snapshot,
        source_content_hash=translation_source_hash(source.translation),
    )
    try:
        with session.begin_nested():
            session.add(job)
            session.flush()
    except IntegrityError as exc:
        constraint = getattr(getattr(exc.orig, "diag", None), "constraint_name", None)
        if constraint != "cleanup_jobs_one_active_per_run":
            raise
        return None, True
    return job, False


def _db_now(session: Session) -> datetime:
    """The database's wall clock now. Not ``now()``: the executor's
    transaction stays open across the LLM calls, and ``now()`` is the
    transaction's start, which would stamp completion before inference ran."""
    return session.execute(select(func.clock_timestamp())).scalar_one()


def claim_job(session: Session, job_id: uuid.UUID) -> CleanupJob | None:
    """queued to running, exactly once; a job already flagged for cancel is
    refused, so a cancel landing between enqueue and delivery wins."""
    claimed = cast(
        CursorResult[Any],
        session.execute(
            update(CleanupJob)
            .where(
                CleanupJob.id == job_id,
                CleanupJob.status == CleanupJobStatus.QUEUED.value,
                CleanupJob.cancel_requested.is_(False),
            )
            .values(status=CleanupJobStatus.RUNNING.value, started_at=func.now())
        ),
    )
    if claimed.rowcount != 1:
        return None
    session.commit()
    return session.get(CleanupJob, job_id)


def request_cancel(session: Session, job_id: uuid.UUID) -> bool:
    """Cancel cooperatively, never clobbering a terminal state. A queued job
    is cancelled outright; a running one is flagged, and force-cancelled when
    it has run past one batch's worst case (a dead executor never clears its
    own slot). Same contract as translation's."""
    flagged = cast(
        CursorResult[Any],
        session.execute(
            update(CleanupJob)
            .where(CleanupJob.id == job_id, CleanupJob.status.in_(_ACTIVE))
            .values(
                cancel_requested=True,
                status=case(
                    (
                        CleanupJob.status == CleanupJobStatus.QUEUED.value,
                        CleanupJobStatus.CANCELLED.value,
                    ),
                    else_=CleanupJob.status,
                ),
            )
        ),
    )
    if flagged.rowcount != 1:
        return False
    status, started_at, config = session.execute(
        select(CleanupJob.status, CleanupJob.started_at, CleanupJob.config).where(
            CleanupJob.id == job_id
        )
    ).one()
    if status == CleanupJobStatus.RUNNING.value and started_at is not None:
        timeout = config.get("llm_timeout_seconds")
        attempts = config.get("llm_attempts_per_batch")
        per_batch = float(
            timeout if _positive_number(timeout) else DEFAULT_LLM_TIMEOUT_SECONDS
        ) * float(attempts if _positive_int(attempts) else 1)
        bound = per_batch + STALE_RUNNING_GRACE_SECONDS
        session.execute(
            update(CleanupJob)
            .where(
                CleanupJob.id == job_id,
                CleanupJob.status == CleanupJobStatus.RUNNING.value,
                CleanupJob.started_at < func.now() - func.make_interval(0, 0, 0, 0, 0, 0, bound),
            )
            .values(status=CleanupJobStatus.CANCELLED.value, finished_at=func.now())
        )
    return True


def _finish(
    session: Session, job_id: uuid.UUID, *, status: CleanupJobStatus, error: str | None = None
) -> None:
    """Guarded active-to-terminal CAS; a FAILED verdict racing an operator
    cancel resolves to CANCELLED, without the failure text."""
    resolved: Any = status.value
    message: Any = error[:MAX_ERROR_CHARS] if error else None
    if status is CleanupJobStatus.FAILED:
        cancelled = CleanupJob.cancel_requested.is_(True)
        resolved = case((cancelled, CleanupJobStatus.CANCELLED.value), else_=status.value)
        message = case((cancelled, None), else_=message)
    session.execute(
        update(CleanupJob)
        .where(CleanupJob.id == job_id, CleanupJob.status.in_(_ACTIVE))
        .values(status=resolved, error=message, finished_at=func.clock_timestamp())
    )
    session.commit()


def _cancel_flag(session: Session, job_id: uuid.UUID) -> bool:
    # A column select, never the identity-mapped job: under READ COMMITTED
    # each statement sees the latest committed flag.
    return bool(
        session.execute(
            select(CleanupJob.cancel_requested).where(CleanupJob.id == job_id)
        ).scalar_one()
    )


def execute_job(
    session_factory: sessionmaker[Session],
    job_id: uuid.UUID,
    *,
    settings: Settings,
    llm: ChatJsonLLM | None = None,
) -> None:
    """The worker body: claim, propose, persist. Never raises for job
    outcomes; failures land on the row as bounded, honest ``error`` text.
    ``llm`` is an injection seam for tests."""
    with session_factory() as session:
        # GPU sharing (#748): the phase check shares the claim's transaction.
        if not admit_lane(session, settings, POST_SEGMENT):
            logger.info("GPU phase closed; clean-up job %s deferred (stays QUEUED)", job_id)
            return
        job = claim_job(session, job_id)
        if job is None:
            return
        app_row = get_app_settings(session)
        if not cleanup_gates_open(settings, app_row):
            _finish(
                session,
                job_id,
                status=CleanupJobStatus.FAILED,
                error="the language model was disabled after this clean-up was queued",
            )
            return
        try:
            exec_settings, fillers = _settings_from_snapshot(settings, job.config)
            source = load_cleanup_source(session, job.pipeline_run_id)
        except (CleanupJobError, CleanupError) as exc:
            _finish(session, job_id, status=CleanupJobStatus.FAILED, error=str(exc))
            return
        refusal = language_refusal(source.translation.source_language)
        if refusal is not None:
            _finish(session, job_id, status=CleanupJobStatus.FAILED, error=refusal)
            return
        # The bundled local model (#67) is resolved live from the row, never
        # snapshotted, like translation and run assets.
        bundled = llm_bundled_active(app_row, settings)
        if bundled:
            exec_settings = exec_settings.model_copy(
                update={
                    "llm_base_url": settings.llm_bundled_base_url,
                    "llm_model": settings.llm_bundled_model,
                }
            )
        started_at = job.started_at or _db_now(session)
        owned_client: HttpLLMClient | None = None
        client: ChatJsonLLM
        if llm is None:
            # The key is resolved live from the row, so a rotation after
            # enqueue takes effect; the bundled endpoint is keyless and greedy.
            effective_key = "" if bundled else resolve_effective_llm_api_key(app_row, settings)
            try:
                owned_client = HttpLLMClient(
                    exec_settings.llm_base_url,
                    exec_settings.llm_model,
                    effective_key,
                    exec_settings.llm_timeout_seconds,
                    sampling=SamplingProfile() if bundled else None,
                    disable_thinking=exec_settings.llm_disable_thinking,
                )
            except Exception:
                logger.exception("clean-up job %s LLM client init failed", job_id)
                _finish(
                    session,
                    job_id,
                    status=CleanupJobStatus.FAILED,
                    error="LLM endpoint could not be initialized"
                    " (check the LLM endpoint setting or LLM_BASE_URL)",
                )
                return
            client = owned_client
        else:
            client = llm

        try:
            proposals = propose_cleanup(
                cast(ProducerChatJsonLLM, client),
                source.lines,
                hint_words=[*fillers["words"], *fillers["phrases"]],
                kept=fillers["kept"],
                settings=exec_settings,
                should_cancel=lambda: _cancel_flag(session, job_id),
            )
        except CleanupCancelled:
            _finish(session, job_id, status=CleanupJobStatus.CANCELLED)
            return
        except CleanupProducerError as exc:
            _finish(session, job_id, status=CleanupJobStatus.FAILED, error=str(exc))
            return
        except Exception as exc:
            logger.exception("clean-up job %s failed unexpectedly", job_id)
            session.rollback()
            _finish(
                session,
                job_id,
                status=CleanupJobStatus.FAILED,
                error=f"unexpected error ({type(exc).__name__}); see worker logs",
            )
            return
        finally:
            if owned_client is not None:
                owned_client.close()

        try:
            if _cancel_flag(session, job_id):
                _finish(session, job_id, status=CleanupJobStatus.CANCELLED)
                return
            # The race guard compares against the hash of the source this
            # generation was built from, not the enqueue-time hash.
            current = load_translation_source(session, job.pipeline_run_id)
            if translation_source_hash(current) != translation_source_hash(source.translation):
                _finish(
                    session, job_id, status=CleanupJobStatus.FAILED, error=SOURCE_CHANGED_ERROR
                )
                return
            row = record_cleanup(
                session,
                source=source,
                proposals=proposals,
                config={
                    "prompt_version": CLEANUP_PROMPT_VERSION,
                    "filler_list": fillers,
                    **{knob: getattr(exec_settings, knob) for knob in _KNOBS},
                },
                model=exec_settings.llm_model,
                producer=PRODUCER_NAME,
                producer_version=PRODUCER_VERSION,
                started_at=started_at,
                completed_at=_db_now(session),
                idempotency_key=str(job_id),
            )
            session.flush()
            stamped = cast(
                CursorResult[Any],
                session.execute(
                    update(CleanupJob)
                    .where(
                        CleanupJob.id == job_id,
                        CleanupJob.status == CleanupJobStatus.RUNNING.value,
                        CleanupJob.cancel_requested.is_(False),
                    )
                    .values(
                        status=CleanupJobStatus.SUCCEEDED.value,
                        cleanup_id=row.id,
                        finished_at=func.clock_timestamp(),
                    )
                ),
            )
            if stamped.rowcount != 1:
                # Cancel won the race; the generation rolls back with us.
                session.rollback()
                _finish(session, job_id, status=CleanupJobStatus.CANCELLED)
                return
            session.commit()
        except (CleanupError, TranslationError) as exc:
            session.rollback()
            _finish(session, job_id, status=CleanupJobStatus.FAILED, error=str(exc))
        except Exception as exc:
            logger.exception("clean-up job %s failed during finalization", job_id)
            session.rollback()
            _finish(
                session,
                job_id,
                status=CleanupJobStatus.FAILED,
                error=f"unexpected error ({type(exc).__name__}); see worker logs",
            )


def is_active(job: CleanupJob) -> bool:
    """Whether the job is queued or running."""
    return job.status in _ACTIVE


def active_or_last_job(session: Session, pipeline_run_id: uuid.UUID) -> CleanupJob | None:
    """The run's active job if it has one, else its most recent job."""
    active = session.execute(
        select(CleanupJob).where(
            CleanupJob.pipeline_run_id == pipeline_run_id, CleanupJob.status.in_(_ACTIVE)
        )
    ).scalar_one_or_none()
    if active is not None:
        return active
    return session.execute(
        select(CleanupJob)
        .where(CleanupJob.pipeline_run_id == pipeline_run_id)
        .order_by(CleanupJob.created_at.desc(), CleanupJob.id.desc())
        .limit(1)
    ).scalar_one_or_none()


def stale_queued_job_ids(
    session: Session, *, cutoff: datetime, limit: int | None = None
) -> list[uuid.UUID]:
    """Return oldest QUEUED job ids created before ``cutoff``."""
    query = (
        select(CleanupJob.id)
        .where(
            CleanupJob.status == CleanupJobStatus.QUEUED.value,
            CleanupJob.created_at < cutoff,
        )
        .order_by(CleanupJob.created_at, CleanupJob.id)
    )
    if limit is not None:
        query = query.limit(limit)
    return list(session.execute(query).scalars())


__all__ = [
    "ENGLISH",
    "SNAPSHOT_INVALID_ERROR",
    "SOURCE_CHANGED_ERROR",
    "CleanupJobError",
    "active_or_last_job",
    "claim_job",
    "cleanup_gates_open",
    "create_job",
    "execute_job",
    "job_config_snapshot",
    "language_refusal",
    "request_cancel",
    "stale_queued_job_ids",
]
