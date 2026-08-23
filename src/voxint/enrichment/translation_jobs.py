"""Translation job lifecycle (#133): durable state for one generation attempt.

The ``translation_jobs`` row is orchestration state only — queued → running
(guarded claim, so a duplicate Celery delivery no-ops) → succeeded | failed |
cancelled — cloned from ``asset_jobs``. The *result* is an immutable
``run_translations`` row written by ``translations.record_translation`` and
linked via ``translation_id`` in the same transaction that stamps the job
SUCCEEDED. A failed or cancelled job records NO translation and consumes NO
generation.

Two guards beyond the asset-job template:

- **Source-changed race**: the executor freezes one source snapshot at job
  start and recomputes the CURRENT transcript hash immediately before
  persisting; a mismatch against the snapshot's hash finishes the job as
  failed ("the transcript changed"), so an edit made while the model was
  translating can never retire a still-valid generation. (The enqueue-time
  hash stored on the row is provenance; an edit landing between enqueue and
  execution start is simply part of what gets translated.)
- **Same-language guard**: a request to translate into the run's detected
  language is refused at creation (the auto-hook silently skips it first).

Deliberate v1 cuts, mirroring ``asset_jobs``: no automatic retries and no
recovery sweep — cancel is deadline-aware (the LLM batch sequence is bounded
by attempts x timeout per batch), so a crashed RUNNING row can always be
cleared and the one-active-per-(run, language) slot recovered.
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Protocol, cast

from sqlalchemy import CursorResult, case, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from voxint.api.languages import LANGUAGE_NAMES, language_label
from voxint.app_settings import (
    get_app_settings,
    llm_bundled_active,
    resolve_effective_llm_api_key,
    resolve_effective_llm_enabled,
    resolve_effective_llm_endpoint,
)
from voxint.clients.llm import HttpLLMClient, SamplingProfile
from voxint.config import DEFAULT_LLM_TIMEOUT_SECONDS, Settings
from voxint.db.models import (
    AppSettings,
    TranslationJob,
    TranslationJobStatus,
)
from voxint.enrichment.producers.translation_llm import (
    PRODUCER_NAME,
    PRODUCER_VERSION,
    TranslationCancelled,
    TranslationProducerError,
    translate_lines,
)
from voxint.enrichment.producers.translation_llm import (
    ChatJsonLLM as ProducerChatJsonLLM,
)
from voxint.enrichment.translations import (
    TranslationError,
    current_translation,
    load_translation_source,
    record_translation,
    translation_source_hash,
)

logger = logging.getLogger(__name__)

MAX_ERROR_CHARS = 500

# Grace a provably-dead RUNNING job gets past its worst-case batch sequence
# before the operator may force-cancel it. The per-batch bound is
# attempts x timeout; the sequence length is unknowable up front, so the
# deadline uses one batch's worth plus grace — the same "one call" arithmetic
# as asset jobs, applied per batch: a live executor refreshes nothing, but its
# next _finish stamps a terminal state anyway, so force-cancel is only ever
# racing a crashed worker.
STALE_RUNNING_GRACE_SECONDS = 60.0

SOURCE_CHANGED_ERROR = (
    "the transcript changed while it was being translated — translate again"
)


class ChatJsonLLM(Protocol):
    """The only capability the executor needs from a client (injection seam)."""

    def chat_json(self, messages: Any) -> dict[str, object]: ...


class TranslationJobError(Exception):
    """A job cannot be created or started — gates off, bad target, unknown id."""


def translation_gates_open(settings: Settings, row: AppSettings | None) -> bool:
    """Checked at job creation AND again in the worker, so queued work cannot
    outlive a capability shutdown. Translation rides on the configured LLM;
    it has no separate feature flag — a target language plus LLM enablement
    IS the capability."""
    return resolve_effective_llm_enabled(row, settings)


def normalized_language(code: str | None) -> str | None:
    """Lowercased, stripped code or ``None`` — the comparison form for the
    detected-vs-target skip (whisper codes are lowercase; a hand-set env value
    may not be)."""
    if code is None:
        return None
    stripped = code.strip().lower()
    return stripped or None


# snapshot key → the Settings field the executor actually reads (the #40
# snapshot-executes doctrine, verbatim from asset_jobs).
_CONFIG_FIELDS: dict[str, str] = {
    "model": "llm_model",
    "base_url": "llm_base_url",
    "llm_timeout_seconds": "llm_timeout_seconds",
    "llm_attempts_per_batch": "llm_attempts_per_batch",
    "llm_batch_max_segments": "llm_batch_max_segments",
    "llm_batch_max_chars": "llm_batch_max_chars",
}


def job_config_snapshot(settings: Settings) -> dict[str, object]:
    return {key: getattr(settings, field) for key, field in _CONFIG_FIELDS.items()}


def _settings_from_snapshot(settings: Settings, config: dict[str, object]) -> Settings:
    update_fields = {
        field: config[key]
        for key, field in _CONFIG_FIELDS.items()
        # bool is an int subclass — a corrupted snapshot must not smuggle
        # True into a numeric budget field.
        if isinstance(config.get(key), (int, float, str)) and not isinstance(config.get(key), bool)
    }
    return settings.model_copy(update=update_fields) if update_fields else settings


def create_job(
    session: Session,
    *,
    pipeline_run_id: uuid.UUID,
    target_language: str,
    settings: Settings,
) -> tuple[TranslationJob | None, bool]:
    """Validate and insert one QUEUED job (the caller commits, then publishes).

    Returns ``(job, already_active)`` — an active job for the same (run,
    language) is a skip, not an error, mapped from the partial unique index
    itself (check-then-insert would race).
    """
    row = get_app_settings(session)
    if not translation_gates_open(settings, row):
        raise TranslationJobError(
            "translation is off — it needs LLM enablement"
            " (env LLM_ENABLED or the in-UI toggle)"
        )
    target = normalized_language(target_language)
    if target is None or target not in LANGUAGE_NAMES:
        raise TranslationJobError(
            f"unknown target language {target_language!r} — pick one from the"
            " language list"
        )
    try:
        source = load_translation_source(session, pipeline_run_id)
    except TranslationError as exc:
        raise TranslationJobError(str(exc)) from exc
    if normalized_language(source.source_language) == target:
        raise TranslationJobError(
            f"the transcript is already in {language_label(target)} —"
            " nothing to translate"
        )
    # Snapshot the ROW-resolved endpoint (the asset-job contract): an env change
    # between enqueue and execution can't silently redirect the call. The API
    # KEY is never snapshotted — it is resolved live at execution from the row.
    base_url, model = resolve_effective_llm_endpoint(row, settings)
    snapshot = job_config_snapshot(
        settings.model_copy(update={"llm_base_url": base_url, "llm_model": model})
    )
    job = TranslationJob(
        pipeline_run_id=pipeline_run_id,
        target_language=target,
        status=TranslationJobStatus.QUEUED.value,
        config=snapshot,
        source_content_hash=translation_source_hash(source),
    )
    try:
        with session.begin_nested():
            session.add(job)
            session.flush()
    except IntegrityError as exc:
        constraint = getattr(getattr(exc.orig, "diag", None), "constraint_name", None)
        if constraint != "translation_jobs_one_active_per_run_language":
            raise
        return None, True
    return job, False


def claim_job(session: Session, job_id: uuid.UUID) -> TranslationJob | None:
    """queued → running, exactly once (duplicate delivery no-ops). A job whose
    cancel flag is already set is refused: a cancel that lands between enqueue
    and delivery must win even if the status write raced."""
    claimed = cast(
        CursorResult[Any],
        session.execute(
            update(TranslationJob)
            .where(
                TranslationJob.id == job_id,
                TranslationJob.status == TranslationJobStatus.QUEUED.value,
                TranslationJob.cancel_requested.is_(False),
            )
            .values(
                status=TranslationJobStatus.RUNNING.value,
                # DB clock, like created_at — an app-clock value could trip
                # the started_at >= created_at CHECK under clock skew.
                started_at=func.now(),
            )
        ),
    )
    if claimed.rowcount != 1:
        return None
    session.commit()
    return session.get(TranslationJob, job_id)


def request_cancel(session: Session, job_id: uuid.UUID) -> bool:
    """Cancel cooperatively — atomically, never clobbering a terminal state
    (the asset-job contract verbatim). A RUNNING job that provably outlived
    one batch's worst case (attempts x timeout + grace) has no live executor
    refreshing it toward success within that bound — force-cancel clears the
    slot; a still-live executor's own terminal stamp is CAS-guarded either way."""
    flagged = cast(
        CursorResult[Any],
        session.execute(
            update(TranslationJob)
            .where(
                TranslationJob.id == job_id,
                TranslationJob.status.in_(
                    (TranslationJobStatus.QUEUED.value, TranslationJobStatus.RUNNING.value)
                ),
            )
            .values(
                cancel_requested=True,
                status=case(
                    (
                        TranslationJob.status == TranslationJobStatus.QUEUED.value,
                        TranslationJobStatus.CANCELLED.value,
                    ),
                    else_=TranslationJob.status,
                ),
            )
        ),
    )
    if flagged.rowcount != 1:
        return False
    # Column select (not session.get) so the identity map cannot serve a
    # pre-UPDATE snapshot of the row just mutated through Core.
    status, started_at, config = session.execute(
        select(TranslationJob.status, TranslationJob.started_at, TranslationJob.config).where(
            TranslationJob.id == job_id
        )
    ).one()
    if status == TranslationJobStatus.RUNNING.value and started_at is not None:
        timeout = config.get("llm_timeout_seconds")
        attempts = config.get("llm_attempts_per_batch")
        per_batch = float(
            timeout if isinstance(timeout, (int, float)) else DEFAULT_LLM_TIMEOUT_SECONDS
        ) * float(attempts if isinstance(attempts, int) and not isinstance(attempts, bool) else 1)
        bound = per_batch + STALE_RUNNING_GRACE_SECONDS
        # DB clock on BOTH sides (make_interval's 7th positional argument is
        # seconds), matching the claim path's now().
        session.execute(
            update(TranslationJob)
            .where(
                TranslationJob.id == job_id,
                TranslationJob.status == TranslationJobStatus.RUNNING.value,
                TranslationJob.started_at
                < func.now() - func.make_interval(0, 0, 0, 0, 0, 0, bound),
            )
            .values(
                status=TranslationJobStatus.CANCELLED.value,
                finished_at=func.now(),
            )
        )
    return True


def _finish(
    session: Session,
    job_id: uuid.UUID,
    *,
    status: TranslationJobStatus,
    error: str | None = None,
) -> None:
    """Guarded active→terminal CAS — a terminal row is never mutated again,
    and a FAILED verdict racing an operator cancel resolves to CANCELLED."""
    resolved: Any = status.value
    if status is TranslationJobStatus.FAILED:
        resolved = case(
            (TranslationJob.cancel_requested.is_(True), TranslationJobStatus.CANCELLED.value),
            else_=status.value,
        )
    session.execute(
        update(TranslationJob)
        .where(
            TranslationJob.id == job_id,
            TranslationJob.status.in_(
                (TranslationJobStatus.QUEUED.value, TranslationJobStatus.RUNNING.value)
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
    llm: ChatJsonLLM | None = None,
) -> None:
    """The worker body: claim, translate, finalize. Never raises for job
    outcomes — failures land on the row as bounded, honest ``error`` text.
    ``llm`` is an injection seam (tests; the CLI's inline mode)."""
    with session_factory() as session:
        job = claim_job(session, job_id)
        if job is None:
            return
        app_row = get_app_settings(session)
        if not translation_gates_open(settings, app_row):
            _finish(
                session,
                job_id,
                status=TranslationJobStatus.FAILED,
                error="the LLM was disabled after this translation was queued",
            )
            return
        # The snapshot the operator saw at enqueue is the contract — never
        # settings changed since.
        exec_settings = _settings_from_snapshot(settings, job.config)
        # Scoped bundled local model (#67): when active, translation routes to
        # the keyless bundled endpoint like enhancement and run-asset
        # summary/entities do — resolved LIVE from the row, never snapshotted.
        bundled = llm_bundled_active(app_row, settings)
        if bundled:
            exec_settings = exec_settings.model_copy(
                update={
                    "llm_base_url": settings.llm_bundled_base_url,
                    "llm_model": settings.llm_bundled_model,
                }
            )
        try:
            source = load_translation_source(session, job.pipeline_run_id)
        except TranslationError as exc:
            _finish(session, job_id, status=TranslationJobStatus.FAILED, error=str(exc))
            return
        source_label = (
            language_label(source.source_language)
            if source.source_language is not None
            else None
        )
        started_at = job.started_at or datetime.now(tz=UTC)
        owned_client: HttpLLMClient | None = None
        client: ChatJsonLLM
        if llm is None:
            # Resolve the effective key LIVE from the row (the asset-job
            # contract): a key rotated after enqueue takes effect with no
            # restart. The bundled endpoint is keyless and pinned greedy.
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
                # Construction sits OUTSIDE the generation try below, so any
                # escape would strand the job RUNNING forever (no recovery
                # sweep). Closed-vocabulary message; details go to the log.
                logger.exception("translation job %s LLM client init failed", job_id)
                _finish(
                    session,
                    job_id,
                    status=TranslationJobStatus.FAILED,
                    error="LLM endpoint could not be initialized"
                    " (check the LLM endpoint setting or LLM_BASE_URL)",
                )
                return
            client = owned_client
        else:
            client = llm
        def _cancel_pending() -> bool:
            # Column select (never the identity-mapped job object): under READ
            # COMMITTED each statement sees the latest committed flag, so a
            # cancel committed by the API between batches is actually seen.
            return bool(
                session.execute(
                    select(TranslationJob.cancel_requested).where(TranslationJob.id == job_id)
                ).scalar_one()
            )

        try:
            translated = translate_lines(
                cast(ProducerChatJsonLLM, client),
                source.lines,
                source_label=source_label,
                target_label=language_label(job.target_language),
                settings=exec_settings,
                should_cancel=_cancel_pending,
            )
        except TranslationCancelled:
            _finish(session, job_id, status=TranslationJobStatus.CANCELLED)
            return
        # No ``except LLMError`` arm on purpose: unlike the one-call asset
        # producer, translate_lines' retry ladder catches every LLMError per
        # attempt and converts the irreducible case to TranslationProducerError
        # (whose message is classification-only — endpoint response bodies
        # stay in the log, never on the row).
        except (TranslationProducerError, TranslationError) as exc:
            _finish(session, job_id, status=TranslationJobStatus.FAILED, error=str(exc))
            return
        except Exception as exc:
            # Last-resort honesty: an unexpected failure must never leave the
            # job RUNNING forever. Closed-vocabulary error only.
            logger.exception("translation job %s failed unexpectedly", job_id)
            session.rollback()
            _finish(
                session,
                job_id,
                status=TranslationJobStatus.FAILED,
                error=f"unexpected error ({type(exc).__name__}) — see worker logs",
            )
            return
        finally:
            if owned_client is not None:
                owned_client.close()

        # Atomic finalization: the translation row and the job stamp commit
        # together, and only while the row is still RUNNING with no cancel
        # pending. The whole block sits under the same failure umbrella as
        # generation — a DB error must land as an honest FAILED row.
        try:
            # A cancel that raced the LLM calls wins: check the flag before
            # persisting anything.
            if bool(
                session.execute(
                    select(TranslationJob.cancel_requested).where(TranslationJob.id == job_id)
                ).scalar_one()
            ):
                _finish(session, job_id, status=TranslationJobStatus.CANCELLED)
                return
            # Source-changed race guard: the operator may have edited or split
            # the transcript while the model was translating. Reload and
            # compare against the EXECUTED snapshot's hash (``source`` was
            # frozen at job start — an edit landing before the start is simply
            # part of what got translated; the enqueue-time hash on the row is
            # provenance, not this guard's baseline). A mismatch means this
            # generation describes a transcript that no longer exists, so it
            # must NOT persist or supersede the (still-valid) previous one.
            current_source = load_translation_source(session, job.pipeline_run_id)
            if translation_source_hash(current_source) != translation_source_hash(source):
                _finish(
                    session,
                    job_id,
                    status=TranslationJobStatus.FAILED,
                    error=SOURCE_CHANGED_ERROR,
                )
                return
            row = record_translation(
                session,
                source=source,
                target_language=job.target_language,
                translated=translated,
                model=exec_settings.llm_model,
                producer=PRODUCER_NAME,
                producer_version=PRODUCER_VERSION,
                started_at=started_at,
                completed_at=datetime.now(tz=UTC),
            )
            session.flush()
            stamped = cast(
                CursorResult[Any],
                session.execute(
                    update(TranslationJob)
                    .where(
                        TranslationJob.id == job_id,
                        TranslationJob.status == TranslationJobStatus.RUNNING.value,
                        TranslationJob.cancel_requested.is_(False),
                    )
                    .values(
                        status=TranslationJobStatus.SUCCEEDED.value,
                        translation_id=row.id,
                        finished_at=func.now(),
                    )
                ),
            )
            if stamped.rowcount != 1:
                # Cancel won the race; the translation insert rolls back with
                # us and the cooperative flag resolves.
                session.rollback()
                _finish(session, job_id, status=TranslationJobStatus.CANCELLED)
                return
            session.commit()
        except TranslationError as exc:
            session.rollback()
            _finish(session, job_id, status=TranslationJobStatus.FAILED, error=str(exc))
        except Exception as exc:
            logger.exception("translation job %s failed during finalization", job_id)
            session.rollback()
            _finish(
                session,
                job_id,
                status=TranslationJobStatus.FAILED,
                error=f"unexpected error ({type(exc).__name__}) — see worker logs",
            )


def active_or_last_job(session: Session, pipeline_run_id: uuid.UUID) -> TranslationJob | None:
    """The active job if one exists, else the most recent one (any language)."""
    rows = (
        session.execute(
            select(TranslationJob)
            .where(TranslationJob.pipeline_run_id == pipeline_run_id)
            .order_by(TranslationJob.created_at.desc(), TranslationJob.id.desc())
        )
        .scalars()
        .all()
    )
    picked: TranslationJob | None = None
    for row in rows:
        if picked is None:
            picked = row
            continue
        if picked.status not in (
            TranslationJobStatus.QUEUED.value,
            TranslationJobStatus.RUNNING.value,
        ) and row.status in (
            TranslationJobStatus.QUEUED.value,
            TranslationJobStatus.RUNNING.value,
        ):
            picked = row
    return picked


def translation_needed(
    session: Session, pipeline_run_id: uuid.UUID, target_language: str
) -> bool:
    """False when a current generation for the target already matches the
    source (the post-finalize hash-skip: an idempotent re-finalize regenerates
    nothing that is already fresh)."""
    head = current_translation(session, pipeline_run_id, target_language)
    if head is None:
        return True
    source = load_translation_source(session, pipeline_run_id)
    return head.source_content_hash != translation_source_hash(source)


__all__ = [
    "SOURCE_CHANGED_ERROR",
    "TranslationJobError",
    "active_or_last_job",
    "claim_job",
    "create_job",
    "execute_job",
    "job_config_snapshot",
    "normalized_language",
    "request_cancel",
    "translation_gates_open",
    "translation_needed",
]
