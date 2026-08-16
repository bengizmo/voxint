"""Run-asset job lifecycle (#41): durable state for one generation attempt.

The ``run_asset_jobs`` row is orchestration state only — queued → running
(guarded claim, so a duplicate Celery delivery no-ops) → succeeded | failed |
cancelled. The asset *result* is an immutable ``run_enrichment_assets`` row
written by ``run_assets.record_asset`` and linked via ``asset_id`` in the
same transaction that stamps the job SUCCEEDED. A failed or cancelled job
records NO asset and consumes NO generation — one kind failing never blocks
or retires the others (the issue's failure-isolation requirement, held
structurally).

Deliberate v1 cuts, mirroring ``research_jobs``: no automatic retries and no
recovery sweep — but cancel is deadline-aware (one LLM call is the only
legitimate overrun), so a crashed RUNNING row can always be cleared and the
one-active-per-(run, kind) slot recovered.
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Protocol, cast

from sqlalchemy import CursorResult, case, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from voxint.app_settings import (
    get_app_settings,
    resolve_effective_llm_api_key,
    resolve_effective_llm_enabled,
    resolve_effective_llm_endpoint,
)
from voxint.clients.llm import ChatMessage, HttpLLMClient, LLMError
from voxint.config import DEFAULT_LLM_TIMEOUT_SECONDS, Settings
from voxint.db.models import AppSettings, RunAssetJob, RunAssetJobStatus, RunAssetKind
from voxint.enrichment.producers.run_assets_llm import (
    CONFIG_SCHEMA_VERSION,
    PAYLOAD_SCHEMA_VERSION,
    PRODUCER_NAME,
    PRODUCER_VERSION,
    RunAssetProducerError,
    config_snapshot,
    generate_payload,
)
from voxint.enrichment.run_assets import (
    RunAssetError,
    latest_assets,
    load_source,
    record_asset,
    source_content_hash,
)

logger = logging.getLogger(__name__)

MAX_ERROR_CHARS = 500

# Grace a provably-dead RUNNING job gets past its one LLM call before the
# operator may force-cancel it.
STALE_RUNNING_GRACE_SECONDS = 60.0


class ChatJsonLLM(Protocol):
    """The only capability the executor needs from a client (injection seam)."""

    def chat_json(self, messages: "list[ChatMessage] | Any") -> dict[str, object]: ...


class RunAssetJobError(Exception):
    """A job cannot be created or started — gates off, bad target, unknown id."""


def run_asset_gates_open(settings: Settings, row: AppSettings | None) -> bool:
    """Checked at job creation AND again in the worker, so queued work cannot
    outlive a capability shutdown.

    LLM enablement is the effective (row-over-env) value — a UI toggle applies with
    no restart, matching transcript enhancement and ``voxint doctor`` (issue #10).
    Callers pass the ``app_settings`` row they already hold via
    :func:`~voxint.app_settings.get_app_settings`.
    """
    return settings.enrichment_run_assets_enabled and resolve_effective_llm_enabled(
        row, settings
    )


# snapshot key → the Settings field the executor actually reads. The worker
# reconstructs its execution settings from the job's snapshot through this
# map, so a settings change between enqueue and execution never silently
# applies (the #40 snapshot-executes doctrine).
_CONFIG_FIELDS: dict[str, str] = {
    "model": "llm_model",
    "base_url": "llm_base_url",
    "max_input_chars": "run_assets_max_input_chars",
    "llm_timeout_seconds": "llm_timeout_seconds",
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


def create_jobs(
    session: Session,
    *,
    pipeline_run_id: uuid.UUID,
    kinds: tuple[RunAssetKind, ...],
    settings: Settings,
) -> tuple[list[RunAssetJob], list[RunAssetKind]]:
    """Validate and insert QUEUED jobs (the caller commits, then publishes).

    Returns ``(created, already_active)`` — a kind with an active job is
    skipped, not an error, so a "generate all" action degrades per kind
    instead of failing whole. The friendly pre-check is the partial unique
    index itself: each insert runs in a savepoint and an IntegrityError maps
    to the skip (check-then-insert would race).
    """
    if not run_asset_gates_open(settings, get_app_settings(session)):
        raise RunAssetJobError(
            "run assets are disabled — they need ENRICHMENT_RUN_ASSETS_ENABLED"
            " and LLM enablement (env LLM_ENABLED or the in-UI toggle)"
        )
    if not kinds:
        raise RunAssetJobError("no asset kinds requested")
    if len(set(kinds)) != len(kinds):
        raise RunAssetJobError(f"duplicate asset kinds: {[k.value for k in kinds]}")
    try:
        load_source(session, pipeline_run_id)  # validates run + transcript exist
    except RunAssetError as exc:
        raise RunAssetJobError(str(exc)) from exc
    # Snapshot the ROW-resolved endpoint (issue #10): the operator's UI base_url /
    # model are the enqueue contract, so an env change between enqueue and execution
    # can't silently redirect the call (the #40 snapshot-executes doctrine). The API
    # KEY is never snapshotted — it is resolved live at execution from the row.
    base_url, model = resolve_effective_llm_endpoint(get_app_settings(session), settings)
    snapshot = job_config_snapshot(
        settings.model_copy(update={"llm_base_url": base_url, "llm_model": model})
    )
    created: list[RunAssetJob] = []
    already_active: list[RunAssetKind] = []
    for kind in kinds:
        job = RunAssetJob(
            pipeline_run_id=pipeline_run_id,
            asset_kind=kind.value,
            status=RunAssetJobStatus.QUEUED.value,
            config=dict(snapshot),
        )
        try:
            with session.begin_nested():
                session.add(job)
                session.flush()
        except IntegrityError as exc:
            # Only the one-active partial unique index means "skip this kind";
            # any other integrity failure is a real bug and must surface.
            constraint = getattr(getattr(exc.orig, "diag", None), "constraint_name", None)
            if constraint != "run_asset_jobs_one_active_per_run_kind":
                raise
            already_active.append(kind)
            continue
        created.append(job)
    return created, already_active


def claim_job(session: Session, job_id: uuid.UUID) -> RunAssetJob | None:
    """queued → running, exactly once (duplicate delivery no-ops). A job whose
    cancel flag is already set is refused: a cancel that lands between enqueue
    and delivery must win even if the status write raced."""
    claimed = cast(
        CursorResult[Any],
        session.execute(
            update(RunAssetJob)
            .where(
                RunAssetJob.id == job_id,
                RunAssetJob.status == RunAssetJobStatus.QUEUED.value,
                RunAssetJob.cancel_requested.is_(False),
            )
            .values(
                status=RunAssetJobStatus.RUNNING.value,
                # DB clock, like created_at — an app-clock value could trip
                # the started_at >= created_at CHECK under clock skew.
                started_at=func.now(),
            )
        ),
    )
    if claimed.rowcount != 1:
        return None
    session.commit()
    return session.get(RunAssetJob, job_id)


def request_cancel(session: Session, job_id: uuid.UUID) -> bool:
    """Cancel cooperatively — atomically, never clobbering a terminal state.

    One guarded UPDATE sets the flag and resolves QUEUED outright (its
    delivery will fail the claim and no-op); a RUNNING executor re-checks the
    flag after its LLM call returns. A RUNNING job that provably outlived its
    one LLM call (timeout + grace) has no live executor left — cancel it
    outright so a worker crash cannot hold the (run, kind) slot forever. The
    caller commits."""
    flagged = cast(
        CursorResult[Any],
        session.execute(
            update(RunAssetJob)
            .where(
                RunAssetJob.id == job_id,
                RunAssetJob.status.in_(
                    (RunAssetJobStatus.QUEUED.value, RunAssetJobStatus.RUNNING.value)
                ),
            )
            .values(
                cancel_requested=True,
                status=case(
                    (
                        RunAssetJob.status == RunAssetJobStatus.QUEUED.value,
                        RunAssetJobStatus.CANCELLED.value,
                    ),
                    else_=RunAssetJob.status,
                ),
            )
        ),
    )
    if flagged.rowcount != 1:
        return False
    # Column select (not session.get) so the identity map cannot serve a
    # pre-UPDATE snapshot of the row just mutated through Core.
    status, started_at, config = session.execute(
        select(RunAssetJob.status, RunAssetJob.started_at, RunAssetJob.config).where(
            RunAssetJob.id == job_id
        )
    ).one()
    if status == RunAssetJobStatus.RUNNING.value and started_at is not None:
        timeout = config.get("llm_timeout_seconds")
        bound = (
            float(timeout if isinstance(timeout, (int, float)) else DEFAULT_LLM_TIMEOUT_SECONDS)
            + STALE_RUNNING_GRACE_SECONDS
        )
        # DB clock on BOTH sides: started_at was stamped with now() at claim,
        # so an app-clock cutoff would reintroduce exactly the skew the claim
        # path avoided (make_interval's 7th positional argument is seconds).
        session.execute(
            update(RunAssetJob)
            .where(
                RunAssetJob.id == job_id,
                RunAssetJob.status == RunAssetJobStatus.RUNNING.value,
                RunAssetJob.started_at < func.now() - func.make_interval(0, 0, 0, 0, 0, 0, bound),
            )
            .values(
                status=RunAssetJobStatus.CANCELLED.value,
                finished_at=func.now(),
            )
        )
    return True


def _finish(
    session: Session,
    job_id: uuid.UUID,
    *,
    status: RunAssetJobStatus,
    error: str | None = None,
) -> None:
    """Guarded active→terminal CAS — a terminal row is never mutated again
    (a force-cancel that already resolved the job must not be overwritten by
    a late worker failure), and a FAILED verdict racing an operator cancel
    resolves to CANCELLED: the operator asked for exactly that outcome."""
    resolved: Any = status.value
    if status is RunAssetJobStatus.FAILED:
        resolved = case(
            (RunAssetJob.cancel_requested.is_(True), RunAssetJobStatus.CANCELLED.value),
            else_=status.value,
        )
    session.execute(
        update(RunAssetJob)
        .where(
            RunAssetJob.id == job_id,
            RunAssetJob.status.in_(
                (RunAssetJobStatus.QUEUED.value, RunAssetJobStatus.RUNNING.value)
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
    """The worker body: claim, generate, finalize. Never raises for job
    outcomes — failures land on the row as bounded, honest ``error`` text.
    ``llm`` is an injection seam (tests; the CLI's inline mode)."""
    with session_factory() as session:
        job = claim_job(session, job_id)
        if job is None:
            return
        if not run_asset_gates_open(settings, get_app_settings(session)):
            _finish(
                session,
                job_id,
                status=RunAssetJobStatus.FAILED,
                error="run assets were disabled after this job was queued",
            )
            return
        kind = RunAssetKind(job.asset_kind)
        # The snapshot the operator saw at enqueue is the contract — never
        # settings changed since.
        exec_settings = _settings_from_snapshot(settings, job.config)
        try:
            source = load_source(session, job.pipeline_run_id)
        except RunAssetError as exc:
            _finish(session, job_id, status=RunAssetJobStatus.FAILED, error=str(exc))
            return
        started_at = job.started_at or datetime.now(tz=UTC)
        owned_client: HttpLLMClient | None = None
        client: ChatJsonLLM
        if llm is None:
            # Resolve the effective key LIVE from the row (issue #10): a UI-stored
            # key wins over env and is never snapshotted into job.config, so a key
            # rotated after enqueue takes effect on execution with no restart.
            effective_key = resolve_effective_llm_api_key(get_app_settings(session), settings)
            owned_client = HttpLLMClient(
                exec_settings.llm_base_url,
                exec_settings.llm_model,
                effective_key,
                exec_settings.llm_timeout_seconds,
            )
            client = owned_client
        else:
            client = llm
        try:
            payload, truncated = generate_payload(client, kind, source, settings=exec_settings)
        except LLMError as exc:
            # LLMError text can embed endpoint response bodies — persist only
            # the classification; the rest goes to the log (#40 doctrine).
            logger.warning("run-asset job %s LLM failure: %s", job_id, exc)
            _finish(
                session,
                job_id,
                status=RunAssetJobStatus.FAILED,
                error=str(exc).split(":", 1)[0],
            )
            return
        except (RunAssetProducerError, RunAssetError) as exc:
            _finish(session, job_id, status=RunAssetJobStatus.FAILED, error=str(exc))
            return
        except Exception as exc:
            # Last-resort honesty: an unexpected failure must never leave the
            # job RUNNING forever. Closed-vocabulary error only.
            logger.exception("run-asset job %s failed unexpectedly", job_id)
            session.rollback()
            _finish(
                session,
                job_id,
                status=RunAssetJobStatus.FAILED,
                error=f"unexpected error ({type(exc).__name__}) — see worker logs",
            )
            return
        finally:
            if owned_client is not None:
                owned_client.close()

        # Atomic finalization: the asset row and the job stamp commit
        # together, and only while the row is still RUNNING with no cancel
        # pending — a cancel that lands between the check below and this stamp
        # must win (the asset rolls back with the missed stamp), and so must a
        # force-cancel. The whole block — the final cancel read included —
        # sits under the same failure umbrella as generation: a DB error here
        # must land as an honest FAILED row, never a forever-RUNNING job
        # (there is no recovery sweep to save it).
        try:
            # A cancel that raced the LLM call wins: check the flag before
            # persisting anything (the single-call analogue of #40's
            # between-round checks).
            if bool(
                session.execute(
                    select(RunAssetJob.cancel_requested).where(RunAssetJob.id == job_id)
                ).scalar_one()
            ):
                _finish(session, job_id, status=RunAssetJobStatus.CANCELLED)
                return
            asset = record_asset(
                session,
                source=source,
                kind=kind,
                payload=payload,
                payload_schema_version=PAYLOAD_SCHEMA_VERSION,
                producer=PRODUCER_NAME,
                producer_version=PRODUCER_VERSION,
                model=exec_settings.llm_model,
                idempotency_key=(
                    f"{PRODUCER_NAME}:{kind.value}:run:{job.pipeline_run_id}:{job_id}"
                ),
                started_at=started_at,
                completed_at=datetime.now(tz=UTC),
                config=config_snapshot(exec_settings, truncated=truncated),
                config_schema_version=CONFIG_SCHEMA_VERSION,
            )
            session.flush()
            stamped = cast(
                CursorResult[Any],
                session.execute(
                    update(RunAssetJob)
                    .where(
                        RunAssetJob.id == job_id,
                        RunAssetJob.status == RunAssetJobStatus.RUNNING.value,
                        RunAssetJob.cancel_requested.is_(False),
                    )
                    .values(
                        status=RunAssetJobStatus.SUCCEEDED.value,
                        asset_id=asset.id,
                        finished_at=func.now(),
                    )
                ),
            )
            if stamped.rowcount != 1:
                # Cancel won the race; the asset insert rolls back with us and
                # the cooperative flag (if the row is still RUNNING) resolves.
                session.rollback()
                _finish(session, job_id, status=RunAssetJobStatus.CANCELLED)
                return
            session.commit()
        except RunAssetError as exc:
            session.rollback()
            _finish(session, job_id, status=RunAssetJobStatus.FAILED, error=str(exc))
        except Exception as exc:
            logger.exception("run-asset job %s failed during finalization", job_id)
            session.rollback()
            _finish(
                session,
                job_id,
                status=RunAssetJobStatus.FAILED,
                error=f"unexpected error ({type(exc).__name__}) — see worker logs",
            )


def active_or_last_jobs(session: Session, pipeline_run_id: uuid.UUID) -> dict[str, RunAssetJob]:
    """Per kind: the active job if one exists, else the most recent one."""
    rows = (
        session.execute(
            select(RunAssetJob)
            .where(RunAssetJob.pipeline_run_id == pipeline_run_id)
            .order_by(RunAssetJob.created_at.desc(), RunAssetJob.id.desc())
        )
        .scalars()
        .all()
    )
    picked: dict[str, RunAssetJob] = {}
    for row in rows:
        current = picked.get(row.asset_kind)
        if current is None or (
            current.status
            not in (
                RunAssetJobStatus.QUEUED.value,
                RunAssetJobStatus.RUNNING.value,
            )
            and row.status
            in (
                RunAssetJobStatus.QUEUED.value,
                RunAssetJobStatus.RUNNING.value,
            )
        ):
            picked[row.asset_kind] = row
    return picked


def kinds_needing_generation(
    session: Session, pipeline_run_id: uuid.UUID
) -> tuple[RunAssetKind, ...]:
    """Kinds with no current asset, or whose asset no longer matches the
    source (the post-finalize hash-skip: an idempotent re-finalize regenerates
    nothing that is already fresh)."""
    source = load_source(session, pipeline_run_id)
    current_hash = source_content_hash(source)
    current = latest_assets(session, pipeline_run_id)
    return tuple(
        kind
        for kind in RunAssetKind
        if kind.value not in current or current[kind.value].source_content_hash != current_hash
    )


__all__ = [
    "RunAssetJobError",
    "active_or_last_jobs",
    "claim_job",
    "create_jobs",
    "execute_job",
    "job_config_snapshot",
    "kinds_needing_generation",
    "request_cancel",
    "run_asset_gates_open",
    "source_content_hash",
]
