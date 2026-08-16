"""FastAPI application: the review console (queue, workbench, media), health.

Adjudication is post-hoc: only COMPLETED runs appear in the queue, and nothing
here touches the pipeline state machine. Every route except ``/healthz`` sits
behind single-operator basic auth; mutations additionally require the live
claim token, and each rendered form carries a fresh server-issued nonce that
becomes the ledger idempotency key — an htmx retry of the same form is a
harmless replay, while a new submission is a new decision (corrections are
appends; the newest ruling per label wins at read time).
"""

import contextlib
import json
import logging
import re
import secrets
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, BinaryIO, cast
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.templating import Jinja2Templates
from sqlalchemy import exists, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.types import ASGIApp, Receive, Scope, Send

from voxint import __version__
from voxint.adjudication.enrollment import EnrollmentError, enroll_new_speaker
from voxint.adjudication.ledger import ConflictingReplayError, record_decision
from voxint.adjudication.resolver import (
    LabelState,
    Resolution,
    adjudication_queue,
    label_states,
)
from voxint.adjudication.slots import (
    ClaimMismatchError,
    ClaimUnavailableError,
    claim_run,
    release_run,
    verify_claim,
)
from voxint.adjudication.transcript import (
    TranscriptText,
    attributed_transcript,
    parse_transcript_text,
)
from voxint.api.auth import require_operator
from voxint.api.csrf import (
    CSRF_ASSETS_CANCEL,
    CSRF_ASSETS_GENERATE,
    CSRF_CANCEL,
    CSRF_CLAIM,
    CSRF_FETCH,
    CSRF_NOTES,
    CSRF_PROFILE_DECISION,
    CSRF_REQUEUE,
    CSRF_RESEARCH_CANCEL,
    CSRF_RESEARCH_START,
    CSRF_ROSTER_ARCHIVE,
    CSRF_ROSTER_EMBEDDING_DELETE,
    CSRF_ROSTER_MERGE,
    CSRF_ROSTER_RENAME,
    CSRF_ROSTER_RESTORE,
    CSRF_RUN_ARCHIVE,
    CSRF_RUN_MEDIA_DELETE,
    CSRF_RUN_UNARCHIVE,
    CSRF_SETTINGS,
    CSRF_SETUP,
    CSRF_SUBMIT,
    mint_csrf_token,
    verify_csrf_token,
)
from voxint.api.health_probe import probe_services
from voxint.api.playback import (
    MediaResolutionError,
    playback_capability,
    representative_turns,
    resolve_servable_media,
)
from voxint.api.runs_query import (
    Cursor,
    InvalidCursorError,
    ReviewFilter,
    list_runs,
    parse_review_filter,
    parse_search_filters,
    parse_status_filter,
    runs_url,
)
from voxint.api.setup_wizard import (
    STEP_ORDER,
    ScanResult,
    SetupValidationError,
    WizardStep,
    next_step,
    normalize_llm_api_key,
    normalize_llm_base_url,
    normalize_llm_model,
    normalize_media_folders,
    normalize_vocabulary,
    parse_step,
    scan_media_folders,
    validate_llm_enable,
)
from voxint.api.speaker_colors import speaker_palette
from voxint.api.stats_query import (
    DEFAULT_WINDOW,
    collect_stats,
    parse_since,
    render_prometheus,
)
from voxint.app_settings import (
    clear_tutorial_completion,
    complete_onboarding,
    effective_llm_key_source,
    get_app_settings,
    get_or_create,
    is_onboarded,
    mark_tutorial_complete,
    ready_tutorial_run_id,
    resolve_effective_llm_api_key,
    resolve_effective_llm_endpoint,
)
from voxint.config import Settings, get_settings, llm_budget_fits_stage_lease
from voxint.db.models import (
    ClaimField,
    Decision,
    DiarizationTurn,
    EnrichmentCandidate,
    PipelineRun,
    ProfileDecision,
    ResearchJob,
    ResearchJobStatus,
    RunAssetJob,
    RunAssetJobStatus,
    RunAssetKind,
    RunStatus,
    Speaker,
    StageRun,
    TranscriptSegment,
)
from voxint.db.session import build_engine, build_session_factory, session_scope
from voxint.enrichment.asset_jobs import (
    RunAssetJobError,
    active_or_last_jobs,
    create_jobs,
    run_asset_gates_open,
)
from voxint.enrichment.asset_jobs import (
    request_cancel as request_asset_cancel,
)
from voxint.enrichment.drafts import EnrichmentScope
from voxint.enrichment.producers.names import (
    PRODUCER_NAME as NAMES_PRODUCER,
)
from voxint.enrichment.producers.names import (
    NameProducerError,
    run_offline_name_producer,
)
from voxint.enrichment.queries import (
    CandidateState,
    CandidateView,
    candidates_for_run,
    candidates_for_speaker,
    latest_producer_run,
)
from voxint.enrichment.research_jobs import (
    ResearchJobError,
    budget_snapshot,
    create_job,
    request_cancel,
    research_gates_open,
)
from voxint.enrichment.review import ConflictingReplayError as EnrichmentReplayError
from voxint.enrichment.review import StaleCandidateError, record_profile_decision
from voxint.enrichment.run_assets import (
    RunAssetError,
    latest_assets,
    load_source,
    source_content_hash,
)
from voxint.export import (
    MEDIA_TYPES,
    TranscriptFormat,
    render_transcript,
    to_rttm,
    transcript_payload,
)
from voxint.ingest import (
    MissingStageError,
    RunMediaNotDeletableError,
    RunNotArchivableError,
    RunNotCancellableError,
    RunNotFailedError,
    RunNotFoundError,
    UploadConflictError,
    UploadTooLargeError,
    UploadValidationError,
    UrlValidationError,
    archive_run,
    cancel_run,
    delete_run_derived_media,
    requeue_failed_run,
    submit_media_item_if_new,
    submit_upload,
    submit_url,
    unarchive_run,
    unlink_media_paths,
)
from voxint.media.reclaim import run_intermediate_reclaimed_at
from voxint.media.redaction import provenance_host
from voxint.media.serving import (
    MediaGate,
    RangeNotSatisfiableError,
    parse_range,
)
from voxint.pipeline.stages.context import StageDataError, normalized_audio_path
from voxint.pipeline.transitions import InvalidTransitionError, StaleRevisionError
from voxint.speakers.matching import gates_from_settings
from voxint.speakers.roster import (
    RosterError,
    RosterNotFoundError,
    active_speakers,
    archive_speaker,
    delete_embedding,
    merge_speakers,
    rename_speaker,
    restore_speaker,
    roster_overview,
    searchable_speakers,
    voiceprint_bars,
)
from voxint.speakers.roster import is_active as roster_is_active
from voxint.tutorial.steps import (
    STEP_COPY,
    STEP_PAGE,
    WALKTHROUGH_TOTAL,
    TutorialPage,
    TutorialStep,
    parse_tutorial_step,
    walkthrough_number,
)

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"

# Prebuilt frontend island bundles (issue #48). The Vite build stage in the
# Dockerfile copies dist/ here; running from source with no build leaves it
# absent, which the manifest helper and the asset route both tolerate (pages
# still render server-side — progressive enhancement holds even without a build).
_APP_ASSETS_DIR = (Path(__file__).parent / "static" / "app").resolve()
_APP_MANIFEST_PATH = _APP_ASSETS_DIR / ".vite" / "manifest.json"
_APP_ASSET_MEDIA_TYPES = {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".css": "text/css",
    ".map": "application/json",
    ".json": "application/json",
    ".woff2": "font/woff2",
    ".svg": "image/svg+xml",
}
# Vite fingerprints emitted filenames as `<name>-<hash>.<ext>` (base64url hash);
# only those get long-immutable caching (an unhashed name could change in place).
_HASHED_ASSET_RE = re.compile(r"-[A-Za-z0-9_-]{8,}\.[A-Za-z0-9]+$")


def _looks_hashed(name: str) -> bool:
    """True for Vite content-hashed filenames, which are safe to cache forever."""
    return _HASHED_ASSET_RE.search(name) is not None


def _load_asset_manifest() -> dict[str, str]:
    """Map each Vite entry name to its served ``/static/app/...`` URL.

    Parsed once at import. The Vite manifest is keyed by source path
    (``src/main.ts``); we key by the path stem (``main``) so templates request
    islands by their logical entry name. Returns ``{}`` when no build is present
    so ``asset_url`` yields ``None`` and templates emit nothing.
    """
    try:
        raw = _APP_MANIFEST_PATH.read_text()
    except OSError:
        return {}
    try:
        records: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("frontend asset manifest is not valid JSON; islands disabled")
        return {}
    by_entry: dict[str, str] = {}
    for src, record in records.items():
        file = record.get("file") if isinstance(record, dict) else None
        if isinstance(file, str):
            by_entry[Path(src).stem] = "/static/app/" + file
    return by_entry


_APP_ASSET_URLS = _load_asset_manifest()


def asset_url(entry_name: str) -> str | None:
    """Jinja global: entry name -> served URL, or ``None`` when unbuilt."""
    return _APP_ASSET_URLS.get(entry_name)


_MEDIA_CHUNK_BYTES = 256 * 1024
# Bound on the per-run operator notes (issue #36) — hygiene for a TEXT column,
# generous enough for real operator prose.
MAX_OPERATOR_NOTES_CHARS = 10_000
# Slack over the per-file upload cap for multipart framing (boundaries,
# Content-Disposition headers, the submission_id field) so the coarse
# Content-Length gate never rejects a legitimately max-sized file; the exact
# per-file cap is enforced while streaming in submit_upload.
_UPLOAD_ENVELOPE_ALLOWANCE = 1024 * 1024


class _RequestSizeLimitMiddleware:
    """Reject an over-cap ``Content-Length`` before the request body is read.

    FastAPI parses a multipart body (spooling file parts to a temp) *before* a
    route's dependencies run, so a per-route check cannot gate body reception —
    by the time the handler executes, the whole body is already spooled. This
    ASGI middleware inspects only the ``Content-Length`` header and returns 413
    before Starlette consumes the body, so an *honestly-declared* oversized upload
    is rejected early ("reject oversized Content-Length early" is real). The
    authoritative per-file cap is still enforced while streaming in
    ``submit_upload``.

    Residual (NOT covered here): a chunked request with no ``Content-Length``, or
    a transport that permits an understated one, is still fully multipart-spooled
    by Starlette before the streaming cap runs — so pre-body spooling is bounded
    only for honest declared lengths, not universally. A truly-bounded streaming
    multipart parse (and moving Basic auth ahead of body parsing, which a per-route
    ``OperatorDep`` cannot, given FastAPI's dispatch order) is deferred to the
    security slice. For single-operator home-IP hosting that residual is low-risk.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            for name, value in scope["headers"]:
                if name != b"content-length":
                    continue
                try:
                    length = int(value)
                except ValueError:
                    break  # unparseable → let the streaming cap be authoritative
                if length > self._max_bytes:
                    await self._reject(send)
                    return
                break
        await self._app(scope, receive, send)

    @staticmethod
    async def _reject(send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [(b"content-type", b"text/plain; charset=utf-8")],
            }
        )
        await send({"type": "http.response.body", "body": b"request entity too large"})


def create_app(
    settings: Settings | None = None,
    session_factory: Any = None,
) -> FastAPI:
    # No docs/OpenAPI surfaces: the UI is server-rendered, and generated docs
    # would be the only unauthenticated routes besides /healthz.
    app = FastAPI(
        title="Voxint",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    resolved = settings or get_settings()
    app.state.settings = resolved
    # CSRF signing secret: the configured value (persistent, shared by every
    # worker) or a random per-process fallback so the console works with zero
    # config. A per-process secret invalidates open forms on restart and mismatches
    # across workers, so warn when we fall back — see voxint.api.csrf.
    app.state.csrf_secret = resolved.csrf_secret or secrets.token_urlsafe(32)
    if not resolved.csrf_secret:
        logger.warning(
            "csrf_secret is unset; using a random per-process CSRF secret. Open "
            "forms will break on restart and across workers. Set csrf_secret to a "
            "persistent random value to avoid this."
        )
    # Lazy: building the engine at import time would make `/healthz` (and any
    # DB-less test import) depend on a reachable database.
    app.state.session_factory = session_factory
    app.state.media_gate = None
    # Coarse, header-only body-size gate that runs before any route parses the
    # body (see _RequestSizeLimitMiddleware); the streaming per-file cap stays
    # authoritative. Envelope allowance keeps a max-sized file from tripping it.
    app.add_middleware(
        _RequestSizeLimitMiddleware,
        max_bytes=resolved.upload_max_bytes + _UPLOAD_ENVELOPE_ALLOWANCE,
    )
    _register_routes(app)
    return app


def _get_session(request: Request) -> Iterator[Session]:
    # Delegates the commit-on-success / rollback-on-exception body to the single
    # session_scope contextmanager rather than duplicating it: FastAPI resumes a
    # yield-dependency past its `yield` on success and throws the route's
    # exception back in on failure, which is exactly the control flow the `with`
    # needs to drive session_scope's commit/rollback. Mutations that commit
    # before publishing (POST /submit, /runs/{id}/requeue) make the trailing
    # commit here a harmless no-op — nothing is left pending.
    factory = request.app.state.session_factory
    if factory is None:
        factory = build_session_factory(build_engine(request.app.state.settings.database_url))
        request.app.state.session_factory = factory
    with session_scope(factory) as session:
        yield session


def _publish_run(run_id: uuid.UUID) -> None:
    """Enqueue the pipeline for a freshly-committed run (commit-before-publish).

    The Celery/broker import stays out of the module top level so the read path
    — and any DB-less import — never pulls in the broker.

    ``ignore_result=True``: nothing in Voxint consumes ``run_pipeline``'s return
    value (no ``.get()``/``AsyncResult`` anywhere; the status string is only for
    logs), so registering a pending result is pure waste. It also makes a
    dead-broker publish fail *precisely*: with a Redis result backend, plain
    ``.delay()`` on a down broker raises a vague ``RuntimeError`` from the result
    consumer's reconnect loop, whereas ignoring the result surfaces the broker
    connect failure itself as ``kombu.exceptions.OperationalError`` — the exact
    exception ``_publish_or_defer`` catches."""
    from voxint.worker.tasks import run_pipeline

    run_pipeline.apply_async((str(run_id),), ignore_result=True)


def _publish_or_defer(run_id: uuid.UUID) -> bool:
    """Publish the run's task, returning ``False`` (never raising) if the broker
    is unreachable so the request can degrade cleanly.

    Commit-before-publish means the durable QUEUED run already exists, so a Redis
    outage is non-fatal: leaving the run QUEUED lets the beat recovery sweep
    re-enqueue it once the broker returns. Only ``OperationalError`` — kombu's
    wrapper for every transport/connection failure — is swallowed, so a genuine
    bug in the publish path still raises rather than being silently deferred."""
    # celery re-exports kombu's OperationalError as the same class; importing it
    # from celery keeps the broker types under the existing celery.* mypy override
    # and stays lazy so the read path never pulls the broker in.
    from celery.exceptions import OperationalError

    try:
        _publish_run(run_id)
    except OperationalError:
        logger.warning(
            "pipeline enqueue deferred (broker unavailable); run %s stays QUEUED "
            "for the recovery sweep",
            run_id,
            exc_info=True,
        )
        return False
    return True


def _publish_research_job(job_id: uuid.UUID) -> bool:
    """Enqueue a committed research job, returning False on a broker outage.

    Mirrors ``_publish_or_defer``, minus the recovery sweep: research jobs have
    none (v1 — hidden re-execution of a non-deterministic loop is worse than a
    visible stall), so the console shows a deferred job as queued with its age
    and the operator cancels and retries."""
    from celery.exceptions import OperationalError

    from voxint.worker.tasks import research_speaker

    try:
        research_speaker.apply_async((str(job_id),), ignore_result=True)
    except OperationalError:
        logger.warning(
            "research enqueue deferred (broker unavailable); job %s stays QUEUED",
            job_id,
            exc_info=True,
        )
        return False
    return True


def _publish_run_asset_job(job_id: uuid.UUID) -> bool:
    """Enqueue a committed run-asset job, returning False on a broker outage.

    Mirrors ``_publish_research_job``: no recovery sweep (v1), so the console
    shows a deferred job as queued with its age and the operator cancels and
    retries."""
    from celery.exceptions import OperationalError

    from voxint.worker.tasks import generate_run_asset

    try:
        generate_run_asset.apply_async((str(job_id),), ignore_result=True)
    except OperationalError:
        logger.warning(
            "run-asset enqueue deferred (broker unavailable); job %s stays QUEUED",
            job_id,
            exc_info=True,
        )
        return False
    return True


def _run_redirect(run_id: uuid.UUID, *, published: bool) -> RedirectResponse:
    """303 back to the run detail; a deferred publish flags the QUEUED banner.

    The ``enqueue=deferred`` marker is read only as a boolean by the detail
    route — never echoed — and worded historically there, since a refresh or
    bookmark can carry the parameter past the point the sweep already published."""
    suffix = "" if published else "?enqueue=deferred"
    return RedirectResponse(f"/runs/{run_id}{suffix}", status_code=303)


def _get_media_gate(request: Request) -> MediaGate:
    gate = cast(MediaGate | None, request.app.state.media_gate)
    if gate is None:
        settings: Settings = request.app.state.settings
        gate = MediaGate(
            settings.media_root,
            ffprobe_bin=settings.ffprobe_bin,
            timeout_seconds=settings.media_probe_timeout_seconds,
        )
        request.app.state.media_gate = gate
    return gate


SessionDep = Annotated[Session, Depends(_get_session)]
OperatorDep = Annotated[str, Depends(require_operator)]


def require_onboarded(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
) -> None:
    """First-run gate: redirect an un-onboarded operator to the setup wizard.

    Wired as a single router-level dependency on the *protected* router that
    carries every non-exempt route (``/healthz``, the htmx asset, and ``/setup``
    stay on ``app`` and so are structurally exempt — no path matching to keep in
    sync). It depends on ``OperatorDep`` so authentication runs first: an
    unauthenticated request gets a 401 challenge, never a redirect that would leak
    onboarding state. It depends on ``SessionDep`` so FastAPI's per-request
    dependency cache hands the gate the same ``Session`` the route handler uses —
    one connection, not two.

    The onboarding read is cached on ``request.state`` for the life of the request
    only. It is deliberately NOT cached on ``app.state``: the Celery worker can
    flip ``onboarding_complete`` in its own process, so a cross-request cache would
    serve a stale answer. Not onboarded ⇒ ``303`` to ``/setup`` for an ordinary
    navigation, or a ``204`` carrying ``HX-Redirect`` for an htmx request (htmx
    performs the client-side redirect; a 303's body would be swapped into the page
    instead of navigating).
    """
    onboarded = getattr(request.state, "onboarded", None)
    if onboarded is None:
        onboarded = is_onboarded(session)
        request.state.onboarded = onboarded
    if onboarded:
        return
    if request.headers.get("HX-Request"):
        raise HTTPException(status_code=204, headers={"HX-Redirect": "/setup"})
    raise HTTPException(status_code=303, headers={"Location": "/setup"})


templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
# Island bundle lookup for base.html: `asset_url('main')` / `asset_url('tailwind')`
# resolve to the hashed built file, or None (guarded in the template) when unbuilt.
templates.env.globals["asset_url"] = asset_url


def _run_or_404(session: Session, run_id: uuid.UUID) -> PipelineRun:
    run = session.get(PipelineRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no run {run_id}")
    return run


def _reject_if_archived(run: PipelineRun) -> None:
    """409 if the run is soft-archived (issue #5) — archived runs are read-only.

    Filtering hides archived runs from ``/runs`` and the review queue, but a
    stale tab or a hand-crafted POST could still drive one live (requeue) or
    claim it. Refuse those mutations until the operator un-archives it, so
    visibility and mutability stay aligned."""
    if run.archived_at is not None:
        raise HTTPException(
            status_code=409,
            detail="run is archived; un-archive it before requeue/claim",
        )


def _media_delete_banner(request: Request) -> dict[str, int] | None:
    """Parse the post-delete PRG counters into a banner dict, or None.

    A derived-media deletion (issue #5) redirects back with ``?media=deleted``
    and non-negative ``files``/``missing``/``failed`` counts. They are read as
    bare ints and never echoed as text; a malformed/absent value yields no
    banner (a refresh or bookmark can carry the parameter past its meaning)."""
    if request.query_params.get("media") != "deleted":
        return None

    def _count(name: str) -> int:
        raw = request.query_params.get(name, "0")
        return int(raw) if raw.isdigit() else 0

    return {
        "files": _count("files"),
        "missing": _count("missing"),
        "failed": _count("failed"),
    }


def _require_csrf(request: Request, action: str, token: str | None) -> None:
    """403 unless ``token`` is a valid CSRF token for ``action`` — call before any
    state change. A missing token and a mis-signed one BOTH 403 (the field is
    Optional, so FastAPI never turns an absent token into a 422), giving a forged
    cross-site POST one uniform refusal before the DB is touched."""
    if not verify_csrf_token(request.app.state.csrf_secret, action, token):
        raise HTTPException(status_code=403, detail="invalid or missing CSRF token")


def _label_previews(
    session: Session, run_id: uuid.UUID, states: list[LabelState], limit: int
) -> dict[str, list[TranscriptSegment]]:
    previews: dict[str, list[TranscriptSegment]] = {}
    for state in states:
        previews[state.label] = list(
            session.execute(
                select(TranscriptSegment)
                .where(
                    TranscriptSegment.pipeline_run_id == run_id,
                    TranscriptSegment.diarization_label == state.label,
                )
                .order_by(TranscriptSegment.segment_index)
                .limit(limit)
            ).scalars()
        )
    return previews


def _run_label_universe(session: Session, run_id: uuid.UUID) -> set[str]:
    """Every diarization label present in a run, from BOTH its diarization turns
    and its transcript segments.

    A transcript segment may carry a label with no turn (the supported degenerate
    case the resolver's turn-derived ``label_states`` does not enumerate), and a
    turn's label may have no segment; the union covers both. This is the ONE
    canonical universe the per-speaker palette (#50) is built from, so the
    transcript page, its JS-off fallback, and the workbench cards color a given
    label identically. Two cheap indexed ``DISTINCT`` queries — deliberately not
    ``label_states`` (which resolves turn stats, proposals, decisions, and merges)."""
    turn_labels = session.execute(
        select(DiarizationTurn.label)
        .where(DiarizationTurn.pipeline_run_id == run_id)
        .distinct()
    ).scalars()
    segment_labels = session.execute(
        select(TranscriptSegment.diarization_label)
        .where(TranscriptSegment.pipeline_run_id == run_id)
        .distinct()
    ).scalars()
    return {*turn_labels, *(label for label in segment_labels if label is not None)}


# Rendering precedence inside one (target, value) suggestion group: a human
# decision is history and outranks a fresh proposed duplicate from a rerun
# (decided candidates are terminal and never superseded), which outranks
# superseded leftovers.
_HINT_STATE_PRECEDENCE = {
    CandidateState.ACCEPTED: 0,
    CandidateState.REJECTED: 1,
    CandidateState.PROPOSED: 2,
    CandidateState.SUPERSEDED: 3,
}


def _name_suggestions(
    session: Session, run_id: uuid.UUID
) -> tuple[list[CandidateView], dict[str, list[CandidateView]]]:
    """Representative NAME suggestions for the workbench: run-level + per-label.

    Each (target, casefolded value) group renders one representative so rerun
    duplicates beside decided history are never presented as new suggestions.
    """
    views = [
        view
        for view in candidates_for_run(session, run_id)
        if view.candidate.field == ClaimField.NAME.value
    ]
    groups: dict[tuple[str | None, str], CandidateView] = {}
    for view in views:
        key = (view.candidate.diarization_label, view.candidate.value.casefold())
        current = groups.get(key)
        if (
            current is None
            or _HINT_STATE_PRECEDENCE[view.state] < _HINT_STATE_PRECEDENCE[current.state]
            or (
                # Equal state (e.g. names.offline and names.llm both proposed
                # for the same value): the stronger-scored claim represents.
                _HINT_STATE_PRECEDENCE[view.state] == _HINT_STATE_PRECEDENCE[current.state]
                and (view.candidate.score or 0.0) > (current.candidate.score or 0.0)
            )
        ):
            groups[key] = view

    def _order(view: CandidateView) -> tuple[int, float, str]:
        return (
            _HINT_STATE_PRECEDENCE[view.state],
            -(view.candidate.score or 0.0),
            view.candidate.value.casefold(),
        )

    run_level = sorted((view for (label, _), view in groups.items() if label is None), key=_order)
    per_label: dict[str, list[CandidateView]] = {}
    for (label, _), view in groups.items():
        if label is not None:
            per_label.setdefault(label, []).append(view)
    for label_views in per_label.values():
        label_views.sort(key=_order)
    return run_level, per_label


def _workbench_context(
    request: Request,
    session: Session,
    run: PipelineRun,
    token: uuid.UUID | None,
) -> dict[str, Any]:
    settings: Settings = request.app.state.settings
    states = label_states(session, run.id)
    # Assignable identities only — merged and archived speakers are curated out
    # of the roster and must not attract new decisions.
    speakers = active_speakers(session)
    name_hints_run, name_hints_labels = _name_suggestions(session, run.id)
    # Per-turn playback (issue #49) + fail-closed seek gating (issue #55). The
    # workbench-player island (mounted OUTSIDE #labels) reads `capability` to
    # enable/disable the server-rendered, htmx-swapped seek buttons; the buttons
    # themselves carry the representative-turn timings for "preview this speaker".
    capability = playback_capability(session, run, settings, _get_media_gate(request))
    return {
        "name_hints_run": name_hints_run,
        "name_hints_labels": name_hints_labels,
        "names_enabled": settings.enrichment_names_enabled,
        "names_last_run": latest_producer_run(session, NAMES_PRODUCER, EnrichmentScope.run(run.id)),
        # An accepted per-label suggestion prefills the Enroll input (editable,
        # never auto-submitted) — the one-click path from hint to enrollment.
        "enroll_prefill": {
            label: next(
                (view.candidate.value for view in views if view.state is CandidateState.ACCEPTED),
                "",
            )
            for label, views in name_hints_labels.items()
        },
        "request": request,
        "run": run,
        "states": states,
        # Per-speaker identity color (issue #50): one canonical map keyed on the
        # raw label, derived from the SAME run-label universe the transcript page
        # uses (turns and segments), so a label's color never drifts between the
        # two surfaces.
        "palette": speaker_palette(_run_label_universe(session, run.id)),
        "previews": _label_previews(session, run.id, states, settings.review_preview_segments),
        # Island props for the workbench-player (mounted OUTSIDE #labels). It owns
        # the <audio>, the speed control, the visible capability banner, and the
        # document-delegated enabling of the server-rendered seek buttons.
        "workbench_props": {
            "runId": str(run.id),
            "mediaUrl": f"/media/{run.id}",
            "capability": capability.to_props(),
        },
        # Per-label representative turn (start, end) for the "preview this speaker"
        # button — longest clean (non-overlap) DiarizationTurn, fallback longest.
        "representative_turns": representative_turns(session, run.id),
        "speakers": speakers,
        "token": token,
        "resolution": Resolution,
        "nonce": lambda: uuid.uuid4().hex,
        # CSRF token for the (re)claim form shown when the run is not claimed here.
        "csrf_claim": mint_csrf_token(request.app.state.csrf_secret, CSRF_CLAIM),
    }


def _setup_context(
    request: Request,
    session: Session,
    step: WizardStep,
    **overrides: Any,
) -> dict[str, Any]:
    """Template context for a setup-wizard step.

    Read-only: it uses ``get_app_settings`` (never ``get_or_create``) so rendering a
    GET can't create the app_settings row. Fields prefill from the saved row layered
    over env defaults, matching how a run would resolve them; a POST that fails
    validation passes ``error=`` plus the raw submitted text via ``overrides`` so the
    operator's in-progress input survives the re-render.
    """
    settings: Settings = request.app.state.settings
    row = get_app_settings(session)
    media_folders = list(row.media_folders) if row and row.media_folders else []
    vocabulary = list(row.vocabulary) if row and row.vocabulary else []
    _llm_base_url, _llm_model = resolve_effective_llm_endpoint(row, settings)
    context: dict[str, Any] = {
        "request": request,
        "step": step,
        "steps": STEP_ORDER,
        "step_index": STEP_ORDER.index(step),
        "next_step": next_step(step),
        "csrf_setup": mint_csrf_token(request.app.state.csrf_secret, CSRF_SETUP),
        "media_root": str(settings.media_root),
        "media_folders": media_folders,
        # Newline-joined text prefills the folder/vocab textareas on a plain GET.
        "media_folders_text": "\n".join(media_folders),
        "vocabulary": vocabulary,
        "vocabulary_text": "\n".join(vocabulary),
        # LLM step: the row's enablement over env, its (non-secret) effective
        # endpoint, and whether an EFFECTIVE key (UI-stored row value winning over
        # env) is present and where it comes from — never the key value itself.
        "llm_enabled": bool(row.llm_enabled) if row is not None else settings.llm_enabled,
        "llm_base_url": _llm_base_url,
        "llm_model": _llm_model,
        "llm_key_present": bool(resolve_effective_llm_api_key(row, settings)),
        "llm_key_source": effective_llm_key_source(row, settings),
        "llm_budget_ok": llm_budget_fits_stage_lease(settings),
        # The finish step launches the tutorial iff it has been seeded; otherwise
        # it prints the `voxint tutorial seed` note and does a plain /review finish.
        "tutorial_available": ready_tutorial_run_id(session) is not None,
        "active_nav": "setup",
        "error": None,
    }
    context.update(overrides)
    return context


def _persist_llm_settings(
    session: Session,
    settings: Settings,
    *,
    enabled: bool,
    raw_base_url: str,
    raw_model: str,
    raw_key: str,
    remove_key: bool,
) -> str | None:
    """Apply the LLM settings from a form submission as ONE deliberate mutation.

    Shared by ``POST /setup/llm`` and ``POST /settings/llm``. API sessions commit on
    every successful response (including error re-renders), so this computes a
    *candidate* state, validates it, then performs a single mutation whose outcome is
    fully defined for every path:

    * **Pure format errors** (malformed URL/model, or the contradictory
      remove+replacement combination) raise :class:`SetupValidationError` *before*
      ``get_or_create`` — nothing is created or mutated, a prior valid config stays
      intact. The caller re-renders with the fixed message.
    * **Validation failure** (enable requested but no effective key / budget doesn't
      fit) still persists the valid non-secret overrides and the valid candidate key
      (a good key the operator typed is not thrown away) but forces
      ``llm_enabled=False`` and returns the fixed message.
    * **Success** persists candidate key + overrides + the requested ``llm_enabled``
      and returns ``None``.

    The candidate key is: NULL when ``remove_key`` (revert to env), the new value on a
    non-blank submission, else the existing row value (blank password = no change —
    it is never prefilled). The key is a credential: it is never rendered, and the
    returned message is a fixed string that never interpolates it.
    """
    base_url = normalize_llm_base_url(raw_base_url)
    model = normalize_llm_model(raw_model)
    new_key = normalize_llm_api_key(raw_key)
    if remove_key and new_key is not None:
        # Contradictory: the operator both typed a replacement and asked to remove.
        # Reject as a format error so neither intent is silently applied.
        raise SetupValidationError(
            "Choose either a new LLM API key or “remove saved key”, not both."
        )
    row = get_or_create(session, llm_enabled_default=settings.llm_enabled)
    if remove_key:
        candidate_key: str | None = None
    elif new_key is not None:
        candidate_key = new_key
    else:
        candidate_key = row.llm_api_key
    # Effective key from the CANDIDATE (row-wins-over-env), matching how a run/job
    # will resolve it post-save, so the enable guard reflects the saved state.
    effective_key = (candidate_key or "").strip() or settings.llm_api_key.strip()
    error: str | None = None
    if enabled:
        try:
            validate_llm_enable(effective_key, settings)
        except SetupValidationError as exc:
            error = str(exc)
    # Single deliberate mutation. On a validation failure we fail closed
    # (llm_enabled=False) but still keep the valid overrides + candidate key.
    row.llm_base_url = base_url
    row.llm_model = model
    row.llm_api_key = candidate_key
    row.llm_enabled = enabled and error is None
    return error


def _tutorial_banner(
    request: Request,
    session: Session,
    *,
    page: TutorialPage,
    run_id: uuid.UUID | None = None,
    token: uuid.UUID | None = None,
) -> dict[str, Any] | None:
    """Resolve the guided-tutorial banner context for a page, or ``None``.

    Renders nothing unless ALL hold: the ``?tutorial=`` value parses to a step
    (an absent/unknown value is a quiet no-banner, never a 422); that step's bound
    page (:data:`STEP_PAGE`) is THIS page; a tutorial run is configured AND still
    present (``ready_tutorial_run_id``); and — for the run-scoped pages — the
    route's ``run_id`` is exactly that tutorial run. So a ``?tutorial=`` spoofed
    onto any other run, or onto the wrong page, shows nothing. Read-only: it never
    creates the ``app_settings`` row and never mutates.

    The returned dict is a flat bag the banner partial reads; each step populates
    only the action fields it needs (a next-link, a claim form, or the export +
    finish controls). The adjudicate→export next-link carries the verified claim
    ``token`` so the workbench stays writable; the export link never does.
    """
    step = parse_tutorial_step(request.query_params.get("tutorial"))
    if step is None or STEP_PAGE[step] is not page:
        return None
    tutorial_run_id = ready_tutorial_run_id(session)
    if tutorial_run_id is None:
        return None
    # Run-scoped pages must be showing THE tutorial run; the queue page carries no
    # run_id and only needs the tutorial run to exist (checked above).
    if page in (TutorialPage.RUN_DETAIL, TutorialPage.WORKBENCH) and run_id != tutorial_run_id:
        return None

    copy = STEP_COPY[step]
    secret = request.app.state.csrf_secret
    banner: dict[str, Any] = {
        "step": step.value,
        "n": walkthrough_number(step),
        "total": WALKTHROUGH_TOTAL,
        "title": copy.title,
        "body": copy.body,
        "next_href": None,
        "next_label": None,
        "claim_run_id": None,
        "claim_label": "Claim the tutorial run →",
        "csrf_claim": None,
        "export_href": None,
        "csrf_settings": None,
    }
    if step is TutorialStep.RUN:
        banner["next_href"] = "/review?tutorial=review"
        banner["next_label"] = "Open the review console →"
    elif step is TutorialStep.REVIEW:
        banner["claim_run_id"] = tutorial_run_id
        banner["csrf_claim"] = mint_csrf_token(secret, CSRF_CLAIM)
    elif step is TutorialStep.ADJUDICATE:
        if token is not None:
            banner["next_href"] = f"/review/{tutorial_run_id}?token={token}&tutorial=export"
            banner["next_label"] = "I've attributed the voices →"
        else:
            # No live claim on this tab — offer to (re)claim and continue rather
            # than a dead next-link that would land on a read-only workbench.
            banner["claim_run_id"] = tutorial_run_id
            banner["csrf_claim"] = mint_csrf_token(secret, CSRF_CLAIM)
    elif step is TutorialStep.EXPORT:
        # Plaintext export opens in a new tab; the claim token is deliberately NOT
        # placed in its URL. Finishing is an explicit CSRF-guarded POST.
        banner["export_href"] = f"/review/{tutorial_run_id}/export.txt"
        banner["csrf_settings"] = mint_csrf_token(secret, CSRF_SETTINGS)
    return banner


def _labels_response(
    request: Request,
    session: Session,
    run: PipelineRun,
    token: uuid.UUID,
) -> Response:
    """Post-mutation response: htmx gets the refreshed label list, a plain
    form POST gets a redirect back to the workbench."""
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse(
            request,
            "fragments/labels.html",
            _workbench_context(request, session, run, token),
        )
    return RedirectResponse(f"/review/{run.id}?token={token}", status_code=303)


# The claim fields the web-research review surface serves; NAME stays on the
# workbench's dedicated suggestion flow.
_PROFILE_FIELDS = (ClaimField.BIO.value, ClaimField.AFFILIATION.value, ClaimField.LINK.value)
_ACTIVE_JOB_STATUSES = (ResearchJobStatus.QUEUED.value, ResearchJobStatus.RUNNING.value)


def _research_state(
    session: Session, settings: Settings, speaker: Speaker, error: str | None = None
) -> dict[str, Any]:
    """One speaker's research block: latest job, budgets, reviewable drafts."""
    job = session.execute(
        select(ResearchJob)
        .where(ResearchJob.speaker_id == speaker.id)
        .order_by(ResearchJob.created_at.desc(), ResearchJob.id.desc())
        .limit(1)
    ).scalar_one_or_none()
    views = [
        view
        for view in candidates_for_speaker(session, speaker.id)
        if view.candidate.field in _PROFILE_FIELDS
    ]
    return {
        "speaker": speaker,
        "job": job,
        "job_active": job is not None and job.status in _ACTIVE_JOB_STATUSES,
        "gates_open": research_gates_open(settings, get_app_settings(session)),
        "budget": budget_snapshot(settings),
        "proposed": [v for v in views if v.state is CandidateState.PROPOSED],
        "decided_count": sum(
            1 for v in views if v.state in (CandidateState.ACCEPTED, CandidateState.REJECTED)
        ),
        "error": error,
    }


def _research_csrf(request: Request) -> dict[str, Any]:
    secret = request.app.state.csrf_secret
    return {
        "csrf_research_start": mint_csrf_token(secret, CSRF_RESEARCH_START),
        "csrf_research_cancel": mint_csrf_token(secret, CSRF_RESEARCH_CANCEL),
        "csrf_profile_decision": mint_csrf_token(secret, CSRF_PROFILE_DECISION),
        "nonce": lambda: uuid.uuid4().hex,
    }


def _research_response(
    request: Request, session: Session, speaker: Speaker, error: str | None = None
) -> Response:
    """The per-speaker research fragment — the polling target and every
    research mutation's response."""
    return templates.TemplateResponse(
        request,
        "fragments/research.html",
        {
            "request": request,
            "research": _research_state(session, request.app.state.settings, speaker, error),
            **_research_csrf(request),
        },
    )


_RUN_ASSET_KINDS = tuple(kind.value for kind in RunAssetKind)
# Asset jobs have their own enum — do not borrow the research tuple just
# because the string values coincide today.
_ASSET_ACTIVE_STATUSES = (
    RunAssetJobStatus.QUEUED.value,
    RunAssetJobStatus.RUNNING.value,
)
_ASSET_KIND_TITLES = {
    "summary": "Summary",
    "topics": "Topics",
    "entity_mentions": "Entity mentions",
}


def _run_assets_state(
    session: Session, settings: Settings, run_id: uuid.UUID, error: str | None = None
) -> dict[str, Any]:
    """The run's asset block: per kind, the current asset (with staleness
    against the freshly recomputed source hash) and the active/last job."""
    assets = latest_assets(session, run_id)
    jobs = active_or_last_jobs(session, run_id)
    current_hash: str | None = None
    source_problem: str | None = None
    try:
        current_hash = source_content_hash(load_source(session, run_id))
    except RunAssetError as exc:
        source_problem = str(exc)
    kinds = []
    any_active = False
    for kind in _RUN_ASSET_KINDS:
        asset = assets.get(kind)
        job = jobs.get(kind)
        active = job is not None and job.status in _ASSET_ACTIVE_STATUSES
        any_active = any_active or active
        kinds.append(
            {
                "kind": kind,
                "title": _ASSET_KIND_TITLES[kind],
                "asset": asset,
                "stale": (
                    asset is not None
                    and current_hash is not None
                    and asset.source_content_hash != current_hash
                ),
                "job": job,
                "job_active": active,
            }
        )
    return {
        "run_id": run_id,
        "kinds": kinds,
        "any_active": any_active,
        "gates_open": run_asset_gates_open(settings, get_app_settings(session)),
        "source_problem": source_problem,
        "error": error,
    }


def _run_assets_response(
    request: Request, session: Session, run_id: uuid.UUID, error: str | None = None
) -> Response:
    """The per-run assets fragment — the polling target and every asset
    mutation's response."""
    secret = request.app.state.csrf_secret
    return templates.TemplateResponse(
        request,
        "fragments/run_assets.html",
        {
            "request": request,
            "assets": _run_assets_state(
                session, request.app.state.settings, run_id, error
            ),
            "csrf_assets_generate": mint_csrf_token(secret, CSRF_ASSETS_GENERATE),
            "csrf_assets_cancel": mint_csrf_token(secret, CSRF_ASSETS_CANCEL),
        },
    )


def _roster_context(request: Request, session: Session, error: str | None = None) -> dict[str, Any]:
    """Template context for the roster page and its htmx fragment."""
    overview = roster_overview(session)
    secret = request.app.state.csrf_secret
    settings: Settings = request.app.state.settings
    return {
        "request": request,
        "overview": overview,
        "voiceprints": {
            entry.speaker.id: voiceprint_bars(entry.embeddings) for entry in overview.active
        },
        "roster_error": error,
        "active_nav": "speakers",
        "csrf_rename": mint_csrf_token(secret, CSRF_ROSTER_RENAME),
        "csrf_merge": mint_csrf_token(secret, CSRF_ROSTER_MERGE),
        "csrf_archive": mint_csrf_token(secret, CSRF_ROSTER_ARCHIVE),
        "csrf_restore": mint_csrf_token(secret, CSRF_ROSTER_RESTORE),
        "csrf_embedding_delete": mint_csrf_token(secret, CSRF_ROSTER_EMBEDDING_DELETE),
        "research_by_speaker": {
            entry.speaker.id: _research_state(session, settings, entry.speaker)
            for entry in overview.active
        },
        **_research_csrf(request),
    }


def _roster_response(request: Request, session: Session, error: str | None = None) -> Response:
    """Post-mutation response, mirroring ``_labels_response``: htmx gets the
    refreshed roster fragment (operator errors rendered inline), a plain form
    POST gets a 303 back to the page — or the full page when it carries an
    error to show. CSRF/auth failures never come here; they stay real 403s."""
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse(
            request, "fragments/roster.html", _roster_context(request, session, error)
        )
    if error is not None:
        return templates.TemplateResponse(
            request, "speakers.html", _roster_context(request, session, error)
        )
    return RedirectResponse("/speakers", status_code=303)


def _register_routes(app: FastAPI) -> None:
    # Two registrars: `app` carries the onboarding-gate-EXEMPT routes (liveness,
    # the htmx asset, and the setup wizard); `protected` carries everything else
    # behind one router-level gate (require_onboarded). Exemption is structural —
    # a route is exempt iff it is registered on `app` rather than `protected`, so
    # there is no path allow-list to keep in sync. New console routes should
    # default to `protected` (the route-inventory test guards against a slip).
    protected = APIRouter(dependencies=[Depends(require_onboarded)])

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    # ---- First-run setup wizard (issue #3) -------------------------------------
    # Every wizard route is registered on `app`, NOT `protected`, so the onboarding
    # gate exempts it: an un-onboarded operator must be able to reach the page the
    # gate redirects them to. Auth still applies (OperatorDep) — only /healthz is
    # unauthenticated. Each POST verifies CSRF_SETUP before any write. The exact
    # paths below are enumerated in the route-inventory test's exempt allowlist
    # (deliberately NOT a blanket /setup prefix, so an accidental ungated route
    # still fails that guard).

    @app.get("/setup", include_in_schema=False)
    def setup(request: Request, operator: OperatorDep, session: SessionDep) -> Response:
        step = parse_step(request.query_params.get("step"))
        context = _setup_context(request, session, step)
        if step is WizardStep.SERVICES:
            # Probe only on the services step — a few-second network op we don't want
            # to pay on every wizard GET. Best-effort; probe_services never raises.
            settings: Settings = request.app.state.settings
            context["services"] = probe_services(settings)
        return templates.TemplateResponse(request, "setup.html", context)

    def _setup_redirect(step: WizardStep) -> RedirectResponse:
        return RedirectResponse(f"/setup?step={step.value}", status_code=303)

    @app.post("/setup/media", include_in_schema=False)
    def setup_media(
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        media_folders: Annotated[str, Form()] = "",
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> Response:
        _require_csrf(request, CSRF_SETUP, csrf_token)
        settings: Settings = request.app.state.settings
        try:
            folders = normalize_media_folders(media_folders.splitlines(), settings.media_root)
        except SetupValidationError as exc:
            # Validation runs BEFORE any get_or_create, so a rejected save writes
            # nothing; re-render the step with the message and the raw input intact.
            return templates.TemplateResponse(
                request,
                "setup.html",
                _setup_context(
                    request,
                    session,
                    WizardStep.MEDIA,
                    error=str(exc),
                    media_folders_text=media_folders,
                ),
            )
        row = get_or_create(session, llm_enabled_default=settings.llm_enabled)
        row.media_folders = folders
        # Back to the media step (not the next one) so the optional "scan for
        # existing media" stays discoverable; the template offers a Continue link.
        return _setup_redirect(WizardStep.MEDIA)

    def _scan_response(request: Request, session: Session, result: ScanResult) -> Response:
        """htmx → the scan preview/result fragment; a plain POST → back to the step.

        The scan feature is htmx-driven (a fragment swapped into the media step);
        without htmx it degrades to a redirect and the operator submits media
        individually — the scan is an optional convenience, never the only path.
        """
        if request.headers.get("HX-Request"):
            return templates.TemplateResponse(
                request,
                "fragments/setup_scan.html",
                {
                    "request": request,
                    "result": result,
                    "csrf_setup": mint_csrf_token(request.app.state.csrf_secret, CSRF_SETUP),
                },
            )
        return _setup_redirect(WizardStep.MEDIA)

    @app.post("/setup/scan", include_in_schema=False)
    def setup_scan(
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> Response:
        _require_csrf(request, CSRF_SETUP, csrf_token)
        settings: Settings = request.app.state.settings
        row = get_app_settings(session)  # read-only: previewing never creates the row
        folders = list(row.media_folders) if row and row.media_folders else []
        result = scan_media_folders(session, settings.media_root, folders, settings)
        return _scan_response(request, session, result)

    @app.post("/setup/scan/confirm", include_in_schema=False)
    def setup_scan_confirm(
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> Response:
        _require_csrf(request, CSRF_SETUP, csrf_token)
        settings: Settings = request.app.state.settings
        row = get_app_settings(session)
        folders = list(row.media_folders) if row and row.media_folders else []
        # Re-scan fresh rather than trusting any client-supplied path list: the
        # preview is advisory and the filesystem may have changed since it rendered.
        result = scan_media_folders(session, settings.media_root, folders, settings)
        # submit_media_item_if_new is race-safe and returns None for an already-known
        # path, so a double-clicked confirm or a concurrent one cannot duplicate runs.
        run_ids = [
            run.id
            for path in result.candidates
            if (run := submit_media_item_if_new(session, path)) is not None
        ]
        # Commit the whole batch ONCE (commit-before-publish); if the commit fails,
        # nothing is published and no partial state escapes.
        session.commit()
        # After the durable QUEUED rows exist, publish each. A broker outage leaves
        # them QUEUED for the recovery sweep — never roll back a published batch.
        published = sum(_publish_or_defer(run_id) for run_id in run_ids)
        confirmed = ScanResult(
            candidates=[],
            inspected=result.inspected,
            hit_entry_cap=result.hit_entry_cap,
            hit_file_cap=result.hit_file_cap,
            root_missing=result.root_missing,
        )
        if request.headers.get("HX-Request"):
            return templates.TemplateResponse(
                request,
                "fragments/setup_scan.html",
                {
                    "request": request,
                    "result": confirmed,
                    "queued": len(run_ids),
                    "published": published,
                    "csrf_setup": mint_csrf_token(request.app.state.csrf_secret, CSRF_SETUP),
                },
            )
        return _setup_redirect(WizardStep.MEDIA)

    @app.post("/setup/vocabulary", include_in_schema=False)
    def setup_vocabulary(
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        vocabulary: Annotated[str, Form()] = "",
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> Response:
        _require_csrf(request, CSRF_SETUP, csrf_token)
        settings: Settings = request.app.state.settings
        try:
            terms = normalize_vocabulary(vocabulary)
        except SetupValidationError as exc:
            return templates.TemplateResponse(
                request,
                "setup.html",
                _setup_context(
                    request,
                    session,
                    WizardStep.VOCABULARY,
                    error=str(exc),
                    vocabulary_text=vocabulary,
                ),
            )
        row = get_or_create(session, llm_enabled_default=settings.llm_enabled)
        row.vocabulary = terms
        return _setup_redirect(WizardStep.LLM)

    @app.post("/setup/llm", include_in_schema=False)
    def setup_llm(
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        enabled: Annotated[bool, Form()] = False,
        llm_base_url: Annotated[str, Form()] = "",
        llm_model: Annotated[str, Form()] = "",
        llm_api_key: Annotated[str, Form()] = "",
        remove_llm_api_key: Annotated[bool, Form()] = False,
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> Response:
        _require_csrf(request, CSRF_SETUP, csrf_token)
        settings: Settings = request.app.state.settings

        def _rerender(error: str) -> Response:
            # The key field is a password, never prefilled — so the submitted key is
            # never echoed. Only the non-secret overrides survive the re-render.
            # `llm_enabled` is NOT overridden: _setup_context reads the persisted row,
            # so a validation failure that fail-closes shows the checkbox OFF — the
            # honest state — rather than echoing the submitted intent as if it stuck
            # (matching /settings/llm).
            return templates.TemplateResponse(
                request,
                "setup.html",
                _setup_context(
                    request,
                    session,
                    WizardStep.LLM,
                    error=error,
                    llm_base_url=llm_base_url or settings.llm_base_url,
                    llm_model=llm_model or settings.llm_model,
                ),
            )

        # Candidate-state → validate → ONE mutation (see _persist_llm_settings). A
        # pure format error raises and changes nothing; a validation failure persists
        # the valid overrides + candidate key, forces llm_enabled=False, and returns
        # the message to re-render — both re-render the LLM step, fail closed.
        try:
            error = _persist_llm_settings(
                session,
                settings,
                enabled=enabled,
                raw_base_url=llm_base_url,
                raw_model=llm_model,
                raw_key=llm_api_key,
                remove_key=remove_llm_api_key,
            )
        except SetupValidationError as exc:
            return _rerender(str(exc))
        if error is not None:
            return _rerender(error)
        return _setup_redirect(WizardStep.SERVICES)

    @app.post("/setup/finish", include_in_schema=False)
    def setup_finish(
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> RedirectResponse:
        _require_csrf(request, CSRF_SETUP, csrf_token)
        settings: Settings = request.app.state.settings
        complete_onboarding(session, llm_enabled_default=settings.llm_enabled)
        # Commit explicitly before the redirect so the request that follows cannot
        # observe stale onboarding state (the gate re-reads per request).
        session.commit()
        # Launch the guided tutorial only AFTER onboarding commits: a pre-onboarding
        # link to /runs/{id}?tutorial=run would hit the protected gate and bounce
        # back to /setup. Fall back to the queue when the tutorial is unseeded.
        tutorial_run = ready_tutorial_run_id(session)
        if tutorial_run is not None:
            return RedirectResponse(f"/runs/{tutorial_run}?tutorial=run", status_code=303)
        return RedirectResponse("/review", status_code=303)

    @protected.get("/", include_in_schema=False)
    def index(operator: OperatorDep) -> RedirectResponse:
        # On the protected router: when onboarded the gate passes and we land on
        # the review queue; when not, the gate has already redirected to /setup,
        # so this stays an unconditional redirect (no second onboarding read).
        return RedirectResponse("/review", status_code=303)

    @app.get("/static/htmx.min.js")
    def htmx_asset(operator: OperatorDep) -> FileResponse:
        # Served as a route, not a StaticFiles mount: mounts bypass the auth
        # dependency, and "everything but /healthz authenticates" is a stated
        # invariant worth keeping absolute.
        return FileResponse(
            Path(__file__).parent / "static" / "htmx.min.js",
            media_type="text/javascript",
        )

    @app.get("/static/app/{asset_path:path}")
    def app_asset(asset_path: str, operator: OperatorDep) -> FileResponse:
        # Same rationale as htmx_asset: a route, not a StaticFiles mount, so the
        # operator auth dependency stays on every byte served ("everything but
        # /healthz authenticates" is absolute). asset_path is untrusted request
        # input; resolve()+is_relative_to() closes the traversal a StaticFiles
        # mount would otherwise expose. The containment guard runs BEFORE any
        # filesystem access, so a traversal attempt gets no timing signal about
        # what exists outside the root. Mirrors the resolve-then-contain shape
        # the model services use for media paths (resolve_media_path).
        # {asset_path:path} (greedy) is required — Vite nests output under
        # assets/, so a plain segment converter would 404 on any '/'.
        candidate = (_APP_ASSETS_DIR / asset_path).resolve()
        if not candidate.is_relative_to(_APP_ASSETS_DIR):
            raise HTTPException(status_code=404, detail="not found")
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail="not found")
        media_type = _APP_ASSET_MEDIA_TYPES.get(candidate.suffix, "application/octet-stream")
        headers: dict[str, str] = {}
        # Hashed filenames are fingerprinted, so long-immutable caching is safe
        # and honest; unhashed entry names must not get it (they change in place).
        if _looks_hashed(candidate.name):
            headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return FileResponse(candidate, media_type=media_type, headers=headers)

    @protected.get("/runs")
    def runs(
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        status: str | None = None,
        review: str | None = None,
        cursor: str | None = None,
        q: str | None = None,
        speaker: str | None = None,
        source: str | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
        archived: str | None = None,
    ) -> Response:
        settings: Settings = request.app.state.settings
        # ?archived=1 flips to the archived-only view (issue #5); anything else
        # (absent / "0" / blank) is the default listing that hides archived runs.
        show_archived = archived == "1"
        try:
            status_filter = parse_status_filter(status)
            review_filter = parse_review_filter(review)
            search_filters = parse_search_filters(
                q=q,
                speaker=speaker,
                source=source,
                created_from=created_from,
                created_to=created_to,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        parsed_cursor: Cursor | None = None
        # A blank cursor means "start at page 1", mirroring blank status/review
        # meaning "all"; only a non-empty but malformed token is a 400.
        if cursor:
            try:
                parsed_cursor = Cursor.decode(cursor)
            except InvalidCursorError as exc:
                raise HTTPException(status_code=400, detail="invalid cursor") from exc
        page = list_runs(
            session,
            status=status_filter,
            review=review_filter,
            cursor=parsed_cursor,
            page_size=settings.runs_page_size,
            filters=search_filters,
            archived=show_archived,
        )
        next_url = (
            runs_url(
                status=status_filter,
                review=review_filter,
                filters=search_filters,
                archived=show_archived,
                cursor=page.next_cursor,
            )
            if page.next_cursor
            else None
        )
        return templates.TemplateResponse(
            request,
            "runs.html",
            {
                "request": request,
                "page": page,
                "status": status_filter,
                "review": review_filter,
                "statuses": list(RunStatus),
                "reviews": list(ReviewFilter),
                "filters": search_filters,
                "show_archived": show_archived,
                # Toggle target: the same listing in the opposite archive view,
                # preserving the active status/review/search facets.
                "archived_toggle_url": runs_url(
                    status=status_filter,
                    review=review_filter,
                    filters=search_filters,
                    archived=not show_archived,
                ),
                "facet_speakers": searchable_speakers(session),
                "next_url": next_url,
                # Server-issued per-render ids: each namespaces its form's path and
                # makes a double-submit idempotent (see POST /submit, POST /fetch).
                # The upload and fetch forms get INDEPENDENT ids so the two unrelated
                # submissions can never collide on a shared source_path.
                "submission_id": uuid.uuid4().hex,
                "fetch_submission_id": uuid.uuid4().hex,
                # Per-form CSRF tokens, each bound to its own action so one form's
                # token is not valid on the other's route (or on /requeue).
                "csrf_submit": mint_csrf_token(request.app.state.csrf_secret, CSRF_SUBMIT),
                "csrf_fetch": mint_csrf_token(request.app.state.csrf_secret, CSRF_FETCH),
                # Gate the URL-fetch form: when off, it renders disabled (POST /fetch
                # also refuses with 403), matching the CLI's ytdlp_enabled refusal.
                "ytdlp_enabled": settings.ytdlp_enabled,
                "active_nav": "runs",
            },
        )

    @protected.post("/submit")
    def submit_media_upload(
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        file: Annotated[UploadFile, File()],
        submission_id: Annotated[str, Form()],
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> RedirectResponse:
        # CSRF before anything: a forged cross-site upload is refused before the DB
        # write / file finalize. (FastAPI spools the multipart file part before this
        # body runs — the pre-body spool is bounded by _RequestSizeLimitMiddleware +
        # the streaming cap, so a forgery cannot write past those; the DB and the
        # os.replace publish are still gated here.)
        _require_csrf(request, CSRF_SUBMIT, csrf_token)
        settings: Settings = request.app.state.settings
        # An over-cap Content-Length was already rejected before the body was read
        # (_RequestSizeLimitMiddleware); submit_upload enforces the exact per-file
        # cap authoritatively while streaming (covers a lying/absent length).
        try:
            run = submit_upload(
                session,
                stream=file.file,
                filename=file.filename or "",
                submission_id=submission_id,
                media_root=settings.media_root,
                max_bytes=settings.upload_max_bytes,
            )
        except UploadTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except UploadValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except UploadConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        run_id = run.id
        # Commit-before-publish: the durable QUEUED run must exist before the
        # enqueue, so commit here rather than leaning on the dependency's
        # post-return commit (which would run after publish). A broker outage is
        # then non-fatal — the run stays QUEUED and the recovery sweep republishes.
        session.commit()
        return _run_redirect(run_id, published=_publish_or_defer(run_id))

    @protected.post("/fetch")
    def fetch_media_url(
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        url: Annotated[str, Form()],
        submission_id: Annotated[str, Form()],
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> RedirectResponse:
        # CSRF first — reject a forged cross-site fetch before any other check, so
        # a forgery is refused regardless of the ytdlp_enabled flag's state.
        _require_csrf(request, CSRF_FETCH, csrf_token)
        # URL ingestion is an authenticated egress capability gated at the
        # submission surface: refuse before touching the DB when it is off, so no
        # row is created and nothing is published. (The worker's ACQUIRE stage
        # never consults the flag — an already-queued URL run still completes.)
        settings: Settings = request.app.state.settings
        if not settings.ytdlp_enabled:
            # Generic message — never echo the submitted URL into an error body.
            raise HTTPException(status_code=403, detail="URL ingestion is disabled")
        # Mirrors POST /submit: the ingest service is broker-free and does its own
        # SSRF/replay handling; map its typed errors to status codes. The error
        # text is URL-free by construction (UrlValidationError never echoes the
        # URL; a conflict names only the internal source_path), so a signed query
        # string can't leak into a 4xx body.
        try:
            run = submit_url(session, url=url, submission_id=submission_id)
        except (UrlValidationError, UploadValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except UploadConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        run_id = run.id
        # Commit-before-publish, exactly as /submit: the durable QUEUED run must
        # exist before the enqueue, so a broker outage leaves it QUEUED for the
        # recovery sweep rather than failing the request.
        session.commit()
        return _run_redirect(run_id, published=_publish_or_defer(run_id))

    @protected.get("/runs/{run_id}")
    def run_detail(
        run_id: uuid.UUID, request: Request, operator: OperatorDep, session: SessionDep
    ) -> Response:
        run = _run_or_404(session, run_id)
        # The attempt ledger, chronological — matches `voxint status`.
        stage_runs = list(
            session.execute(
                select(StageRun)
                .where(StageRun.pipeline_run_id == run_id)
                .order_by(StageRun.started_at)
            ).scalars()
        )
        settings: Settings = request.app.state.settings
        # Present-only links, decided in Postgres (no filesystem on the read path).
        # Audio reuses the SAME exactly-one-artifact predicate /media serves through
        # (normalized_audio_path resolves iff there is exactly one preprocessed-audio
        # row), so the link never promises a page that would 404; a transcript link
        # needs only that TRANSCRIBE wrote at least one segment.
        try:
            normalized_audio_path(session, run_id, settings.media_root)
            audio_available = True
        except StageDataError:
            audio_available = False
        # GC (issue #15) may have reclaimed the normalized-audio intermediate.
        # The DB row survives, so normalized_audio_path still resolves — but the
        # file is gone. Suppress the dead audio link and show a reclaimed notice.
        media_reclaimed_at = run_intermediate_reclaimed_at(session, run_id)
        if media_reclaimed_at is not None:
            audio_available = False
        transcript_available = bool(
            session.scalar(select(exists().where(TranscriptSegment.pipeline_run_id == run_id)))
        )
        return templates.TemplateResponse(
            request,
            "run_detail.html",
            {
                "request": request,
                "run": run,
                "stage_runs": stage_runs,
                # Provenance for a URL run, reduced to a bare host — the raw
                # source_url (which can carry a signed token in its query) is
                # NEVER passed to the template; None for a local/uploaded run
                # renders as a neutral placeholder.
                "provenance": provenance_host(run.media_item.source_url),
                "audio_available": audio_available,
                "media_reclaimed_at": media_reclaimed_at,
                "transcript_available": transcript_available,
                # Read as a bare boolean (never echoed): a submit/requeue whose
                # enqueue was deferred by a broker outage redirects here with it.
                "enqueue_deferred": request.query_params.get("enqueue") == "deferred",
                # CSRF token for the requeue form (rendered only when FAILED).
                "csrf_requeue": mint_csrf_token(request.app.state.csrf_secret, CSRF_REQUEUE),
                # CSRF token for the cancel form (rendered only for a live run:
                # queued / running / awaiting_adjudication — issue #5).
                "csrf_cancel": mint_csrf_token(request.app.state.csrf_secret, CSRF_CANCEL),
                # Soft-archive + derived-media deletion (issue #5, slice 2). The
                # buttons render only for a terminal run; un-archive replaces
                # archive once the run carries a stamp.
                "run_archived": run.archived_at is not None,
                "run_terminal": run.status
                in (
                    RunStatus.COMPLETED.value,
                    RunStatus.FAILED.value,
                    RunStatus.CANCELLED.value,
                ),
                "csrf_run_archive": mint_csrf_token(
                    request.app.state.csrf_secret, CSRF_RUN_ARCHIVE
                ),
                "csrf_run_unarchive": mint_csrf_token(
                    request.app.state.csrf_secret, CSRF_RUN_UNARCHIVE
                ),
                "csrf_run_media_delete": mint_csrf_token(
                    request.app.state.csrf_secret, CSRF_RUN_MEDIA_DELETE
                ),
                # Post-redirect banner after a derived-media deletion (PRG): the
                # non-negative counts are read as bare ints, never echoed as text.
                "media_delete_result": _media_delete_banner(request),
                # Acquisition context (issue #36): the write-once snapshot, or
                # None for uploads / pre-capture URL runs. Scraped metadata and
                # the operator's own notes render in separate sections.
                "source_metadata": run.media_item.source_metadata,
                "csrf_notes": mint_csrf_token(request.app.state.csrf_secret, CSRF_NOTES),
                # Run-level assets (issue #41): current summary/topics/entity
                # mentions with staleness, plus generation controls.
                "assets": _run_assets_state(session, settings, run_id),
                "csrf_assets_generate": mint_csrf_token(
                    request.app.state.csrf_secret, CSRF_ASSETS_GENERATE
                ),
                "csrf_assets_cancel": mint_csrf_token(
                    request.app.state.csrf_secret, CSRF_ASSETS_CANCEL
                ),
                "tutorial": _tutorial_banner(
                    request, session, page=TutorialPage.RUN_DETAIL, run_id=run_id
                ),
                "active_nav": "runs",
            },
        )

    @protected.get("/runs/{run_id}/transcript")
    def run_transcript(
        run_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        text: str | None = None,
    ) -> Response:
        run = _run_or_404(session, run_id)
        try:
            variant = parse_transcript_text(text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        # Compute once, reuse for both the server-rendered fallback (`lines`) and
        # the island props (avoids a double query). The transcript-player island
        # (issue #48) reuses the already-auth-gated, Range-capable GET /media
        # for its <audio src>; no new backend route.
        lines = attributed_transcript(session, run_id, text=variant)
        settings: Settings = request.app.state.settings
        # Fail-closed seek gating (issue #55): the island only offers per-line
        # playback when GET /media would truly serve and the timeline is sound.
        capability = playback_capability(session, run, settings, _get_media_gate(request))
        # Per-speaker identity color (issue #50): derive the palette from the run's
        # canonical label universe (turns and segments) — the SAME universe the
        # workbench card uses — so a label's color agrees across both surfaces and
        # the JS-off fallback matches the hydrated island. The union also colors a
        # transcript-only label (a segment whose label has no turn). Two cheap
        # DISTINCT queries, kept explicit so the shared universe is provable here.
        palette = speaker_palette(_run_label_universe(session, run_id))
        island_props = {
            "runId": str(run_id),
            "mediaUrl": f"/media/{run_id}",
            "capability": capability.to_props(),
            "segments": [
                {
                    "start": ln.start_seconds,
                    "end": ln.end_seconds,
                    "speaker": ln.speaker,
                    "text": ln.text,
                    "label": ln.diarization_label,
                    # None short-circuits (palette is keyed on real labels only);
                    # keeps mypy happy without changing the value (get(None) → None).
                    "paletteIndex": (
                        palette.get(ln.diarization_label)
                        if ln.diarization_label is not None
                        else None
                    ),
                }
                for ln in lines
            ],
        }
        return templates.TemplateResponse(
            request,
            "transcript.html",
            {
                "request": request,
                "run": run,
                "lines": lines,
                "island_props": island_props,
                "palette": palette,
                "text": variant,
                "variants": list(TranscriptText),
                "active_nav": "runs",
            },
        )

    @protected.post("/runs/{run_id}/requeue")
    def requeue_run(
        run_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        revision: Annotated[int, Form()],
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> RedirectResponse:
        _require_csrf(request, CSRF_REQUEUE, csrf_token)
        # An archived run is read-only — refuse before touching pipeline state so
        # a stale tab can't drive a hidden run live (issue #5).
        _reject_if_archived(_run_or_404(session, run_id))
        # Exact-revision CAS from the form's hidden field: a stale browser tab
        # holding an older revision 409s rather than requeuing a run that already
        # moved on. FAILED-only, at the failed stage (the service enforces both).
        try:
            requeue_failed_run(session, run_id, expected_revision=revision)
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RunNotFailedError, MissingStageError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (StaleRevisionError, InvalidTransitionError) as exc:
            # StaleRevisionError: the tab's revision lost the CAS.
            # InvalidTransitionError: defensive — this FAILED→QUEUED-same-stage
            # path cannot trip it, but the CAS contract permits it.
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        # Commit-before-publish, mirroring /submit: the durable QUEUED run must
        # exist before the enqueue, and a broker outage leaves it QUEUED for the
        # recovery sweep rather than failing the request.
        session.commit()
        return _run_redirect(run_id, published=_publish_or_defer(run_id))

    @protected.post("/runs/{run_id}/cancel")
    def cancel_run_route(
        run_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        revision: Annotated[int, Form()],
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> RedirectResponse:
        """Cancel a live run (QUEUED / RUNNING / AWAITING_ADJUDICATION) — issue #5.

        Exact-revision CAS from the form's hidden field, mirroring /requeue: a
        stale tab holding an older revision 409s rather than cancelling a run
        that already moved on. Cancellation is pure DB state (the existing
        ``→ CANCELLED`` transition), so unlike /submit and /requeue there is
        NOTHING to publish — a worker mid-run observes the cancel at its next
        stage boundary (the currently executing stage body finishes first), and
        a QUEUED run simply never starts. Cancelling an already-cancelled run is
        an idempotent 303, not a 409 (a double-click is success)."""
        _require_csrf(request, CSRF_CANCEL, csrf_token)
        try:
            cancel_run(session, run_id, expected_revision=revision)
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RunNotCancellableError, StaleRevisionError, InvalidTransitionError) as exc:
            # RunNotCancellableError: a COMPLETED/FAILED run can't be cancelled.
            # StaleRevisionError: the tab's revision lost the CAS.
            # InvalidTransitionError: defensive — the cancellable-status guard
            # already excludes the states the CAS would reject, but the CAS
            # contract permits raising it.
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        # Pure DB state, no enqueue — commit and return to the run detail (PRG).
        session.commit()
        return RedirectResponse(f"/runs/{run_id}", status_code=303)

    @protected.post("/runs/{run_id}/archive")
    def archive_run_route(
        run_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> RedirectResponse:
        """Soft-archive a terminal run — hide it from /runs and /review while
        keeping every row intact (issue #5). Last-write-wins operator-visibility
        metadata, so — unlike /requeue and /cancel — there is NO revision field
        and no CAS: a stale tab never 409s, and re-archiving is an idempotent
        303. Only terminal runs archive; a live run 409s (cancel it first)."""
        _require_csrf(request, CSRF_RUN_ARCHIVE, csrf_token)
        try:
            archive_run(session, run_id)
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RunNotArchivableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        session.commit()
        return RedirectResponse(f"/runs/{run_id}", status_code=303)

    @protected.post("/runs/{run_id}/unarchive")
    def unarchive_run_route(
        run_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> RedirectResponse:
        """Reverse an archive — the run reappears in /runs and /review. Always
        allowed and idempotent (un-archiving a non-archived run is a no-op 303).
        Last-write-wins, no CAS (issue #5)."""
        _require_csrf(request, CSRF_RUN_UNARCHIVE, csrf_token)
        try:
            unarchive_run(session, run_id)
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        session.commit()
        return RedirectResponse(f"/runs/{run_id}", status_code=303)

    @protected.post("/runs/{run_id}/media/delete")
    def delete_run_media_route(
        run_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> RedirectResponse:
        """Delete a terminal run's DERIVED audio files — the preprocessed wav and
        per-chunk files — and their rows (issue #5). Irreversible, but scoped:
        it NEVER touches the shared original source (that would orphan a sibling
        run) or the evidence ledger. Commit-before-side-effect: the row delete
        commits first, THEN the files are unlinked (a rolled-back request must
        not leave the DB pointing at a file already gone). A live run 409s."""
        _require_csrf(request, CSRF_RUN_MEDIA_DELETE, csrf_token)
        settings: Settings = request.app.state.settings
        try:
            plan = delete_run_derived_media(
                session, run_id, media_root=settings.media_root
            )
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RunMediaNotDeletableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        session.commit()
        # Post-commit unlink (best-effort, idempotent): the rows are already gone,
        # so a partial filesystem failure is swept later, never fails the action.
        result = unlink_media_paths(plan.paths)
        params = urlencode(
            {
                "media": "deleted",
                "files": result.files_deleted,
                "missing": result.files_missing,
                "failed": result.files_failed,
            }
        )
        return RedirectResponse(f"/runs/{run_id}?{params}", status_code=303)

    @protected.post("/runs/{run_id}/notes")
    def save_operator_notes(
        run_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        notes: Annotated[str, Form()] = "",
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> RedirectResponse:
        """Save the run's operator notes (issue #36) — human context, stored
        apart from scraped source metadata and deliberately OUTSIDE the CAS
        revision: notes are operator prose, not pipeline state, so a plain
        last-write-wins update is the honest single-operator semantics (unlike
        /requeue, where a stale tab acting on pipeline state must 409)."""
        _require_csrf(request, CSRF_NOTES, csrf_token)
        run = _run_or_404(session, run_id)
        cleaned = notes.strip()
        if len(cleaned) > MAX_OPERATOR_NOTES_CHARS:
            raise HTTPException(
                status_code=422,
                detail=f"notes exceed {MAX_OPERATOR_NOTES_CHARS} characters",
            )
        run.operator_notes = cleaned or None
        session.commit()
        return RedirectResponse(f"/runs/{run_id}", status_code=303)

    @protected.get("/runs/{run_id}/export.json")
    def export_run_json(
        run_id: uuid.UUID,
        operator: OperatorDep,
        session: SessionDep,
        text: str | None = None,
    ) -> Response:
        """Run-level JSON export (issue #36): an object envelope carrying the
        run, its acquisition metadata snapshot, the operator notes, and the
        same segment objects as the pinned transcript export. A NEW endpoint on
        purpose — /review/{run_id}/export.json is a frozen bare-array contract
        consumed as-is, so context rides a versioned envelope here instead of a
        breaking shape change there."""
        run = _run_or_404(session, run_id)
        try:
            variant = parse_transcript_text(text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        snapshot = run.media_item.source_metadata
        source_metadata = (
            {
                "source_kind": snapshot.source_kind,
                "title": snapshot.title,
                "uploader": snapshot.uploader,
                "uploader_url": snapshot.uploader_url,
                "channel": snapshot.channel,
                "channel_url": snapshot.channel_url,
                "description": snapshot.description,
                "upload_date": (snapshot.upload_date.isoformat() if snapshot.upload_date else None),
                "duration_seconds": snapshot.duration_seconds,
                "tags": snapshot.tags,
                "canonical_url": snapshot.canonical_url,
                "extractor": snapshot.extractor,
                "extractor_version": snapshot.extractor_version,
                "raw": snapshot.raw,
                "raw_schema_version": snapshot.raw_schema_version,
                "acquired_at": snapshot.acquired_at.isoformat(),
            }
            if snapshot is not None
            else None
        )
        # Run-level assets (issue #41): latest successful asset per kind with
        # full provenance + freshness. An ADDITIVE key under schema_version 1
        # (nullable; absent kinds are simply missing) — failed/running attempts
        # are operational state and stay off the portable export.
        assets = latest_assets(session, run_id)
        current_hash: str | None = None
        # No transcript yet → staleness is simply unknown.
        with contextlib.suppress(RunAssetError):
            current_hash = source_content_hash(load_source(session, run_id))
        enrichment_assets = (
            {
                kind: {
                    "payload": asset.payload,
                    "payload_schema_version": asset.payload_schema_version,
                    "producer": asset.producer,
                    "producer_version": asset.producer_version,
                    "model": asset.model,
                    "generation": asset.generation,
                    "source_content_hash": asset.source_content_hash,
                    "stale": (
                        asset.source_content_hash != current_hash
                        if current_hash is not None
                        else None
                    ),
                    "machine_generated": True,
                    "completed_at": asset.completed_at.isoformat(),
                }
                for kind, asset in sorted(assets.items())
            }
            if assets
            else None
        )
        envelope = {
            "schema_version": 1,
            "run": {
                "id": str(run.id),
                "status": run.status,
                "created_at": run.created_at.isoformat(),
                "operator_notes": run.operator_notes,
            },
            "source_metadata": source_metadata,
            "segments": transcript_payload(attributed_transcript(session, run_id, text=variant)),
            "enrichment_assets": enrichment_assets,
        }
        return Response(
            content=json.dumps(envelope, ensure_ascii=False, indent=2) + "\n",
            media_type=MEDIA_TYPES["json"],
        )

    @protected.get("/review")
    def review_queue(request: Request, operator: OperatorDep, session: SessionDep) -> Response:
        return templates.TemplateResponse(
            request,
            "queue.html",
            {
                "request": request,
                "entries": adjudication_queue(session),
                "operator": operator,
                # CSRF token for the per-row claim forms.
                "csrf_claim": mint_csrf_token(request.app.state.csrf_secret, CSRF_CLAIM),
                "tutorial": _tutorial_banner(request, session, page=TutorialPage.REVIEW_QUEUE),
                "active_nav": "review",
            },
        )

    @protected.post("/review/{run_id}/claim")
    def claim(
        run_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> RedirectResponse:
        # Claim mints the run's claim token, so — unlike the other workbench
        # mutations — it has no unguessable token of its own to gate a forged POST.
        # A CSRF token closes that (a cross-site claim would otherwise pin a run to
        # the victim for the claim TTL). Verified before claim_run touches the DB.
        _require_csrf(request, CSRF_CLAIM, csrf_token)
        settings: Settings = request.app.state.settings
        # An archived run is hidden from the queue; refuse a stale/forged claim so
        # it can't be pinned to a reviewer while hidden (issue #5).
        _reject_if_archived(_run_or_404(session, run_id))
        try:
            token = claim_run(
                session,
                run_id,
                reviewer=operator,
                ttl_seconds=settings.review_claim_ttl_seconds,
            )
        except ClaimUnavailableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        # Guided-tutorial continuity: claiming the tutorial run — while the
        # walkthrough is still active (not yet completed) — lands on the workbench
        # in `adjudicate` mode. Keyed on RUN IDENTITY, not a hidden form field, so
        # EVERY claim control continues the tutorial identically: the banner's own
        # button, the queue row's ordinary "Review" button, and the workbench claim
        # button. A first-time user therefore cannot silently fall out of the
        # walkthrough by clicking the "wrong" (but identical-looking) button. A
        # non-tutorial run is never rewritten, and once completed the tutorial run
        # claims normally (no banner) — replay clears completion and re-activates it.
        suffix = ""
        if ready_tutorial_run_id(session) == run_id:
            row = get_app_settings(session)
            if row is not None and row.tutorial_completed_at is None:
                suffix = "&tutorial=adjudicate"
        return RedirectResponse(f"/review/{run_id}?token={token}{suffix}", status_code=303)

    @protected.get("/review/{run_id}")
    def workbench(
        run_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        token: uuid.UUID | None = None,
    ) -> Response:
        run = _run_or_404(session, run_id)
        if token is not None:
            try:
                verify_claim(session, run_id, token)
            except ClaimMismatchError:
                token = None  # stale tab: render read-only with a claim button
        context = _workbench_context(request, session, run, token)
        # Full-page render only: the banner lives above the body (base.html),
        # outside the #labels htmx target, so decision/enroll fragment swaps — which
        # go through _labels_response and never carry `tutorial` — leave it in place.
        context["tutorial"] = _tutorial_banner(
            request, session, page=TutorialPage.WORKBENCH, run_id=run_id, token=token
        )
        return templates.TemplateResponse(request, "run.html", context)

    @protected.post("/review/{run_id}/release")
    def release(
        run_id: uuid.UUID,
        operator: OperatorDep,
        session: SessionDep,
        token: Annotated[uuid.UUID, Form()],
    ) -> RedirectResponse:
        try:
            release_run(session, run_id, token)
        except (ClaimMismatchError, ClaimUnavailableError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return RedirectResponse("/review", status_code=303)

    @protected.post("/review/{run_id}/labels/{label}/decision")
    def decide(
        run_id: uuid.UUID,
        label: str,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        token: Annotated[uuid.UUID, Form()],
        nonce: Annotated[str, Form(min_length=8, max_length=64)],
        action: Annotated[str, Form()],
        speaker_id: Annotated[uuid.UUID | None, Form()] = None,
    ) -> Response:
        try:
            run = verify_claim(session, run_id, token, for_update=True)
        except ClaimMismatchError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        try:
            decision = Decision(action)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"unknown action {action!r}") from exc
        if (decision is Decision.ASSIGN) != (speaker_id is not None):
            raise HTTPException(
                status_code=422, detail="assign requires speaker_id; others forbid it"
            )
        if speaker_id is not None:
            # FOR SHARE: a concurrent archive/merge takes FOR UPDATE on this
            # row, so the active check and the ledger append below serialize
            # with roster curation instead of racing it.
            speaker = session.execute(
                select(Speaker).where(Speaker.id == speaker_id).with_for_update(read=True)
            ).scalar_one_or_none()
            if speaker is None:
                raise HTTPException(status_code=422, detail=f"no speaker {speaker_id}")
            if not roster_is_active(speaker):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"speaker {speaker.display_name!r} is no longer an active"
                        " roster identity — refresh and pick another"
                    ),
                )
        if label not in {s.label for s in label_states(session, run_id)}:
            raise HTTPException(status_code=404, detail=f"no label {label!r} in run")
        try:
            record_decision(
                session,
                pipeline_run_id=run_id,
                diarization_label=label,
                decision=decision,
                operator=operator,
                idempotency_key=nonce,
                speaker_id=speaker_id,
            )
        except ConflictingReplayError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _labels_response(request, session, run, token)

    @protected.post("/review/{run_id}/labels/{label}/enroll")
    def enroll(
        run_id: uuid.UUID,
        label: str,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        token: Annotated[uuid.UUID, Form()],
        nonce: Annotated[str, Form(min_length=8, max_length=64)],
        display_name: Annotated[str, Form()],
    ) -> Response:
        try:
            run = verify_claim(session, run_id, token, for_update=True)
        except ClaimMismatchError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        settings: Settings = request.app.state.settings
        try:
            enroll_new_speaker(
                session,
                run_id=run_id,
                diarization_label=label,
                display_name=display_name,
                operator=operator,
                idempotency_key=nonce,
                gates=gates_from_settings(settings),
            )
        except EnrollmentError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ConflictingReplayError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _labels_response(request, session, run, token)

    @protected.post("/review/{run_id}/enrich/names")
    def enrich_names(
        run_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        token: Annotated[uuid.UUID, Form()],
    ) -> Response:
        """Operator-triggered offline name sweep (issue #38), synchronous.

        Pure regex + DB over one run's stored rows — sub-second — so it runs
        inline and the htmx response swaps the refreshed suggestion list in.
        Claim-token-gated like every other workbench mutation.
        """
        settings: Settings = request.app.state.settings
        if not settings.enrichment_names_enabled:
            raise HTTPException(status_code=404, detail="name enrichment is disabled")
        try:
            run = verify_claim(session, run_id, token)
        except ClaimMismatchError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        try:
            run_offline_name_producer(session, run_id=run_id, settings=settings)
        except NameProducerError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _labels_response(request, session, run, token)

    @protected.post("/review/{run_id}/candidates/{candidate_id}/decision")
    def decide_name_candidate(
        run_id: uuid.UUID,
        candidate_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        token: Annotated[uuid.UUID, Form()],
        nonce: Annotated[str, Form(min_length=8, max_length=64)],
        verdict: Annotated[str, Form()],
    ) -> Response:
        """Accept/reject one name suggestion — a review record, never identity.

        Writes the profile-review trail only: no speaker, assignment, or
        adjudication ruling is created (drafts are suggestions about identity).
        """
        settings: Settings = request.app.state.settings
        if not settings.enrichment_names_enabled:
            raise HTTPException(status_code=404, detail="name enrichment is disabled")
        try:
            run = verify_claim(session, run_id, token)
        except ClaimMismatchError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        try:
            decision = ProfileDecision(verdict)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"unknown verdict {verdict!r}") from exc
        candidate = session.get(EnrichmentCandidate, candidate_id)
        # This route serves the NAME surface only — a candidate from another
        # run, or another claim field never rendered here, is not decidable.
        if (
            candidate is None
            or candidate.pipeline_run_id != run_id
            or candidate.field != ClaimField.NAME.value
        ):
            raise HTTPException(status_code=404, detail="no such candidate in this run")
        try:
            record_profile_decision(
                session,
                candidate_id=candidate_id,
                decision=decision,
                operator=operator,
                idempotency_key=nonce,
            )
        except StaleCandidateError as exc:
            raise HTTPException(
                status_code=409,
                detail="superseded by a newer sweep — refresh and re-review",
            ) from exc
        except EnrichmentReplayError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _labels_response(request, session, run, token)

    # Transcript downloads: one attributed read shaped by a pure formatter (see
    # voxint/export). The sibling extensions (.txt/.srt/.vtt/.json) share the CLI's
    # exact byte output through render_transcript, so a download and a piped
    # `voxint export … --format …` can never disagree. RTTM lives on its own route
    # (it reads diarization turns, not attributed lines). All accept ?text=raw|
    # enhanced (default enhanced), except RTTM which is speaker-label-only.
    def _export_transcript(
        run_id: uuid.UUID, session: Session, fmt: TranscriptFormat, text: str | None
    ) -> Response:
        _run_or_404(session, run_id)
        try:
            variant = parse_transcript_text(text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        lines = attributed_transcript(session, run_id, text=variant)
        return Response(content=render_transcript(lines, fmt), media_type=MEDIA_TYPES[fmt.value])

    @protected.get("/review/{run_id}/export.txt")
    def export_transcript_txt(
        run_id: uuid.UUID, operator: OperatorDep, session: SessionDep, text: str | None = None
    ) -> Response:
        return _export_transcript(run_id, session, TranscriptFormat.TXT, text)

    @protected.get("/review/{run_id}/export.srt")
    def export_transcript_srt(
        run_id: uuid.UUID, operator: OperatorDep, session: SessionDep, text: str | None = None
    ) -> Response:
        return _export_transcript(run_id, session, TranscriptFormat.SRT, text)

    @protected.get("/review/{run_id}/export.vtt")
    def export_transcript_vtt(
        run_id: uuid.UUID, operator: OperatorDep, session: SessionDep, text: str | None = None
    ) -> Response:
        return _export_transcript(run_id, session, TranscriptFormat.VTT, text)

    @protected.get("/review/{run_id}/export.json")
    def export_transcript_json(
        run_id: uuid.UUID, operator: OperatorDep, session: SessionDep, text: str | None = None
    ) -> Response:
        return _export_transcript(run_id, session, TranscriptFormat.JSON, text)

    @protected.get("/review/{run_id}/export.rttm")
    def export_transcript_rttm(
        run_id: uuid.UUID, operator: OperatorDep, session: SessionDep
    ) -> Response:
        _run_or_404(session, run_id)
        turns = (
            session.execute(
                select(DiarizationTurn)
                .where(DiarizationTurn.pipeline_run_id == run_id)
                .order_by(DiarizationTurn.turn_index)
            )
            .scalars()
            .all()
        )
        return Response(content=to_rttm(turns, str(run_id)), media_type=MEDIA_TYPES["rttm"])

    # ---- Prometheus metrics ----------------------------------------------------
    # Read-only aggregate exposition on the *protected* router: Prometheus scrapes
    # it with basic_auth, so the "everything but /healthz authenticates" invariant
    # holds without a new flag or token path. The one windowed series
    # (voxint_runs_created_24h) bakes its window into the metric name.

    @protected.get("/metrics")
    def metrics(operator: OperatorDep, session: SessionDep) -> Response:
        now = datetime.now(UTC)
        stats = collect_stats(session, since=now - DEFAULT_WINDOW, now=now)
        return Response(
            content=render_prometheus(stats),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    # ---- Operator dashboard (issue #13) ----------------------------------------
    # The HTML sibling of /metrics: the same collect_stats() snapshot, rendered
    # for a human instead of a scraper. Protected like every non-/healthz page.
    # An htmx poll (hx-trigger="every 15s") re-requests this same route with an
    # HX-Request header; we answer that with just the numbers fragment so the
    # nav/chrome are not re-swapped. A malformed ?since= degrades to the 24h
    # default rather than 500-ing a bookmarked or hand-edited URL.

    @protected.get("/dashboard")
    def dashboard(request: Request, operator: OperatorDep, session: SessionDep) -> Response:
        now = datetime.now(UTC)
        raw_since = request.query_params.get("since")
        since = now - DEFAULT_WINDOW
        since_invalid = False
        if raw_since:
            try:
                since = parse_since(raw_since, now=now)
            except ValueError:
                # A bad/bookmarked value degrades to the default window rather
                # than 500-ing; we still tell the operator we ignored it so the
                # page never silently shows a different window than was asked for.
                since_invalid = True
        stats = collect_stats(session, since=since, now=now)
        context = {
            "request": request,
            "stats": stats,
            "active_nav": "dashboard",
            # Iterate the enum (not the sparse status_counts map) so the status
            # table renders in a stable order and zero-fills empty statuses, the
            # same contract format_stats_text/render_prometheus hold.
            "run_statuses": list(RunStatus),
            # Backlog keyed off the enum, not a literal, so a status rename can't
            # silently zero the headline number.
            "review_backlog": stats.status_counts.get(RunStatus.AWAITING_ADJUDICATION.value, 0),
            # Carry the accepted window through the 15s htmx poll so a custom
            # ?since= isn't lost on the first refresh. Only echo a value we
            # actually honored (an invalid one falls back to the default, so we
            # drop it from the poll URL too).
            "since_param": "" if since_invalid or not raw_since else raw_since,
            "since_invalid": since_invalid,
        }
        template = (
            "fragments/dashboard_metrics.html"
            if request.headers.get("HX-Request")
            else "dashboard.html"
        )
        return templates.TemplateResponse(request, template, context)

    # ---- Settings + guided-tutorial lifecycle (issue #3, slice 6) --------------
    # The persistent, re-runnable entry point: re-open the setup wizard, and
    # start / replay / complete the guided tutorial. All @protected (an
    # un-onboarded operator is bounced to /setup by the gate). The two POSTs verify
    # CSRF_SETTINGS and 409 when no tutorial run is available, so a stray token can
    # never "complete" or "replay" an unseeded tutorial.

    def _settings_context(
        request: Request, session: Session, **overrides: Any
    ) -> dict[str, Any]:
        """Shared context for the settings page (GET render + POST re-render).

        Carries the effective LLM state (issue #10) — enablement over env, the
        effective endpoint, and whether an effective key is present and where it
        comes from — never the key value. ``overrides`` lets a POST re-render carry
        an ``error`` and the operator's submitted (non-secret) endpoint inputs.
        """
        settings: Settings = request.app.state.settings
        tutorial_run = ready_tutorial_run_id(session)
        row = get_app_settings(session)
        llm_base_url, llm_model = resolve_effective_llm_endpoint(row, settings)
        context: dict[str, Any] = {
            "request": request,
            "tutorial_available": tutorial_run is not None,
            "tutorial_run_id": tutorial_run,
            "tutorial_completed_at": row.tutorial_completed_at if row else None,
            "llm_enabled": bool(row.llm_enabled) if row is not None else settings.llm_enabled,
            "llm_base_url": llm_base_url,
            "llm_model": llm_model,
            "llm_key_present": bool(resolve_effective_llm_api_key(row, settings)),
            "llm_key_source": effective_llm_key_source(row, settings),
            "llm_budget_ok": llm_budget_fits_stage_lease(settings),
            # Completion celebration after POST /settings/tutorial/complete —
            # shown ONLY when the tutorial is genuinely completed, so a spoofed
            # or bookmarked ?tutorial=done on an unseeded/incomplete tutorial
            # does not falsely claim completion.
            "tutorial_done": (
                request.query_params.get("tutorial") == "done"
                and row is not None
                and row.tutorial_completed_at is not None
            ),
            "csrf_settings": mint_csrf_token(request.app.state.csrf_secret, CSRF_SETTINGS),
            "active_nav": "settings",
            "llm_error": None,
        }
        context.update(overrides)
        return context

    @protected.get("/settings")
    def settings_page(request: Request, operator: OperatorDep, session: SessionDep) -> Response:
        return templates.TemplateResponse(
            request, "settings.html", _settings_context(request, session)
        )

    @protected.post("/settings/llm")
    def settings_llm(
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        enabled: Annotated[bool, Form()] = False,
        llm_base_url: Annotated[str, Form()] = "",
        llm_model: Annotated[str, Form()] = "",
        llm_api_key: Annotated[str, Form()] = "",
        remove_llm_api_key: Annotated[bool, Form()] = False,
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> Response:
        _require_csrf(request, CSRF_SETTINGS, csrf_token)
        settings: Settings = request.app.state.settings

        def _rerender(error: str) -> Response:
            # Password field, never prefilled: the submitted key is never echoed.
            return templates.TemplateResponse(
                request,
                "settings.html",
                _settings_context(request, session, llm_error=error),
            )

        # Same candidate → validate → ONE mutation contract as /setup/llm.
        try:
            error = _persist_llm_settings(
                session,
                settings,
                enabled=enabled,
                raw_base_url=llm_base_url,
                raw_model=llm_model,
                raw_key=llm_api_key,
                remove_key=remove_llm_api_key,
            )
        except SetupValidationError as exc:
            return _rerender(str(exc))
        if error is not None:
            return _rerender(error)
        session.commit()
        return RedirectResponse("/settings", status_code=303)

    @protected.post("/settings/tutorial/complete")
    def tutorial_complete(
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> RedirectResponse:
        _require_csrf(request, CSRF_SETTINGS, csrf_token)
        # Explicit, idempotent completion (mark_tutorial_complete stamps only when
        # currently NULL, so a refresh/repost preserves the original time); 409 when
        # there is no available tutorial run to complete.
        if not mark_tutorial_complete(session):
            raise HTTPException(status_code=409, detail="no tutorial run to complete")
        session.commit()
        return RedirectResponse("/settings?tutorial=done", status_code=303)

    @protected.post("/settings/tutorial/replay")
    def tutorial_replay(
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> RedirectResponse:
        _require_csrf(request, CSRF_SETTINGS, csrf_token)
        run_id = ready_tutorial_run_id(session)
        if run_id is None:
            raise HTTPException(status_code=409, detail="no tutorial run to replay")
        # Non-destructive replay: clear the completion stamp and re-enter the
        # walkthrough. Prior speaker rulings on the run are intentionally preserved
        # (see clear_tutorial_completion / settings.html copy).
        clear_tutorial_completion(session)
        session.commit()
        return RedirectResponse(f"/runs/{run_id}?tutorial=run", status_code=303)

    # ---- Speaker roster curation (issue #7) ------------------------------------
    # View, rename, merge, archive/restore, and remove enrollment embeddings.
    # The append-only decision ledger is never written here — every mutation goes
    # through speakers.roster, which curates only the mutable side. Each POST
    # verifies its own per-action CSRF token before any write; operator-level
    # refusals (RosterError) re-render the roster with the message inline, while
    # missing speakers/embeddings stay real 404s.

    @protected.get("/speakers")
    def speakers_page(request: Request, operator: OperatorDep, session: SessionDep) -> Response:
        return templates.TemplateResponse(
            request, "speakers.html", _roster_context(request, session)
        )

    @protected.post("/speakers/{speaker_id}/rename")
    def speaker_rename(
        speaker_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        display_name: Annotated[str, Form()],
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> Response:
        _require_csrf(request, CSRF_ROSTER_RENAME, csrf_token)
        try:
            rename_speaker(session, speaker_id, display_name)
        except RosterNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RosterError as exc:
            session.rollback()
            return _roster_response(request, session, error=str(exc))
        return _roster_response(request, session)

    @protected.post("/speakers/{speaker_id}/merge")
    def speaker_merge(
        speaker_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        target_id: Annotated[uuid.UUID, Form()],
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> Response:
        _require_csrf(request, CSRF_ROSTER_MERGE, csrf_token)
        try:
            merge_speakers(session, speaker_id, target_id)
        except RosterNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RosterError as exc:
            session.rollback()
            return _roster_response(request, session, error=str(exc))
        return _roster_response(request, session)

    @protected.post("/speakers/{speaker_id}/archive")
    def speaker_archive(
        speaker_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> Response:
        _require_csrf(request, CSRF_ROSTER_ARCHIVE, csrf_token)
        try:
            archive_speaker(session, speaker_id)
        except RosterNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RosterError as exc:
            session.rollback()
            return _roster_response(request, session, error=str(exc))
        return _roster_response(request, session)

    @protected.post("/speakers/{speaker_id}/restore")
    def speaker_restore(
        speaker_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> Response:
        _require_csrf(request, CSRF_ROSTER_RESTORE, csrf_token)
        try:
            restore_speaker(session, speaker_id)
        except RosterNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RosterError as exc:
            session.rollback()
            return _roster_response(request, session, error=str(exc))
        return _roster_response(request, session)

    @protected.post("/speakers/{speaker_id}/embeddings/{embedding_id}/delete")
    def speaker_embedding_delete(
        speaker_id: uuid.UUID,
        embedding_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> Response:
        _require_csrf(request, CSRF_ROSTER_EMBEDDING_DELETE, csrf_token)
        try:
            delete_embedding(session, speaker_id, embedding_id)
        except RosterNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RosterError as exc:
            session.rollback()
            return _roster_response(request, session, error=str(exc))
        return _roster_response(request, session)

    # ---- Web-research jobs + profile-draft review (issue #40) -----------------
    # All research mutations answer with the per-speaker fragment; the fragment
    # re-polls itself (hx-trigger="every 3s") only while its job is active, so
    # polling stops the moment a terminal render goes out.

    def _speaker_or_404(session: Session, speaker_id: uuid.UUID) -> Speaker:
        speaker = session.get(Speaker, speaker_id)
        if speaker is None:
            raise HTTPException(status_code=404, detail="no such speaker")
        return speaker

    @protected.get("/speakers/{speaker_id}/research")
    def research_fragment(
        speaker_id: uuid.UUID, request: Request, operator: OperatorDep, session: SessionDep
    ) -> Response:
        return _research_response(request, session, _speaker_or_404(session, speaker_id))

    @protected.post("/speakers/{speaker_id}/research/start")
    def research_start(
        speaker_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        operator_note: Annotated[str | None, Form(max_length=1000)] = None,
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> Response:
        """Start one budgeted research job for this speaker.

        The rendered form is the budget preview the operator approved; those
        budgets are snapshotted onto the job. Commit-before-publish like every
        enqueue: a broker outage leaves an honest QUEUED job (no hidden
        recovery — the operator cancels and retries)."""
        _require_csrf(request, CSRF_RESEARCH_START, csrf_token)
        speaker = _speaker_or_404(session, speaker_id)
        settings: Settings = request.app.state.settings
        if (
            session.execute(
                select(ResearchJob.id)
                .where(
                    ResearchJob.speaker_id == speaker_id,
                    ResearchJob.status.in_(_ACTIVE_JOB_STATUSES),
                )
                .limit(1)
            ).scalar_one_or_none()
            is not None
        ):
            return _research_response(
                request, session, speaker, error="a research job is already active"
            )
        try:
            job = create_job(
                session,
                speaker_id=speaker_id,
                settings=settings,
                operator_note=operator_note,
            )
        except ResearchJobError as exc:
            session.rollback()
            return _research_response(request, session, speaker, error=str(exc))
        except IntegrityError:
            # The DB's one-active-job-per-speaker partial unique index caught a
            # start the friendly pre-check raced past (double-submit, two tabs).
            session.rollback()
            return _research_response(
                request, session, speaker, error="a research job is already active"
            )
        job_id = job.id
        session.commit()
        _publish_research_job(job_id)
        return _research_response(request, session, speaker)

    @protected.post("/speakers/{speaker_id}/research/{job_id}/cancel")
    def research_cancel(
        speaker_id: uuid.UUID,
        job_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> Response:
        """Cooperative cancel: the loop stops before its next round."""
        _require_csrf(request, CSRF_RESEARCH_CANCEL, csrf_token)
        speaker = _speaker_or_404(session, speaker_id)
        job = session.get(ResearchJob, job_id)
        if job is None or job.speaker_id != speaker_id:
            raise HTTPException(status_code=404, detail="no such research job")
        request_cancel(session, job_id)
        # Commit now so the worker's between-rounds check sees it immediately,
        # not after this response finishes rendering.
        session.commit()
        return _research_response(request, session, speaker)

    @protected.post("/speakers/{speaker_id}/research/candidates/{candidate_id}/decision")
    def decide_profile_candidate(
        speaker_id: uuid.UUID,
        candidate_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        nonce: Annotated[str, Form(min_length=8, max_length=64)],
        verdict: Annotated[str, Form()],
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> Response:
        """Accept/reject one profile claim — a review record, never identity.

        Field-by-field: each candidate row carries one field/value and gets
        its own terminal ruling. Writes the profile-review trail only."""
        _require_csrf(request, CSRF_PROFILE_DECISION, csrf_token)
        speaker = _speaker_or_404(session, speaker_id)
        try:
            decision = ProfileDecision(verdict)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"unknown verdict {verdict!r}") from exc
        candidate = session.get(EnrichmentCandidate, candidate_id)
        # This surface serves speaker-scoped profile fields only; NAME stays on
        # the workbench's suggestion flow.
        if (
            candidate is None
            or candidate.speaker_id != speaker_id
            or candidate.field not in _PROFILE_FIELDS
        ):
            raise HTTPException(status_code=404, detail="no such candidate for this speaker")
        try:
            record_profile_decision(
                session,
                candidate_id=candidate_id,
                decision=decision,
                operator=operator,
                idempotency_key=nonce,
            )
        except StaleCandidateError as exc:
            raise HTTPException(
                status_code=409,
                detail="superseded by a newer research run — refresh and re-review",
            ) from exc
        except EnrichmentReplayError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _research_response(request, session, speaker)

    @protected.get("/runs/{run_id}/assets")
    def run_assets_fragment(
        run_id: uuid.UUID, request: Request, operator: OperatorDep, session: SessionDep
    ) -> Response:
        _run_or_404(session, run_id)
        return _run_assets_response(request, session, run_id)

    @protected.post("/runs/{run_id}/assets/generate")
    def run_assets_generate(
        run_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        kind: Annotated[str | None, Form()] = None,
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> Response:
        """Start generation jobs — one kind, or all three when none is named.

        Kinds with an active job are skipped (create_jobs maps the partial
        unique index to a per-kind skip), so "generate all" degrades per kind.
        Commit-before-publish like every enqueue."""
        _require_csrf(request, CSRF_ASSETS_GENERATE, csrf_token)
        _run_or_404(session, run_id)
        settings: Settings = request.app.state.settings
        if kind is None:
            kinds = tuple(RunAssetKind)
        else:
            try:
                kinds = (RunAssetKind(kind),)
            except ValueError as exc:
                raise HTTPException(
                    status_code=422, detail=f"unknown asset kind {kind!r}"
                ) from exc
        try:
            created, _skipped = create_jobs(
                session, pipeline_run_id=run_id, kinds=kinds, settings=settings
            )
        except RunAssetJobError as exc:
            session.rollback()
            return _run_assets_response(request, session, run_id, error=str(exc))
        job_ids = [job.id for job in created]
        session.commit()
        deferred = sum(1 for job_id in job_ids if not _publish_run_asset_job(job_id))
        # Honest UX: a broker outage leaves real QUEUED rows with no recovery
        # sweep — say so instead of rendering a silently-stuck spinner.
        notice = (
            "worker broker unavailable — the queued job(s) will not start;"
            " cancel and retry once the worker is back"
            if deferred
            else None
        )
        return _run_assets_response(request, session, run_id, error=notice)

    @protected.post("/runs/{run_id}/assets/{job_id}/cancel")
    def run_assets_cancel(
        run_id: uuid.UUID,
        job_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> Response:
        """Cancel: resolves a QUEUED job outright; a RUNNING one is re-checked
        after its LLM call, and a provably-dead one is force-cancelled."""
        _require_csrf(request, CSRF_ASSETS_CANCEL, csrf_token)
        _run_or_404(session, run_id)
        job = session.get(RunAssetJob, job_id)
        if job is None or job.pipeline_run_id != run_id:
            raise HTTPException(status_code=404, detail="no such asset job")
        request_asset_cancel(session, job_id)
        # Commit now so the executor's post-call check sees it immediately.
        session.commit()
        return _run_assets_response(request, session, run_id)

    @protected.get("/media/{run_id}")
    @protected.head("/media/{run_id}")
    def media(
        run_id: uuid.UUID, request: Request, operator: OperatorDep, session: SessionDep
    ) -> Response:
        settings: Settings = request.app.state.settings
        _run_or_404(session, run_id)
        gate = _get_media_gate(request)
        # ONE servability seam (issue #55): resolve_servable_media runs the same
        # reclaimed -> artifact -> gate checks the playback-capability predicate
        # uses, and carries the honest HTTP status on each failure (410 reclaimed,
        # 404 missing/unservable). Capability can therefore never advertise
        # seek_enabled while this route would 404/410.
        try:
            fh, size = resolve_servable_media(session, run_id, settings, gate)
        except MediaResolutionError as exc:
            raise HTTPException(status_code=exc.http_status, detail=str(exc)) from exc
        try:
            byte_range = parse_range(request.headers.get("range"), size)
        except RangeNotSatisfiableError:
            fh.close()
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        headers = {"Accept-Ranges": "bytes", "Content-Type": "audio/wav"}
        if byte_range is None:
            status, start, length = 200, 0, size
        else:
            status, start, length = 206, byte_range.start, byte_range.length
            headers["Content-Range"] = f"bytes {byte_range.start}-{byte_range.end}/{size}"
        headers["Content-Length"] = str(length)
        if request.method == "HEAD":
            fh.close()
            return Response(status_code=status, headers=headers)
        return StreamingResponse(
            _stream_file(fh, start, length), status_code=status, headers=headers
        )

    # Mount the gated routes last: every @protected route above is now attached to
    # `app` behind require_onboarded, while the @app routes stay exempt.
    app.include_router(protected)


def _stream_file(fh: BinaryIO, start: int, length: int) -> Iterator[bytes]:
    """Stream from the gate-validated descriptor — never reopen by path."""
    remaining = length
    with fh:
        fh.seek(start)
        while remaining > 0:
            chunk = fh.read(min(_MEDIA_CHUNK_BYTES, remaining))
            if not chunk:
                return  # file shrank mid-stream; truncate rather than hang
            remaining -= len(chunk)
            yield chunk


app = create_app()
