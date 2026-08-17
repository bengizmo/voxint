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
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, BinaryIO, cast
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.templating import Jinja2Templates
from sqlalchemy import exists, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.types import ASGIApp, Receive, Scope, Send

from voxint import __version__
from voxint.adjudication.enrollment import EnrollmentError, enroll_new_speaker
from voxint.adjudication.ledger import (
    ConflictingReplayError,
    WordRangeError,
    record_decision,
)
from voxint.adjudication.merge import (
    MergeConflictError,
    MergeError,
    apply_merge,
    preview_merge,
)
from voxint.adjudication.resolver import (
    QUEUE_SORTS,
    LabelState,
    Resolution,
    adjudication_queue,
    label_states,
    segment_states,
)
from voxint.adjudication.review_state import (
    set_correction,
    set_verified,
    verified_progress,
)
from voxint.adjudication.slots import (
    ClaimMismatchError,
    ClaimUnavailableError,
    claim_run,
    release_run,
    verify_claim,
)
from voxint.adjudication.splits import (
    UnsplittableError,
    derive_children,
    record_split,
    splittable_words,
)
from voxint.adjudication.transcript import (
    TranscriptLine,
    TranscriptText,
    attributed_transcript,
    effective_text,
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
    PlaybackCapability,
    playback_capability,
    representative_turns,
    resolve_servable_media,
)
from voxint.api.presentation import (
    format_age,
    format_duration,
    friendly_media_label,
    humanize_stage,
    humanize_status,
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
    EffectiveFlags,
    clear_tutorial_completion,
    complete_onboarding,
    effective_llm_key_source,
    feature_flag_state,
    get_app_settings,
    get_or_create,
    is_onboarded,
    llm_endpoint_form_fields,
    mark_tutorial_complete,
    ready_tutorial_run_id,
    resolve_effective_enrichment_names_enabled,
    resolve_effective_enrichment_web_research_enabled,
    resolve_effective_llm_api_key,
    resolve_effective_llm_enabled,
    resolve_effective_source_authority_domains,
    resolve_effective_voxint_web_research,
    resolve_effective_web_search_base_url,
    resolve_effective_ytdlp_enabled,
    validate_effective_flags,
)
from voxint.config import Settings, get_settings, llm_budget_fits_stage_lease
from voxint.db.models import (
    MAX_CORRECTED_TEXT_CHARS,
    AssignmentMethod,
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
    SegmentReviewState,
    SegmentSplitBoundary,
    Speaker,
    SpeakerAssignment,
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
from voxint.enrichment.triage import (
    EvidenceRef,
    TriageInputs,
    TriageScore,
    VoiceSignal,
    parse_authority_domains,
)
from voxint.enrichment.triage import (
    score as triage_score,
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
from voxint.media.peaks import (
    CachedPeaks,
    PeaksError,
    SourceFingerprint,
    compute_peaks,
    load_cached_peaks,
    peaks_artifact_row,
    store_peaks,
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
# Operator-facing display helpers (issue #56), called directly from the console
# templates. `format_age` takes an injected `now` the routes pass in context.
templates.env.globals["friendly_media_label"] = friendly_media_label
templates.env.globals["format_duration"] = format_duration
templates.env.globals["format_age"] = format_age
templates.env.globals["humanize_stage"] = humanize_stage
templates.env.globals["humanize_status"] = humanize_status


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


@dataclass(frozen=True)
class _VoiceRow:
    """Per-label grounded cosine facts for triage voice-support."""

    name_norm: str
    confidence: float | None
    grounded: bool


@dataclass(frozen=True)
class _HintTriage:
    """The triage a template renders beside one representative suggestion."""

    priority: float
    components: dict[str, float]
    # Proposed candidates for the same (label, value) hidden behind this
    # representative — so a decision on one producer's claim never silently
    # buries another producer's still-open proposal (#42).
    unresolved_peers: int


def _name_match_key(value: str) -> str:
    """The one normalization for matching a name candidate to its peers, to a
    voice assignment, and for representative grouping: strip + casefold.

    Deliberately NOT ``roster.normalize_display_name`` — that raises above 120
    chars, and a NAME candidate ``value`` may be longer (the column allows 4000);
    calling it in this read path would 500 the workbench. Using one key
    everywhere keeps a representative card and its agreement/voice signals about
    the same set of candidates.
    """
    return value.strip().casefold()


def _voice_by_label(session: Session, run_id: uuid.UUID) -> dict[str, _VoiceRow]:
    """Grounded cosine facts per diarization label (one cosine row per label).

    Only ``method='cosine'`` carries a roster speaker + confidence + grounding;
    ``llm_hint`` has none. **Active roster identities only** — a since-merged or
    archived speaker's stale display name must not drive voice matching (it would
    invert the signal: a false conflict against the merge target, or false
    support for a tombstone). ``UNIQUE(run, label, method)`` gives one row per
    label; the ORDER BY only makes the dict-build deterministic if that ever
    changes.
    """
    rows = session.execute(
        select(
            SpeakerAssignment.diarization_label,
            Speaker.display_name,
            SpeakerAssignment.confidence,
            SpeakerAssignment.grounded,
        )
        .join(Speaker, Speaker.id == SpeakerAssignment.speaker_id)
        .where(
            SpeakerAssignment.pipeline_run_id == run_id,
            SpeakerAssignment.method == AssignmentMethod.COSINE.value,
            Speaker.merged_into_id.is_(None),
            Speaker.deleted_at.is_(None),
        )
        .order_by(SpeakerAssignment.id)
    ).all()
    return {
        label: _VoiceRow(
            name_norm=_name_match_key(name),
            confidence=confidence,
            grounded=grounded,
        )
        for label, name, confidence, grounded in rows
    }


def _name_peer_counts(views: Sequence[CandidateView]) -> dict[tuple[str | None, str], int]:
    """Distinct producers proposing the same (label, normalized name) across
    ACTIVE candidates (proposed or accepted). Rejected/superseded never
    corroborate — computed over all views, before representative collapsing."""
    producers: dict[tuple[str | None, str], set[str]] = {}
    for view in views:
        if view.state not in (CandidateState.PROPOSED, CandidateState.ACCEPTED):
            continue
        key = (view.candidate.diarization_label, _name_match_key(view.candidate.value))
        producers.setdefault(key, set()).add(view.candidate.producer_run.producer)
    return {key: len(names) for key, names in producers.items()}


def _triage_for(
    view: CandidateView,
    *,
    voice: _VoiceRow | None,
    peer_count: int,
    authority: frozenset[str],
) -> TriageScore:
    """Fuse one candidate's signals into an explainable review priority."""
    candidate = view.candidate
    voice_signal: VoiceSignal | None = None
    if voice is not None:
        voice_signal = VoiceSignal(
            matches_value=_name_match_key(candidate.value) == voice.name_norm,
            grounded=voice.grounded,
            confidence=voice.confidence,
        )
    return triage_score(
        TriageInputs(
            field=candidate.field,
            producer=candidate.producer_run.producer,
            producer_score=candidate.score,
            producer_components=candidate.score_components or {},
            evidence=tuple(EvidenceRef(kind=e.kind, url=e.url) for e in view.evidence),
            voice=voice_signal,
            peer_producer_count=peer_count,
            authority_domains=authority,
        )
    )


def _name_suggestions(
    session: Session, run_id: uuid.UUID
) -> tuple[list[CandidateView], dict[str, list[CandidateView]], dict[uuid.UUID, _HintTriage]]:
    """Representative NAME suggestions for the workbench: run-level + per-label,
    triage-ordered, with a per-representative triage map.

    Each (target, normalized value) group renders one representative so rerun
    duplicates beside decided history are never presented as new suggestions.
    Cross-producer facts (voice support, agreement) and each candidate's triage
    priority are computed over all active candidates BEFORE collapsing.
    """
    views = [
        view
        for view in candidates_for_run(session, run_id)
        if view.candidate.field == ClaimField.NAME.value
    ]
    voice_map = _voice_by_label(session, run_id)
    peer_counts = _name_peer_counts(views)

    def _score(view: CandidateView) -> TriageScore:
        label = view.candidate.diarization_label
        peer_key = (label, _name_match_key(view.candidate.value))
        return _triage_for(
            view,
            voice=voice_map.get(label) if label is not None else None,
            peer_count=peer_counts.get(peer_key, 1),
            authority=frozenset(),  # name candidates carry no URL evidence
        )

    scores: dict[uuid.UUID, TriageScore] = {v.candidate.id: _score(v) for v in views}

    def _order(view: CandidateView) -> tuple[int, float, str, str]:
        # Decided history first (a decided value is never re-shown as new), then
        # higher triage PRIORITY — never a raw cross-producer score — then a
        # stable tiebreak. Same key selects representatives and orders the lists.
        return (
            _HINT_STATE_PRECEDENCE[view.state],
            -scores[view.candidate.id].priority,
            view.candidate.value.casefold(),
            str(view.candidate.id),
        )

    groups: dict[tuple[str | None, str], CandidateView] = {}
    proposed_counts: dict[tuple[str | None, str], int] = {}
    for view in views:
        key = (view.candidate.diarization_label, _name_match_key(view.candidate.value))
        if view.state is CandidateState.PROPOSED:
            proposed_counts[key] = proposed_counts.get(key, 0) + 1
        current = groups.get(key)
        if current is None or _order(view) < _order(current):
            groups[key] = view

    triage: dict[uuid.UUID, _HintTriage] = {}
    for key, view in groups.items():
        score = scores[view.candidate.id]
        # Proposed peers hidden behind this representative — never below zero.
        rep_is_proposed = 1 if view.state is CandidateState.PROPOSED else 0
        hidden = max(0, proposed_counts.get(key, 0) - rep_is_proposed)
        triage[view.candidate.id] = _HintTriage(
            priority=score.priority, components=score.components, unresolved_peers=hidden
        )

    run_level = sorted((view for (label, _), view in groups.items() if label is None), key=_order)
    per_label: dict[str, list[CandidateView]] = {}
    for (label, _), view in groups.items():
        if label is not None:
            per_label.setdefault(label, []).append(view)
    for label_views in per_label.values():
        label_views.sort(key=_order)
    return run_level, per_label, triage


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
    name_hints_run, name_hints_labels, name_triage = _name_suggestions(session, run.id)
    # Per-turn playback (issue #49) + fail-closed seek gating (issue #55). The
    # workbench-player island (mounted OUTSIDE #labels) reads `capability` to
    # enable/disable the server-rendered, htmx-swapped seek buttons; the buttons
    # themselves carry the representative-turn timings for "preview this speaker".
    capability = playback_capability(session, run, settings, _get_media_gate(request))
    return {
        "name_hints_run": name_hints_run,
        "name_hints_labels": name_hints_labels,
        # Review-priority + component breakdown per representative (#42), keyed by
        # candidate id so labels.html can render it without changing the hint shape.
        "name_triage": name_triage,
        "names_enabled": resolve_effective_enrichment_names_enabled(
            get_app_settings(session), settings
        ),
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
        # Per-segment overrides (issue #54 Phase B): keyed by segment id, so a
        # preview segment can show its this-segment attribution + a reset control.
        "segment_overrides": segment_states(session, run.id),
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


def _wants_html(request: Request) -> bool:
    """True for a plain browser form navigation (Accept prefers text/html), False
    for the island's fetch (which asks for application/json explicitly). Lets one
    write route serve the JSON island contract AND a JS-off HTML redirect fallback
    without a second route. The default httpx/TestClient Accept (``*/*``) is not
    HTML, so it keeps the JSON path — existing route tests are unaffected."""
    accept = request.headers.get("accept", "")
    return "text/html" in accept and "application/json" not in accept


# Response header marking a 409 as a lost/again-taken claim (issue #59), so the
# island can distinguish it from a segment-STATE 409 the same route raises (a
# non-child range, an already-split parent). A claim loss must stop the review
# loop and prompt a re-claim; a state conflict shows an inline reason and keeps
# the claim. The value is opaque; only presence + "claim" matters to the client.
_CLAIM_CONFLICT_HEADERS = {"X-Voxint-Conflict": "claim"}


def _wants_island_json(request: Request) -> bool:
    """True only when the caller explicitly asks for JSON (the island's ``apiFetch``
    sets ``Accept: application/json``). Unlike ``not _wants_html``, this is a
    POSITIVE signal: the htmx labels workbench and the default ``*/*`` test client
    stay on the server-rendered path, so a route that also serves the island keeps
    its HTML-fragment contract byte-identical for every non-island caller."""
    return "application/json" in request.headers.get("accept", "")


# Capability reason codes meaning GET /media would not serve bytes at all (as
# opposed to a present-but-untrusted timeline). Mirrors MEDIA_UNAVAILABLE_CODES
# in frontend/src/components/PlaybackControls.tsx.
_MEDIA_UNAVAILABLE_CODES = frozenset({"media_missing", "media_reclaimed", "media_unservable"})


def _transcript_island_props(
    session: Session,
    run_id: uuid.UUID,
    lines: list[TranscriptLine],
    palette: dict[str, int],
    capability: PlaybackCapability,
    settings: Settings,
) -> dict[str, Any]:
    """Shared island props for the linear transcript surfaces (issues #48/#50/#53).

    Both the read-only ``transcript-player`` and the claim-gated ``review-stepper``
    read the SAME per-segment shape, so the hydrated island and the JS-off
    fallback flag/color identically and a segment's write id never drifts between
    the display and the review loop.
    """
    # Waveform strip (issue #57): the strip's colored regions come from the
    # DIARIZATION TURNS, not the transcript segments — a segment carries only
    # its dominant-overlap label, which is not an honest who-spoke-when map
    # (it hides overlaps and untranscribed speech). Same palette as the list
    # badges, so the colors can never disagree.
    turn_rows = session.execute(
        select(
            DiarizationTurn.start_seconds,
            DiarizationTurn.end_seconds,
            DiarizationTurn.label,
            DiarizationTurn.overlap,
        )
        .where(DiarizationTurn.pipeline_run_id == run_id)
        .order_by(DiarizationTurn.start_seconds, DiarizationTurn.turn_index)
    ).all()
    # peaksUrl is server-owned truth like mediaUrl: non-null only when the peaks
    # route could actually answer 200 — either the WAV is servable (a first
    # request computes the envelope) OR it was formally RECLAIMED and a cached
    # envelope survives (served unverified, by design). A cached row does NOT
    # rescue media_missing/media_unservable: with no reclamation stamp the route
    # cannot verify the (absent/unopenable) WAV, so it fails closed to 404/410 —
    # emitting the URL there would make the island fetch on a loop. Any capability
    # reason not about media servability (a bad timeline) still leaves the
    # amplitude route answerable, so it does not gate peaksUrl.
    reason_codes = {r.code for r in capability.reasons}
    media_unavailable = bool(reason_codes & _MEDIA_UNAVAILABLE_CODES)
    reclaimed_with_cache = (
        "media_reclaimed" in reason_codes
        and peaks_artifact_row(session, run_id) is not None
    )
    peaks_available = not media_unavailable or reclaimed_with_cache
    return {
        "runId": str(run_id),
        "mediaUrl": f"/media/{run_id}",
        "peaksUrl": f"/media/{run_id}/peaks" if peaks_available else None,
        "capability": capability.to_props(),
        "turns": [
            {
                "start": start,
                "end": end,
                "paletteIndex": palette.get(label),
                "overlap": overlap,
            }
            for start, end, label, overlap in turn_rows
        ],
        # Low-confidence triage threshold (issue #53): the island and the JS-off
        # fallback compare against the SAME server setting, so they flag
        # identically. A segment with confidence None is never flagged.
        "lowConfidenceThreshold": settings.review_low_confidence_threshold,
        "segments": [_island_segment(ln, palette) for ln in lines],
    }


def _island_segment(ln: TranscriptLine, palette: dict[str, int]) -> dict[str, Any]:
    """One transcript line as the island's per-segment shape — the ONE builder the
    hydrated props and the split-route response share, so a page reload and a live
    split can never disagree on a segment's fields.

    ``sourceSegmentId`` is the immutable PARENT id (issue #59): the verify / correct
    / split write target, identical to ``segmentId`` for an unsplit line and shared
    across a split parent's derived children. ``reviewTarget`` is true on exactly
    one line per parent — the queue entry — so the N-of-M loop counts one target
    per parent and never double-counts children.
    """
    return {
        "start": ln.start_seconds,
        "end": ln.end_seconds,
        "speaker": ln.speaker,
        "text": ln.text,
        "label": ln.diarization_label,
        "confidence": ln.confidence,
        # None short-circuits (palette is keyed on real labels only); keeps mypy
        # happy without changing the value (get(None) → None).
        "paletteIndex": (
            palette.get(ln.diarization_label) if ln.diarization_label is not None else None
        ),
        # Per-segment review state (issues #53/#58). segmentId is the write target
        # for verify/correct; verified/corrected drive the verify-and-advance loop
        # and the "edited" badge. None segmentId (a synthetic/blank line) is simply
        # never a review target.
        "segmentId": (str(ln.segment_id) if ln.segment_id is not None else None),
        "verified": ln.verified,
        "corrected": ln.corrected,
        # Split provenance (issue #59): the parent write target + the single
        # queue-entry flag. sourceSegmentId == segmentId for an unsplit line.
        "sourceSegmentId": (
            str(ln.source_segment_id) if ln.source_segment_id is not None else None
        ),
        "reviewTarget": ln.review_target,
        # Word-range coordinates of a split child (issue #59 slice 3): what the
        # per-child reassign picker posts to /relabel to scope a ruling to this
        # child. Both None on unsplit and synthetic lines.
        "wordStart": ln.word_start,
        "wordEnd": ln.word_end,
        # The child's OWN range-override speaker id (None ⇒ inheriting): the picker
        # binds its <select> to this so it shows a child-scoped assignment only when
        # one exists, never an inherited speaker mislabeled as a child ruling.
        "wordRangeSpeakerId": (
            str(ln.word_range_speaker_id) if ln.word_range_speaker_id is not None else None
        ),
    }


def _run_island_segments(session: Session, run_id: uuid.UUID) -> list[dict[str, Any]]:
    """The run's island segment payload (issue #59) — CORRECTED variant, split
    parents expanded — for a live write to reconcile the console against server
    truth. Same builder as hydration, so a split response and a page reload agree."""
    palette = speaker_palette(_run_label_universe(session, run_id))
    lines = attributed_transcript(session, run_id, text=TranscriptText.CORRECTED)
    return [_island_segment(ln, palette) for ln in lines]


def _run_reconcile_response(session: Session, run_id: uuid.UUID) -> JSONResponse:
    """The whole-run island reconcile — every segment (split parents expanded) plus
    the run's N-of-M counter — the shape a STRUCTURAL write returns so the console
    adopts server truth wholesale rather than patching one line. Shared by /split
    and the island /relabel path (a reassignment changes a child's speaker string,
    which a per-segment patch cannot express, so both re-render the whole run)."""
    verified_n, total = verified_progress(session, run_id)
    return JSONResponse(
        {
            "segments": _run_island_segments(session, run_id),
            "progress": {"verified": verified_n, "total": total},
        }
    )


def _segment_is_split(session: Session, segment_id: uuid.UUID) -> bool:
    """Whether a segment carries at least one operator split boundary (issue #59)."""
    return (
        session.execute(
            select(SegmentSplitBoundary.id)
            .where(SegmentSplitBoundary.parent_segment_id == segment_id)
            .limit(1)
        ).first()
        is not None
    )


def _segment_child_ranges(
    session: Session, segment: TranscriptSegment
) -> set[tuple[int, int]]:
    """The half-open ``(word_start, word_end)`` ranges of a segment's current
    derived split children (issue #59 slice 3).

    The reassign route validates a submitted range against this set so a ruling
    can only target a child that actually exists right now — an arbitrary range
    would write a ledger row the read path never applies (it matches children by
    exact coordinates). Empty for an unsplit or unsplittable segment."""
    cuts = list(
        session.execute(
            select(SegmentSplitBoundary.word_index).where(
                SegmentSplitBoundary.parent_segment_id == segment.id
            )
        ).scalars()
    )
    if not cuts:
        return set()
    children = derive_children(segment, cuts)
    if children is None or len(children) < 2:
        return set()
    return {(child.word_start, child.word_end) for child in children}


def _segment_is_corrected(session: Session, segment_id: uuid.UUID) -> bool:
    """Whether a segment has operator-corrected text (issues #58/#59)."""
    row = session.get(SegmentReviewState, segment_id)
    return row is not None and row.corrected_text is not None


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
    _base_value, _base_default, _model_value, _model_default = llm_endpoint_form_fields(
        row, settings
    )
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
        # LLM step: the row's enablement over env, its (non-secret) endpoint OVERRIDE
        # (blank when inheriting env; env default shown as placeholder — issue #46),
        # and whether an EFFECTIVE key (UI-stored row value winning over env) is
        # present and where it comes from — never the key value itself.
        "llm_enabled": bool(row.llm_enabled) if row is not None else settings.llm_enabled,
        "llm_base_url": _base_value,
        "llm_base_url_default": _base_default,
        "llm_model": _model_value,
        "llm_model_default": _model_default,
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
    # Tri-state "revert to installation setting" (issue #46): a blank field already
    # normalizes to None, but a submission that merely echoes the env default (the
    # forms render blank with that default as placeholder, yet an operator may still
    # type it) must ALSO store NULL so the row keeps inheriting env — otherwise
    # saving any LLM change silently pins the env value onto the row and a later
    # LLM_BASE_URL/LLM_MODEL change stops applying with no cue.
    if base_url is not None and base_url == settings.llm_base_url:
        base_url = None
    if model is not None and model == settings.llm_model:
        model = None
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


# The live-read feature flags the Settings "Features" section exposes as tri-state
# runtime toggles (issue #62). Each entry is (column/config name, operator label,
# help text). Order is display order. LLM enablement lives in its own section, and
# the web-research provider toggles are the external-sources child (#76), so
# neither appears here. The names match the ``AppSettings`` columns / ``Settings``
# fields exactly, so the resolvers and the persist path key off them directly.
_FEATURE_FLAG_META: tuple[tuple[str, str, str], ...] = (
    (
        "enrichment_names_enabled",
        "Speaker name suggestions",
        "Scan finalized transcripts for likely speaker names. Runs fully offline —"
        " no LLM required.",
    ),
    (
        "enrichment_names_llm_enabled",
        "LLM name pass",
        "Additionally ask the enhancement LLM to propose names. Requires LLM"
        " enhancement and speaker name suggestions to be on.",
    ),
    (
        "enrichment_run_assets_enabled",
        "Run assets (summary, topics, entities)",
        "Generate a summary, topic list, and grounded entity mentions for each run."
        " Requires LLM enhancement.",
    ),
    (
        "enrichment_run_assets_autogenerate",
        "Auto-generate run assets",
        "Start run-asset generation automatically when a run is finalized. Requires"
        " run assets to be on.",
    ),
    (
        "ytdlp_enabled",
        "Download media from a URL",
        "Allow submitting media by URL, fetched with yt-dlp. Independent of the LLM"
        " features.",
    ),
)
_FEATURE_FLAG_NAMES: tuple[str, ...] = tuple(name for name, _, _ in _FEATURE_FLAG_META)
_FEATURE_FLAG_CHOICES: tuple[str, ...] = ("on", "off", "inherit")

# Operator-plain copy for the invariant violations this section can surface (#62).
# validate_effective_flags is the SINGLE source of WHICH combinations are invalid,
# but its messages name the flag identifiers (enrichment_run_assets_enabled, …) —
# the exact jargon this arc exists to keep out of a non-technical operator's way.
# So the Features boundary translates the reachable messages to plain language,
# while the config boot validator keeps the identifier-bearing strings (a .env
# editor wants the variable name). Keyed on the exact shared message; an
# un-mapped message (e.g. a web-research invariant, unreachable from this form)
# falls through to the original. A drift test locks the four reachable keys so a
# reworded invariant can never silently fall back to jargon here.
_FEATURE_INVARIANT_COPY: dict[str, str] = {
    "enrichment_names_llm_enabled requires llm_enabled=true — the "
    "LLM name pass reuses the configured enhancement endpoint": (
        "The LLM name pass needs LLM transcript enhancement turned on. Turn it on"
        " in the LLM section below, or turn the LLM name pass off."
    ),
    "enrichment_names_llm_enabled requires enrichment_names_enabled=true"
    " — the LLM pass is additive to the offline name producer": (
        "The LLM name pass needs speaker name suggestions turned on — it adds to"
        " the offline name finder."
    ),
    "enrichment_run_assets_enabled requires llm_enabled=true — the"
    " asset generators reuse the configured enhancement endpoint": (
        "Run assets need LLM transcript enhancement turned on. Turn it on in the"
        " LLM section below, or turn run assets off."
    ),
    "enrichment_run_assets_autogenerate requires"
    " enrichment_run_assets_enabled=true — the post-finalize step"
    " only enqueues the feature it rides on": (
        "Auto-generating run assets needs run assets turned on."
    ),
}


def _persist_feature_flags(
    session: Session, settings: Settings, *, submitted: dict[str, str]
) -> list[str]:
    """Apply the Features-section tri-state toggles as ONE deliberate mutation (#62).

    ``submitted`` maps each flag name to ``"on"``/``"off"``/``"inherit"``. The
    candidate column value is ``True``/``False``/``None`` respectively (``None`` =
    inherit the env default — the tri-state that never permanently pins an
    override). Returns the list of operator-plain error messages (empty ⇒ success);
    following the ``_persist_llm_settings`` contract, it computes the candidate
    effective combination and validates it through the SINGLE shared
    :func:`validate_effective_flags` BEFORE touching the row, so an
    invariant-violating submission (e.g. the LLM name pass without LLM enhancement)
    writes NOTHING — not even a get_or_create (the API session commits on the 200
    error re-render). A valid submission then performs the single mutation and the
    caller commits.

    An unexpected choice value (a stale client, a hand-crafted POST — never the
    shipped radios) is REJECTED rather than silently coerced: mapping it to "off"
    would quietly disable a feature, and to "inherit" would quietly drop an
    override. Missing fields still default to ``"inherit"`` — this is a full-form
    replace, and the real form always submits all five radios.

    The flags NOT edited here (``llm_enabled`` and the web-research provider trio)
    are resolved at their CURRENT effective value so a dependency invariant fires
    against the real system state — enabling ``run_assets`` while LLM is off is
    rejected — and this section never flips an unrelated setting.
    """
    row = get_app_settings(session)
    candidates: dict[str, bool | None] = {}
    for name in _FEATURE_FLAG_NAMES:
        choice = submitted.get(name, "inherit")
        if choice not in _FEATURE_FLAG_CHOICES:
            return ["Unrecognized feature setting — choose On, Off, or Use installation setting."]
        candidates[name] = None if choice == "inherit" else (choice == "on")

    def _effective(name: str) -> bool:
        candidate = candidates[name]
        return bool(getattr(settings, name)) if candidate is None else candidate

    errors = validate_effective_flags(
        EffectiveFlags(
            # Not edited in this section — resolved at the current effective value.
            llm_enabled=resolve_effective_llm_enabled(row, settings),
            voxint_web_research=resolve_effective_voxint_web_research(row, settings),
            enrichment_web_research_enabled=resolve_effective_enrichment_web_research_enabled(
                row, settings
            ),
            web_search_base_url=resolve_effective_web_search_base_url(row, settings),
            # Edited here — the candidate over env default.
            enrichment_names_enabled=_effective("enrichment_names_enabled"),
            enrichment_names_llm_enabled=_effective("enrichment_names_llm_enabled"),
            enrichment_run_assets_enabled=_effective("enrichment_run_assets_enabled"),
            enrichment_run_assets_autogenerate=_effective(
                "enrichment_run_assets_autogenerate"
            ),
        )
    )
    if errors:
        # Translate every violated invariant to operator-plain copy (all of them,
        # so a two-fault submission fixes in one pass). Nothing is written.
        return [_FEATURE_INVARIANT_COPY.get(message, message) for message in errors]
    # Valid → one mutation. get_or_create only now, so a rejected save writes nothing.
    row = get_or_create(session, llm_enabled_default=settings.llm_enabled)
    for name, value in candidates.items():
        setattr(row, name, value)
    return []


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
    row = get_app_settings(session)
    authority = parse_authority_domains(
        resolve_effective_source_authority_domains(row, settings)
    )
    triage: dict[uuid.UUID, TriageScore] = {
        view.candidate.id: _triage_for(view, voice=None, peer_count=1, authority=authority)
        for view in views
        if view.state is CandidateState.PROPOSED
    }
    # The unresolved bucket: proposed (undecided) drafts, highest review priority
    # first (#42). No score floor — a floor is an uncalibrated implicit reject.
    proposed = sorted(
        (v for v in views if v.state is CandidateState.PROPOSED),
        key=lambda v: (
            -triage[v.candidate.id].priority,
            v.candidate.value.casefold(),
            v.candidate.created_at,
        ),
    )
    return {
        "speaker": speaker,
        "job": job,
        "job_active": job is not None and job.status in _ACTIVE_JOB_STATUSES,
        "gates_open": research_gates_open(settings, row),
        "budget": budget_snapshot(settings),
        "proposed": proposed,
        "triage": triage,
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
            # Echo the operator's SUBMITTED endpoint text verbatim (blank stays
            # blank, with the env default as the placeholder) so their in-progress
            # input survives — never fall back to settings.llm_base_url here, which
            # would put the env default in the input `value` and falsely show an
            # inheriting field as pinned (issue #46). `llm_enabled` is NOT
            # overridden: _setup_context reads the persisted row, so a validation
            # failure that fail-closes shows the checkbox OFF — the honest state —
            # rather than echoing the submitted intent as if it stuck (matching
            # /settings/llm).
            return templates.TemplateResponse(
                request,
                "setup.html",
                _setup_context(
                    request,
                    session,
                    WizardStep.LLM,
                    error=error,
                    llm_base_url=llm_base_url,
                    llm_model=llm_model,
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
                "ytdlp_enabled": resolve_effective_ytdlp_enabled(
                    get_app_settings(session), settings
                ),
                # Injected clock for the relative-age render (format_age(now=…)).
                "now": datetime.now(UTC),
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
        if not resolve_effective_ytdlp_enabled(get_app_settings(session), settings):
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
        island_props = _transcript_island_props(
            session, run_id, lines, palette, capability, settings
        )
        return templates.TemplateResponse(
            request,
            "transcript.html",
            {
                "request": request,
                "run": run,
                "lines": lines,
                "island_props": island_props,
                "palette": palette,
                "low_confidence_threshold": settings.review_low_confidence_threshold,
                "text": variant,
                "variants": list(TranscriptText),
                "active_nav": "runs",
            },
        )

    @protected.get("/review/{run_id}/transcript")
    def review_transcript(
        run_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        token: uuid.UUID | None = None,
    ) -> Response:
        """Claim-gated verify-and-advance surface (issue #53) + inline text
        correction (issue #58). Reuses the SAME linear player/props as the
        read-only transcript; the difference is the claim token (carried in
        ``?token=`` from the workbench, never re-acquired here — claims are
        takeover, so a fresh claim would kill the workbench tab) which unlocks the
        review-stepper island. A stale/absent token degrades to read-only with a
        prompt to claim, exactly like the workbench GET."""
        run = _run_or_404(session, run_id)
        if token is not None:
            try:
                verify_claim(session, run_id, token)
            except ClaimMismatchError:
                token = None  # stale tab: render read-only with a claim prompt
        # The review surface always shows operator-effective text (corrected →
        # enhanced → raw); raw evidence stays reachable on the read-only page.
        lines = attributed_transcript(session, run_id, text=TranscriptText.CORRECTED)
        settings: Settings = request.app.state.settings
        capability = playback_capability(session, run, settings, _get_media_gate(request))
        palette = speaker_palette(_run_label_universe(session, run_id))
        verified_n, total = verified_progress(session, run_id)
        island_props = _transcript_island_props(
            session, run_id, lines, palette, capability, settings
        )
        # Extras the review loop needs beyond the shared display props.
        island_props["reviewToken"] = str(token) if token is not None else None
        island_props["initialProgress"] = {"verified": verified_n, "total": total}
        # The assignable roster for the per-child reassign picker (issue #59 slice
        # 3): ACTIVE identities only — merged/archived speakers are curated out and
        # must not attract new rulings, mirroring the /relabel route's own
        # roster_is_active guard so the picker never offers a speaker the write
        # would reject. The read-only transcript-player never gets this (it cannot
        # relabel); only the claim-gated review-stepper needs a picker.
        island_props["speakers"] = [
            {"id": str(sp.id), "displayName": sp.display_name}
            for sp in active_speakers(session)
        ]
        return templates.TemplateResponse(
            request,
            "review_transcript.html",
            {
                "request": request,
                "run": run,
                "lines": lines,
                "island_props": island_props,
                "palette": palette,
                "low_confidence_threshold": settings.review_low_confidence_threshold,
                "token": token,
                "progress": {"verified": verified_n, "total": total},
                "csrf_claim": mint_csrf_token(request.app.state.csrf_secret, CSRF_CLAIM),
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
        # Whitelist the sort so a bookmarked/garbage value degrades to the
        # default FIFO order rather than erroring (issue #56).
        sort = request.query_params.get("sort") or "oldest"
        if sort not in QUEUE_SORTS:
            sort = "oldest"
        return templates.TemplateResponse(
            request,
            "queue.html",
            {
                "request": request,
                "entries": adjudication_queue(session, sort=sort),
                "sort": sort,
                # Injected clock for the relative-age render (format_age(now=…)).
                "now": datetime.now(UTC),
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
        # `inherit` is a segment-scope reset only — never a whole-label ruling
        # (the DB CHECK would otherwise reject it as a raw 500).
        if decision not in (Decision.ASSIGN, Decision.EXCLUDE, Decision.UNKNOWN):
            raise HTTPException(status_code=422, detail=f"invalid label action {action!r}")
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

    @protected.post("/review/{run_id}/merge/preview")
    def merge_preview(
        run_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        token: Annotated[uuid.UUID, Form()],
        labels: Annotated[list[str], Form()],
        target: Annotated[str, Form()],
        new_name: Annotated[str | None, Form()] = None,
    ) -> Response:
        """Server-computed impact of merging labels — reads only, writes nothing.

        The confirm step the operator sees before applying: the exact turns and
        segments the merge touches (never an advisory client count), plus the
        optimistic-concurrency token (each label's current effective ruling id)
        echoed into the confirm form so :func:`merge_apply` can reject a stale
        confirm. Claim-gated like every workbench mutation; JS-off never reaches
        here (the panel enhances progressively — see fragments/labels.html).

        ``target`` is the single unambiguous survivor chooser from the panel: the
        sentinel ``"new"`` (enroll ``new_name``) or an existing speaker's UUID.
        The confirm form it renders echoes the resolved speaker_id XOR
        display_name, so :func:`merge_apply` never has to disambiguate.
        """
        try:
            run = verify_claim(session, run_id, token)
        except ClaimMismatchError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        target_speaker: Speaker | None = None
        display_name: str | None = None
        if target == "new":
            display_name = (new_name or "").strip() or None
            if display_name is None:
                raise HTTPException(status_code=400, detail="enter a name for the new speaker")
        else:
            try:
                target_speaker = session.get(Speaker, uuid.UUID(target))
            except ValueError as exc:
                raise HTTPException(status_code=422, detail="choose a survivor speaker") from exc
            if target_speaker is None:
                raise HTTPException(status_code=400, detail="that speaker no longer exists")
        try:
            preview = preview_merge(session, run_id, labels)
        except MergeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        expected = json.dumps(
            {
                impact.label: (
                    str(impact.expected_decision_id)
                    if impact.expected_decision_id is not None
                    else None
                )
                for impact in preview.labels
            }
        )
        return templates.TemplateResponse(
            request,
            "fragments/merge_confirm.html",
            {
                "request": request,
                "run": run,
                "token": token,
                "nonce": lambda: uuid.uuid4().hex,
                "preview": preview,
                "target_speaker": target_speaker,
                "target_name": display_name,
                "expected_json": expected,
                "palette": speaker_palette(_run_label_universe(session, run_id)),
            },
        )

    @protected.post("/review/{run_id}/merge")
    def merge_apply(
        run_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        token: Annotated[uuid.UUID, Form()],
        nonce: Annotated[str, Form(min_length=8, max_length=64)],
        labels: Annotated[list[str], Form()],
        expected: Annotated[str, Form()],
        speaker_id: Annotated[uuid.UUID | None, Form()] = None,
        display_name: Annotated[str | None, Form()] = None,
    ) -> Response:
        """Rule that several labels are one speaker in this run — atomically.

        Run-local: records one assign ruling per label to a single survivor; it
        never calls the roster-wide merge_speakers. Under the claim lock it
        re-verifies the previewed rulings still hold (409 if they drifted) and
        appends every ruling in one transaction with deterministic child
        idempotency keys, so a replay returns the original outcome and a partial
        apply is impossible.
        """
        try:
            run = verify_claim(session, run_id, token, for_update=True)
        except ClaimMismatchError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        # An untouched text field posts "" not None; the confirm form omits the
        # unused target entirely, but normalise defensively so the XOR check sees
        # a clean None rather than an empty string.
        display_name = (display_name or "").strip() or None
        try:
            raw = json.loads(expected)
            if not isinstance(raw, dict):
                raise TypeError("expected-state must be a JSON object")
            expected_ids: dict[str, uuid.UUID | None] = {
                str(label): (uuid.UUID(value) if value is not None else None)
                for label, value in raw.items()
            }
        except (ValueError, AttributeError, TypeError) as exc:
            raise HTTPException(status_code=422, detail="malformed expected-state") from exc
        settings: Settings = request.app.state.settings
        try:
            apply_merge(
                session,
                run_id=run_id,
                labels=labels,
                operator=operator,
                nonce=nonce,
                gates=gates_from_settings(settings),
                target_speaker_id=speaker_id,
                target_name=display_name,
                expected=expected_ids,
            )
        except MergeConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ConflictingReplayError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except MergeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _labels_response(request, session, run, token)

    @protected.post("/review/{run_id}/segments/{segment_id}/relabel")
    def relabel_segment(
        run_id: uuid.UUID,
        segment_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        token: Annotated[uuid.UUID, Form()],
        nonce: Annotated[str, Form(min_length=8, max_length=64)],
        action: Annotated[str, Form()],
        speaker_id: Annotated[uuid.UUID | None, Form()] = None,
        start_word_index: Annotated[int | None, Form()] = None,
        end_word_index: Annotated[int | None, Form()] = None,
    ) -> Response:
        """Two-scope relabel, THIS-SEGMENT scope (issue #54 Phase B), optionally
        narrowed to a word-range (issue #59 slice 3).

        Overrides one transcript segment's attribution without touching the rest
        of its label. ``action`` is ``assign`` (a speaker just for this segment)
        or ``inherit`` (append-only reset: the segment follows its label's
        resolution again). The diarization label is derived from the segment row
        server-side, never trusted from the client. Claim-gated and idempotent
        like every workbench ruling; a later whole-label ruling leaves the
        override intact, and inherit tracks the label live rather than freezing.

        With ``start_word_index``/``end_word_index`` the ruling scopes just that
        half-open ``[start, end)`` word-range — reassigning ONE derived split
        child. The range must match a child that currently exists (validated
        against the segment's live cut set), so a ruling can only target a real
        partition, never an arbitrary span the read path would ignore. Both
        indices are set together or both omitted (whole-segment scope).
        """
        try:
            run = verify_claim(session, run_id, token, for_update=True)
        except ClaimMismatchError as exc:
            # Marked so the island can tell a lost claim from the segment-STATE 409
            # this route also raises (a non-child range): the picker treats a plain
            # 409 as a state conflict, but a claim loss must stop the loop.
            raise HTTPException(
                status_code=409, detail=str(exc), headers=_CLAIM_CONFLICT_HEADERS
            ) from exc
        try:
            decision = Decision(action)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"unknown action {action!r}") from exc
        if decision not in (Decision.ASSIGN, Decision.INHERIT):
            raise HTTPException(
                status_code=422, detail="segment scope allows only assign or inherit"
            )
        if (decision is Decision.ASSIGN) != (speaker_id is not None):
            raise HTTPException(
                status_code=422, detail="assign requires speaker_id; inherit forbids it"
            )
        if (start_word_index is None) != (end_word_index is None):
            raise HTTPException(
                status_code=422,
                detail="start_word_index and end_word_index must be set together",
            )
        segment = session.get(TranscriptSegment, segment_id)
        if segment is None or segment.pipeline_run_id != run_id:
            raise HTTPException(status_code=404, detail="no such segment in this run")
        if segment.diarization_label is None:
            raise HTTPException(
                status_code=400, detail="segment has no diarization label to override"
            )
        if start_word_index is not None and end_word_index is not None:
            # A ranged ruling may only target a child that exists right now, so it
            # can never write a row the read path silently drops.
            child_ranges = _segment_child_ranges(session, segment)
            if (start_word_index, end_word_index) not in child_ranges:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"word-range [{start_word_index}, {end_word_index}) is not a "
                        "current split child of this segment; split it at that "
                        "boundary first"
                    ),
                )
        if speaker_id is not None:
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
        try:
            record_decision(
                session,
                pipeline_run_id=run_id,
                diarization_label=segment.diarization_label,
                decision=decision,
                operator=operator,
                idempotency_key=nonce,
                speaker_id=speaker_id,
                transcript_segment_id=segment_id,
                start_word_index=start_word_index,
                end_word_index=end_word_index,
            )
        except ConflictingReplayError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except WordRangeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        # The island (JSON Accept) reassigns a split child and reconciles against a
        # whole-run render — a child's speaker string moved, which the per-segment
        # review shape can't carry. The htmx labels workbench (and any non-island
        # caller) keeps the byte-identical server-rendered fragment.
        if _wants_island_json(request):
            return _run_reconcile_response(session, run_id)
        return _labels_response(request, session, run, token)

    def _segment_review_json(
        session: Session, run_id: uuid.UUID, segment: TranscriptSegment
    ) -> JSONResponse:
        """The state a triage-loop write returns to the island: this segment's
        verified/corrected flags + effective text, and the run's N-of-M counter."""
        row = session.get(SegmentReviewState, segment.id)
        corrected = row.corrected_text if row is not None else None
        verified_n, total = verified_progress(session, run_id)
        return JSONResponse(
            {
                "segmentId": str(segment.id),
                "verified": row is not None and row.verified_at is not None,
                "corrected": corrected is not None,
                "text": effective_text(segment, corrected),
                "progress": {"verified": verified_n, "total": total},
            }
        )

    @protected.post("/review/{run_id}/segments/{segment_id}/verify")
    def verify_segment(
        run_id: uuid.UUID,
        segment_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        token: Annotated[uuid.UUID, Form()],
        verified: Annotated[bool, Form()] = True,
    ) -> Response:
        """Mark (or unmark) a segment verified — the verify-and-advance step
        (issue #53). Claim-gated; a mutable UPSERT, so idempotent without a nonce.
        The island (fetch) gets the updated state + N-of-M progress as JSON; a
        JS-off browser form navigation is redirected back to the review page, so
        the server-rendered fallback verifies for real (degrade-to-plain-HTML)."""
        try:
            verify_claim(session, run_id, token, for_update=True)
        except ClaimMismatchError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        segment = session.get(TranscriptSegment, segment_id)
        if segment is None or segment.pipeline_run_id != run_id:
            raise HTTPException(status_code=404, detail="no such segment in this run")
        set_verified(session, segment=segment, verified=verified)
        if _wants_html(request):
            return RedirectResponse(
                f"/review/{run_id}/transcript?token={token}", status_code=303
            )
        return _segment_review_json(session, run_id, segment)

    @protected.post("/review/{run_id}/segments/{segment_id}/text")
    def correct_segment(
        run_id: uuid.UUID,
        segment_id: uuid.UUID,
        operator: OperatorDep,
        session: SessionDep,
        token: Annotated[uuid.UUID, Form()],
        text: Annotated[str, Form(max_length=MAX_CORRECTED_TEXT_CHARS)] = "",
    ) -> JSONResponse:
        """Set or clear the operator's corrected text for a segment (issue #58).
        Empty text, or text equal to the pipeline rendering, reverts to no
        correction. Editing clears the segment's verified mark in the same
        transaction. Claim-gated; the corrected text is written beside raw_text,
        never over it (raw stays the immutable ASR evidence)."""
        try:
            verify_claim(session, run_id, token, for_update=True)
        except ClaimMismatchError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        segment = session.get(TranscriptSegment, segment_id)
        if segment is None or segment.pipeline_run_id != run_id:
            raise HTTPException(status_code=404, detail="no such segment in this run")
        # A split parent renders as word-derived children, which free-form
        # corrected text cannot be partitioned across (issue #59, deferred). Refuse
        # correcting a split segment — the mirror of forbidding a split on an
        # already-corrected segment, so the two never coexist.
        if _segment_is_split(session, segment_id):
            raise HTTPException(
                status_code=409,
                detail="cannot correct a split segment; remove the split first",
            )
        set_correction(session, segment=segment, text=text)
        return _segment_review_json(session, run_id, segment)

    @protected.post("/review/{run_id}/segments/{segment_id}/split")
    def split_segment(
        run_id: uuid.UUID,
        segment_id: uuid.UUID,
        operator: OperatorDep,
        session: SessionDep,
        token: Annotated[uuid.UUID, Form()],
        word_index: Annotated[int, Form()],
    ) -> JSONResponse:
        """Split a segment at a word boundary (issue #59), inserting one cut
        "before word ``word_index``". Claim-gated and structurally idempotent (the
        boundary's UNIQUE key makes a replayed split a no-op — no nonce needed).

        Refuses a corrected segment (mutually exclusive with correction) and an
        unsplittable one (no aligned word timings, or materially-enhanced text) with
        the operator-facing reason. Returns the run's re-rendered island segments so
        the console reconciles against server truth — the same shape as hydration,
        with the parent now expanded into its derived children."""
        try:
            verify_claim(session, run_id, token, for_update=True)
        except ClaimMismatchError as exc:
            # Marked so the island distinguishes a lost claim from the segment-STATE
            # 409 this route also raises (already-split / corrected): the split
            # handler treats a plain 409 as a state conflict, a claim loss must stop.
            raise HTTPException(
                status_code=409, detail=str(exc), headers=_CLAIM_CONFLICT_HEADERS
            ) from exc
        segment = session.get(TranscriptSegment, segment_id)
        if segment is None or segment.pipeline_run_id != run_id:
            raise HTTPException(status_code=404, detail="no such segment in this run")
        row = session.get(SegmentReviewState, segment_id)
        if row is not None and row.corrected_text is not None:
            raise HTTPException(
                status_code=409,
                detail="cannot split a corrected segment; clear the correction first",
            )
        # This release supports a SINGLE cut per parent (two children). A second,
        # DISTINCT cut would re-derive the children and orphan any word-range
        # reassignment keyed on the old child coordinates — a written ruling the
        # read path then silently ignores (issue #59 slice 3). The UI already
        # disables further splits, but a second tab sharing the claim could still
        # POST one; refuse it server-side. A replay of the EXISTING cut still falls
        # through to record_split's idempotent no-op (same word_index), so /split
        # stays idempotent.
        existing_cuts = {
            wi
            for (wi,) in session.execute(
                select(SegmentSplitBoundary.word_index).where(
                    SegmentSplitBoundary.parent_segment_id == segment_id
                )
            )
        }
        if existing_cuts and word_index not in existing_cuts:
            raise HTTPException(
                status_code=409,
                detail=(
                    "segment is already split; splitting into more than two parts is "
                    "not supported in this release — re-transcribe to clear the split"
                ),
            )
        try:
            record_split(
                session, parent=segment, word_index=word_index, operator=operator
            )
        except UnsplittableError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _run_reconcile_response(session, run_id)

    @protected.get("/review/{run_id}/segments/{segment_id}/words")
    def segment_words(
        run_id: uuid.UUID,
        segment_id: uuid.UUID,
        operator: OperatorDep,
        session: SessionDep,
    ) -> JSONResponse:
        """The active segment's word tokens for the split UI (issue #59), fetched
        LAZILY only when the operator enters split mode — never bloating the shared
        read payload with every run's words. Reports ``splittable`` (+ a reason when
        not) so the console shows an honest disabled affordance rather than a
        split that would fail."""
        segment = session.get(TranscriptSegment, segment_id)
        if segment is None or segment.pipeline_run_id != run_id:
            raise HTTPException(status_code=404, detail="no such segment in this run")
        words = splittable_words(segment)
        if words is None:
            corrected = _segment_is_corrected(session, segment_id)
            reason = (
                "this segment has an operator correction; clear it to split"
                if corrected
                else "no aligned word timings for this segment (or its text was enhanced)"
            )
            return JSONResponse(
                {"segmentId": str(segment_id), "splittable": False, "reason": reason, "words": []}
            )
        if _segment_is_corrected(session, segment_id):
            return JSONResponse(
                {
                    "segmentId": str(segment_id),
                    "splittable": False,
                    "reason": "this segment has an operator correction; clear it to split",
                    "words": [],
                }
            )
        return JSONResponse(
            {
                "segmentId": str(segment_id),
                "splittable": True,
                "reason": None,
                "words": [
                    {"start": w.start, "end": w.end, "word": w.text} for w in words
                ],
            }
        )

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
        if not resolve_effective_enrichment_names_enabled(get_app_settings(session), settings):
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
        if not resolve_effective_enrichment_names_enabled(get_app_settings(session), settings):
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
    # (it reads diarization turns, not attributed lines). All accept
    # ?text=corrected|enhanced|raw (default corrected: operator corrections applied
    # over enhanced/raw; enhanced = pipeline text, no corrections; raw = immutable
    # ASR evidence), except RTTM which is speaker-label-only.
    def _export_transcript(
        run_id: uuid.UUID,
        session: Session,
        fmt: TranscriptFormat,
        text: str | None,
        *,
        timestamps: bool = True,
    ) -> Response:
        _run_or_404(session, run_id)
        try:
            variant = parse_transcript_text(text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        lines = attributed_transcript(session, run_id, text=variant)
        return Response(
            content=render_transcript(lines, fmt, timestamps=timestamps),
            media_type=MEDIA_TYPES[fmt.value],
        )

    @protected.get("/review/{run_id}/export.txt")
    def export_transcript_txt(
        run_id: uuid.UUID,
        operator: OperatorDep,
        session: SessionDep,
        text: str | None = None,
        timestamps: bool = True,
    ) -> Response:
        # ?timestamps=false drops the [start end] bracket column for a clean
        # reading copy (issue #52). TXT is the only format where the flag applies.
        return _export_transcript(
            run_id, session, TranscriptFormat.TXT, text, timestamps=timestamps
        )

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
        comes from — never the key value — plus the Features-section tri-state flag
        rows (issue #62). ``overrides`` lets a POST re-render carry a section
        ``*_error`` and (for Features) ``features_submitted``, the operator's
        submitted radio selections, so a rejected save re-renders their choices.
        """
        settings: Settings = request.app.state.settings
        tutorial_run = ready_tutorial_run_id(session)
        row = get_app_settings(session)
        base_value, base_default, model_value, model_default = llm_endpoint_form_fields(
            row, settings
        )
        # Features section (issue #62): one tri-state row per live-read flag. On an
        # invariant-rejected save, render the operator's submitted choices back
        # (``features_submitted``); otherwise render the stored raw tri-state.
        features_submitted: dict[str, str] | None = overrides.pop("features_submitted", None)
        feature_flags = [
            {
                "name": name,
                "label": label,
                "help": help_text,
                "state": (
                    features_submitted.get(name, "inherit")
                    if features_submitted is not None
                    else feature_flag_state(row, name)
                ),
                "env_default": bool(getattr(settings, name)),
            }
            for name, label, help_text in _FEATURE_FLAG_META
        ]
        context: dict[str, Any] = {
            "request": request,
            "tutorial_available": tutorial_run is not None,
            "tutorial_run_id": tutorial_run,
            "tutorial_completed_at": row.tutorial_completed_at if row else None,
            "llm_enabled": bool(row.llm_enabled) if row is not None else settings.llm_enabled,
            # Endpoint OVERRIDE (blank when inheriting env), env default as
            # placeholder — the tri-state render that keeps an untouched save from
            # pinning the env value onto the row (issue #46).
            "llm_base_url": base_value,
            "llm_base_url_default": base_default,
            "llm_model": model_value,
            "llm_model_default": model_default,
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
            "feature_flags": feature_flags,
            "features_errors": [],
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

    @protected.post("/settings/features")
    def settings_features(
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        enrichment_names_enabled: Annotated[str, Form()] = "inherit",
        enrichment_names_llm_enabled: Annotated[str, Form()] = "inherit",
        enrichment_run_assets_enabled: Annotated[str, Form()] = "inherit",
        enrichment_run_assets_autogenerate: Annotated[str, Form()] = "inherit",
        ytdlp_enabled: Annotated[str, Form()] = "inherit",
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> Response:
        _require_csrf(request, CSRF_SETTINGS, csrf_token)
        settings: Settings = request.app.state.settings
        submitted = {
            "enrichment_names_enabled": enrichment_names_enabled,
            "enrichment_names_llm_enabled": enrichment_names_llm_enabled,
            "enrichment_run_assets_enabled": enrichment_run_assets_enabled,
            "enrichment_run_assets_autogenerate": enrichment_run_assets_autogenerate,
            "ytdlp_enabled": ytdlp_enabled,
        }
        # Candidate → validate (shared invariants) → ONE mutation. On an invariant
        # violation nothing is written and the operator's choices are re-rendered
        # with the plain-language message(s) (issue #62).
        errors = _persist_feature_flags(session, settings, submitted=submitted)
        if errors:
            return templates.TemplateResponse(
                request,
                "settings.html",
                _settings_context(
                    request, session, features_errors=errors, features_submitted=submitted
                ),
            )
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

    @protected.get("/media/{run_id}/peaks")
    def media_peaks(
        run_id: uuid.UUID, request: Request, operator: OperatorDep, session: SessionDep
    ) -> Response:
        """Waveform amplitude envelope (issue #57): lazy compute, cached artifact.

        Cache trust: while the WAV is live, the cached row's source fingerprint
        is fstat-verified (prepare replaces the WAV before its DB commit, so a
        crash can strand a stale row) and a mismatch recomputes. A reclaimed WAV
        cannot be verified — the cache is served as-is so a static waveform
        still renders (derived evidence, like the transcript itself survives
        reclamation). No cache + no servable WAV answers with /media's honest
        status (410 reclaimed, 404 missing/unservable); a WAV that cannot yield
        trustworthy peaks answers 404 and caches nothing (fail closed).
        """
        settings = request.app.state.settings
        _run_or_404(session, run_id)
        media_root: Path = settings.media_root

        # Fast path: a trusted cache hit, no lock.
        cached = load_cached_peaks(session, run_id, media_root)
        if cached is not None and _peaks_cache_trusted(
            session, run_id, media_root, cached
        ):
            return _peaks_cache_response(request, cached.body, cached.artifact_id)

        # Slow path: serialize compute+publish per run with a transaction-scoped
        # advisory lock, so two first-paint tabs (possibly straddling a prepare
        # re-run) can never publish divergent bytes under one row/ETag. Held
        # until this request's transaction ends (the commit below). Key: a
        # stable 63-bit digest of the run UUID — same run, same lock, any worker.
        lock_key = run_id.int & 0x7FFFFFFFFFFFFFFF
        session.execute(select(func.pg_advisory_xact_lock(lock_key)))

        # Re-check under the lock: a racing request may have just populated the
        # cache while we waited, and its committed row is the canonical one.
        cached = load_cached_peaks(session, run_id, media_root)
        if cached is not None and _peaks_cache_trusted(
            session, run_id, media_root, cached
        ):
            return _peaks_cache_response(request, cached.body, cached.artifact_id)

        gate = _get_media_gate(request)
        try:
            fh, _size = resolve_servable_media(session, run_id, settings, gate)
        except MediaResolutionError as exc:
            raise HTTPException(
                status_code=exc.http_status, detail=f"waveform unavailable: {exc}"
            ) from exc
        try:
            fingerprint = SourceFingerprint.of_descriptor(fh)
            try:
                payload = compute_peaks(fh)
            except PeaksError as exc:
                raise HTTPException(
                    status_code=404, detail=f"waveform unavailable: {exc}"
                ) from exc
        finally:
            fh.close()
        row_id = store_peaks(session, run_id, media_root, payload, fingerprint)
        session.commit()
        return _peaks_cache_response(request, payload.to_json_bytes(), row_id)

    # Mount the gated routes last: every @protected route above is now attached to
    # `app` behind require_onboarded, while the @app routes stay exempt.
    app.include_router(protected)


def _peaks_cache_trusted(
    session: Session, run_id: uuid.UUID, media_root: Path, cached: CachedPeaks
) -> bool:
    """Whether a cached envelope may be served for this run.

    Fail-closed: trusted ONLY when the WAV is formally reclaimed (nothing left
    to verify against — the static waveform is honest derived evidence) OR the
    live WAV's fstat fingerprint still matches what the peaks were computed
    from. Any other live-media state — no resolvable preprocessed row, a stat
    failure, a malformed stored fingerprint — is untrusted, so the route falls
    through to ``resolve_servable_media`` and answers the honest 404/410 (or
    recomputes) rather than serving unverified bytes.
    """
    if run_intermediate_reclaimed_at(session, run_id) is not None:
        return True
    try:
        wav_path = normalized_audio_path(session, run_id, media_root)
    except StageDataError:
        return False
    live = SourceFingerprint.of_path(wav_path)
    return live is not None and live == cached.source_fingerprint


def _peaks_cache_response(request: Request, body: bytes, artifact_id: uuid.UUID) -> Response:
    """Serve the cached envelope with real conditional-GET semantics.

    The artifact row UUID is a valid strong ETag because a row's bytes are
    immutable once published — every refresh (re-prepare, stale-fingerprint
    recompute) mints a new row or rewrites meta+file together before this is
    called. ``no-cache`` forces revalidation, so a requeued run can never show
    a stale waveform from the browser cache; the 304 makes that revalidation
    cost one conditional GET.

    ``If-None-Match`` uses the weak comparison RFC 9110 §13.1.2 mandates, so a
    ``W/"<uuid>"`` validator from an intermediary still matches our strong tag.
    """
    etag = f'"{artifact_id}"'
    headers = {"ETag": etag, "Cache-Control": "private, no-cache"}
    if_none_match = request.headers.get("if-none-match")
    if if_none_match is not None:
        # Strip an optional weak-validator prefix before comparing (weak match).
        candidates = {
            v.strip()[2:] if v.strip().startswith('W/') else v.strip()
            for v in if_none_match.split(",")
        }
        if "*" in candidates or etag in candidates:
            return Response(status_code=304, headers=headers)
    return Response(content=body, media_type="application/json", headers=headers)


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
