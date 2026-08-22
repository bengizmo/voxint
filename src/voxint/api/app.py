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
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, BinaryIO, cast
from urllib.parse import urlencode

import httpx
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.templating import Jinja2Templates
from sqlalchemy import Engine, exists, func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.datastructures import MutableHeaders
from starlette.requests import ClientDisconnect
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from voxint import __version__
from voxint.adjudication.annotations import (
    AnnotationError,
    AnnotationIdempotencyError,
    AnnotationNotFoundError,
    AnnotationStaleError,
    AnnotationTagConflictError,
    AnnotationValidationError,
    CaptureEndpoint,
    CapturePayload,
    ResolvedAnnotation,
    annotations_for_run,
    capture_annotation,
    clip_lines_for_export,
    create_tag,
    derive_live_anchor,
    list_tags,
    live_annotation_or_404,
    load_covered_segments,
    normalize_note,
    reanchor_annotation,
    refresh_annotation,
    resolve_annotation_spans,
    resolve_tag_names,
    resolved_order_key,
    soft_delete_annotation,
    stored_anchor_from_derived,
    stored_anchor_from_row,
    tags_for_annotations,
    update_annotation,
    update_tag,
)
from voxint.adjudication.corrections_view import (
    DeclaredRuleIndex,
    build_declared_rule_index,
    resolve_segment_provenance,
    run_reconciliation,
)
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
    review_backlog_count,
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
    trace_has_entries,
)
from voxint.adjudication.transcript import (
    TranscriptLine,
    TranscriptText,
    attributed_transcript,
    effective_text,
    paragraphize_transcript,
    parse_transcript_text,
)
from voxint.api.auth import require_operator
from voxint.api.csrf import (
    CSRF_ANNOTATION_TAGS,
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
from voxint.api.meaning_query import search_passages
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
    title_from_snapshot,
)
from voxint.api.resource_status import (
    ResourceSnapshot,
    build_resource_strip,
    collect_resource_status,
    device_state,
    gib,
    render_resource_prometheus,
    short_uuid,
    vram_percent,
)
from voxint.api.runs_query import (
    Cursor,
    InvalidCursorError,
    ReviewFilter,
    latest_completed_run,
    list_runs,
    parse_review_filter,
    parse_search_filters,
    parse_status_filter,
    runs_url,
)
from voxint.api.setup_wizard import (
    MAX_MEDIA_FOLDERS,
    STEP_ORDER,
    ScanResult,
    SetupValidationError,
    WizardStep,
    list_media_subdirs,
    next_step,
    normalize_llm_api_key,
    normalize_llm_base_url,
    normalize_llm_model,
    normalize_media_folders,
    normalize_vocabulary,
    normalize_web_search_api_key,
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
from voxint.app_settings import SINGLETON_ID as APP_SETTINGS_SINGLETON_ID
from voxint.app_settings import (
    EffectiveFlags,
    clear_tutorial_completion,
    complete_onboarding,
    effective_llm_key_source,
    effective_web_search_key_source,
    feature_flag_state,
    get_app_settings,
    get_or_create,
    is_onboarded,
    llm_bundled_active,
    llm_endpoint_form_fields,
    mark_tutorial_complete,
    ready_tutorial_run_id,
    resolve_effective_enrichment_names_enabled,
    resolve_effective_enrichment_names_llm_enabled,
    resolve_effective_enrichment_run_assets_autogenerate,
    resolve_effective_enrichment_run_assets_enabled,
    resolve_effective_enrichment_web_research_enabled,
    resolve_effective_llm_api_key,
    resolve_effective_llm_enabled,
    resolve_effective_source_authority_domains,
    resolve_effective_voxint_web_research,
    resolve_effective_watch_folder_enabled,
    resolve_effective_web_search_api_key,
    resolve_effective_web_search_base_url,
    resolve_effective_ytdlp_enabled,
    semantic_index_flags_ok,
    str_flag_form_field,
    validate_effective_flags,
    validate_web_search_base_url,
)
from voxint.config import Settings, get_settings, llm_budget_fits_stage_lease
from voxint.db.models import (
    HIGHLIGHT_PALETTE_SIZE,
    MAX_ANNOTATION_NOTE_CHARS,
    MAX_ANNOTATION_QUOTE_CHARS,
    MAX_ANNOTATION_SPAN_SEGMENTS,
    MAX_CORRECTED_TEXT_CHARS,
    MAX_TAG_NAME_CHARS,
    MAX_TAGS_PER_ANNOTATION,
    AnnotationTag,
    AppSettings,
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
    Stage,
    StageRun,
    TranscriptAnnotation,
    TranscriptSegment,
)
from voxint.db.session import build_engine, build_session_factory, session_scope
from voxint.diagnostics import check_state, run_diagnostics
from voxint.domain_packs.base import DomainPackError
from voxint.domain_packs.corrections import (
    MAX_MATCH_CHARS,
    MAX_REPLACEMENT_CHARS,
    MAX_RULES_PER_PACK,
    OperatorCorrectionError,
    normalize_operator_corrections,
    operator_correction_message,
)
from voxint.domain_packs.registry import available_domain_packs, default_domain_pack
from voxint.embeddings.onnx_embedder import minilm_artifacts_available
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
    validate_authority_domains,
)
from voxint.enrichment.triage import (
    score as triage_score,
)
from voxint.export import (
    ANNOTATION_BULK_SEPARATOR,
    ANNOTATION_MEDIA_TYPES,
    MEDIA_TYPES,
    TranscriptFormat,
    annotation_pull_quote,
    format_timespan,
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
from voxint.media.source_metadata import RAW_URL_KEYS
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
from voxint.tutorial.seed import seed_tutorial_run
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

# Bounded, non-secret operator guidance for a failed UI-triggered tutorial seed
# (issue #75). At most two messages: a storage failure vs. broken/missing bundled
# data. The real exception is logged server-side; the operator never sees a path
# or traceback.
_TUTORIAL_SEED_STORAGE_ERROR = (
    "Could not set up the tutorial: the media folder is not writable. "
    "Check that the app can write to its media directory, then try again."
)
_TUTORIAL_SEED_ASSET_ERROR = (
    "Could not set up the tutorial: its bundled sample data is missing or "
    "unreadable. This is likely a broken installation — reinstall or reach out "
    "for help."
)


def _try_seed_tutorial(
    session: Session, settings: Settings
) -> tuple[uuid.UUID | None, str | None]:
    """Seed the tutorial, mapping known environment/asset failures to bounded copy.

    Returns ``(run_id, None)`` on success or ``(None, message)`` on a classified
    failure. Only filesystem-storage and bundled-resource failures are caught —
    programmer/DB/builder defects (SQLAlchemy errors, ``TutorialSeedError`` from a
    fixture-invariant bug) propagate so they are never masked as tutorial copy
    (issue #75): a builder bug is not fixed by "reinstalling", so it must surface
    loudly, not be dressed up as missing data. The real exception is logged
    server-side; the message is path-free and traceback-free. The CALLER must
    ``session.rollback()`` before re-rendering — the request session commits on any
    successful response.

    Classification is by exception *type*, not origin — a v1 approximation. In the
    rare cross case (e.g. a ``PermissionError`` reading bundled resources) the
    operator gets the storage message; acceptable for a single-operator install and
    the true exception is always in the server log.
    """
    try:
        run_id = seed_tutorial_run(
            session, media_root=settings.media_root, settings=settings
        )
    # FileNotFoundError MUST precede the OSError clause below — it is an OSError
    # subclass, and a missing bundled asset is a data problem, not a storage one.
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        logger.exception("Tutorial seed failed: bundled sample data missing/unreadable")
        return None, _TUTORIAL_SEED_ASSET_ERROR
    except OSError:
        logger.exception("Tutorial seed failed: media folder not writable")
        return None, _TUTORIAL_SEED_STORAGE_ERROR
    return run_id, None


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
    """Bound request-body reception at ``max_bytes`` regardless of Content-Length.

    FastAPI parses a multipart body (spooling file parts to a temp) *before* a
    route's dependencies run, so a per-route check cannot gate body reception —
    by the time the handler executes, the whole body is already spooled. This ASGI
    middleware bounds spooling in two layers:

    * **Fast path** — an *honestly-declared* over-cap ``Content-Length`` is refused
      with 413 before Starlette reads a single body byte.
    * **Streaming cap (finding D3)** — the middleware also wraps ``receive`` and
      counts body bytes as they arrive, so a chunked request with **no**
      ``Content-Length`` (or an understated one) is cut off the moment the running
      total exceeds ``max_bytes`` — it can no longer be fully multipart-spooled
      first. On overflow it truncates the stream to the app (an injected
      ``http.disconnect`` → the parser unwinds) and emits a single bare 413,
      swallowing the app's own response so exactly one response is sent.

    The authoritative per-file cap still runs while streaming in ``submit_upload``.
    Not addressed here (deliberately, on a loopback single-operator console):
    moving Basic auth *ahead* of body parsing, which a per-route ``OperatorDep``
    cannot given FastAPI's dispatch order — an unauthenticated over-cap body is
    still bounded by this cap, just not rejected with 401 before the bounded read.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        # Fast path: refuse an honestly-declared over-cap body before any read.
        for name, value in scope["headers"]:
            if name != b"content-length":
                continue
            try:
                if int(value) > self._max_bytes:
                    await self._reject(send)
                    return
            except ValueError:
                pass  # unparseable → the streaming cap below is authoritative
            break

        received = 0
        rejecting = False  # committed to sending our own 413 in place of the app's
        real_response_started = False

        async def counting_receive() -> Message:
            nonlocal received, rejecting
            if rejecting:
                # Keep the app unwinding; do not pull more real body.
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self._max_bytes and not real_response_started:
                    rejecting = True
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal real_response_started
            if rejecting:
                return  # our 413 stands in for the app's response
            if message["type"] == "http.response.start":
                real_response_started = True
            await send(message)

        try:
            await self._app(scope, counting_receive, guarded_send)
        except ClientDisconnect:
            # Expected: truncating the over-cap body makes the body reader raise.
            if not rejecting:
                raise
        except Exception:
            # Once we have committed to rejecting we have already swallowed the
            # app's output, so any exception it raises while unwinding the
            # truncated stream (a parser error, not just ClientDisconnect) must
            # not escape — the 413 below is the single response. Only re-raise a
            # genuine error from a request we were NOT capping.
            if not rejecting:
                raise
            logger.warning(
                "over-cap request body: app raised while unwinding the truncated "
                "stream; returning 413",
                exc_info=True,
            )
        if rejecting:
            await self._reject(send)

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


class _SecurityHeadersMiddleware:
    """Inject the console's conservative baseline security headers.

    The per-response policy lives in ``_apply_security_headers``; this middleware
    stamps it on the ``http.response.start`` of every response. The bulk of the
    policy contains the URL-borne claim token (finding D1).

    The review console carries the per-claim token in the URL (``?token=``); that
    token is *both* the review lock and the CSRF defense for claim-gated mutations.
    Two cheap headers contain its disclosure surface:

    * ``Referrer-Policy: no-referrer`` on **every** response — without it, a
      followed link or a cross-origin subresource on a token-bearing page could
      leak the token in the ``Referer`` header, the one exploitable disclosure
      vector under this threat model. ``no-referrer`` suppresses ``Referer`` on
      every navigation and subresource fetch.
    * ``Cache-Control: no-store`` on every ``/review`` response — the review pages
      embed the token (hidden form fields, island props) and the claim/mutation
      redirects carry it in ``Location``; stamping it centrally guarantees no
      token-bearing review response is ever written to a browser/proxy cache, and
      cannot be missed when a new ``/review`` route is added.

    This is a *mitigation*, not token removal: the token still appears in browser
    history, address-bar screenshots, and server access logs. That residual is
    consciously accepted for a single-operator, loopback-bound console — carrying
    the token in a cookie/session would add disproportionate machinery. See
    docs/security/audit-2026-08-18.md (D1). A third baseline header,
    ``X-Content-Type-Options: nosniff`` (issue #103), rides the same seam. All use
    ``setdefault`` so a route that needs a stricter/looser policy can still
    override them.
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        review_path = scope.get("path", "").startswith("/review")

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                _apply_security_headers(
                    MutableHeaders(scope=message), review_path=review_path
                )
            await send(message)

        await self._app(scope, receive, send_with_headers)


def _apply_security_headers(headers: MutableHeaders, *, review_path: bool) -> None:
    """Stamp the console's baseline security headers (idempotent via ``setdefault``).

    Shared by ``_SecurityHeadersMiddleware`` and the 500 handler so the policy has
    one definition.

    * ``Referrer-Policy: no-referrer`` and ``Cache-Control: no-store`` (on
      ``/review``) contain the URL-borne claim token (finding D1).
    * ``X-Content-Type-Options: nosniff`` on **every** response (issue #103): the
      console serves user-controlled transcript text as downloadable exports
      (``.txt``/``.md``/``.srt``/``.vtt``/``.json``/``.rttm``) and the built
      frontend bundle as first-party assets. ``nosniff`` forces the browser to
      honour the server-declared ``Content-Type`` instead of sniffing the bytes,
      so a transcript carrying crafted markup cannot be reinterpreted as HTML and
      executed. Every asset the Vite build emits already resolves to a correct
      type in ``_APP_ASSET_MEDIA_TYPES``, so this blocks nothing legitimate.
    """
    headers.setdefault("referrer-policy", "no-referrer")
    headers.setdefault("x-content-type-options", "nosniff")
    if review_path:
        headers.setdefault("cache-control", "no-store")


async def _security_headers_on_error(request: Request, exc: Exception) -> Response:
    """Re-apply the D1 headers to an unhandled-exception 500.

    Starlette's ``ServerErrorMiddleware`` wraps the whole app *outside* the
    user-added ``_SecurityHeadersMiddleware``, so a truly-unhandled exception's
    500 would otherwise skip the header stamp. Registering this as the ``Exception``
    handler closes that gap. ``ServerErrorMiddleware`` still re-raises after sending
    this response, so it does not mask exceptions from the server or tests."""
    response = Response("Internal Server Error", status_code=500, media_type="text/plain")
    _apply_security_headers(
        response.headers, review_path=request.url.path.startswith("/review")
    )
    return response


def _export_raw_host_only(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    """Reduce the URL keys in an exported ``raw`` snapshot to host-only (D4).

    ``raw`` is an allowlisted scalar subset — only ``RAW_URL_KEYS`` hold URLs — so
    those are reduced to a bare host (matching the run-detail provenance policy)
    while every other key passes through unchanged."""
    if raw is None:
        return None
    return {
        key: (provenance_host(value) if key in RAW_URL_KEYS else value)
        for key, value in raw.items()
    }


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
    # Default Referrer-Policy on every response (+ no-store on /review) so a
    # token-bearing review URL never leaks in a Referer header (finding D1). Added
    # last → outermost among user middleware, so it decorates normal responses and
    # redirects; the Exception handler below covers unhandled 500s, which Starlette
    # generates OUTSIDE this middleware.
    app.add_middleware(_SecurityHeadersMiddleware)
    app.add_exception_handler(Exception, _security_headers_on_error)
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


def _publish_run(run_id: uuid.UUID, *, stage: Stage | None = None) -> None:
    """Enqueue a fresh submission or stage-routed resumption (commit-before-publish).

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
    from voxint.worker.tasks import pipeline_task_for_stage

    pipeline_task_for_stage(stage).apply_async((str(run_id),), ignore_result=True)


def _publish_or_defer(run_id: uuid.UUID, *, stage: Stage | None = None) -> bool:
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
        _publish_run(run_id, stage=stage)
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


def _submit_domain_pack_detail(exc: DomainPackError) -> str:
    """Operator-facing text for a freeze-time ``DomainPackError`` at a submit boundary.

    ``_run_domain_pack_snapshot`` raises this when the run's resolved domain pack
    can't be applied — either an unresolvable pack name (issue #11) or an operator
    correction that collides with the folder's pack (issue #84). Neutral wording
    covers both, and the underlying reason is softened out of pack jargon so a
    non-technical operator can act on it instead of seeing a 500."""
    return (
        "The domain pack for this media couldn't be applied: "
        f"{operator_correction_message(str(exc))} "
        "Check Settings → Corrections and the folder's domain-pack assignment."
    )


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
# Hardware-telemetry display helpers (W3): bytes -> GiB number, used-VRAM %.
templates.env.globals["gib"] = gib
templates.env.globals["vram_percent"] = vram_percent
templates.env.globals["short_uuid"] = short_uuid
templates.env.globals["device_state"] = device_state


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

# Annotation-layer 409 markers (issue #86), mirroring _CLAIM_CONFLICT_HEADERS so
# the console can tell a stale-quote/anchor conflict, a replayed-nonce idempotency
# conflict, and a duplicate tag name apart from a lost claim. The taxonomy lives in
# docs/annotations.md ("API surface and error taxonomy").
_ANNOTATION_STALE_HEADERS = {"X-Voxint-Conflict": "stale"}
_ANNOTATION_IDEMPOTENCY_HEADERS = {"X-Voxint-Conflict": "idempotency"}
_ANNOTATION_TAG_CONFLICT_HEADERS = {"X-Voxint-Conflict": "duplicate-tag"}


def _annotation_http_error(exc: AnnotationError) -> HTTPException:
    """Map an annotation-domain error to its HTTP shape (docs/annotations.md error
    taxonomy). 422 validation, 404 not-found (fail closed — forged is not
    distinguished from missing), and three distinctly-marked 409s."""
    if isinstance(exc, AnnotationValidationError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, AnnotationNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, AnnotationStaleError):
        return HTTPException(status_code=409, detail=str(exc), headers=_ANNOTATION_STALE_HEADERS)
    if isinstance(exc, AnnotationIdempotencyError):
        return HTTPException(
            status_code=409, detail=str(exc), headers=_ANNOTATION_IDEMPOTENCY_HEADERS
        )
    if isinstance(exc, AnnotationTagConflictError):
        return HTTPException(
            status_code=409, detail=str(exc), headers=_ANNOTATION_TAG_CONFLICT_HEADERS
        )
    # Defensive: an unmapped AnnotationError is a validation-class failure, never a 500.
    return HTTPException(status_code=422, detail=str(exc))


def _annotation_limits() -> dict[str, int]:
    """The server-enforced annotation caps, echoed to the island so the client can
    pre-validate (the server stays the source of truth). Names mirror the constants
    in docs/annotations.md."""
    return {
        "paletteSize": HIGHLIGHT_PALETTE_SIZE,
        "maxSpanSegments": MAX_ANNOTATION_SPAN_SEGMENTS,
        "maxNoteChars": MAX_ANNOTATION_NOTE_CHARS,
        "maxTagsPerAnnotation": MAX_TAGS_PER_ANNOTATION,
        "maxQuoteChars": MAX_ANNOTATION_QUOTE_CHARS,
        "maxTagNameChars": MAX_TAG_NAME_CHARS,
    }


def _tag_shape(tag: AnnotationTag) -> dict[str, Any]:
    """One tag's island/JSON shape (camelCase, matching the frontend islands)."""
    return {
        "id": str(tag.id),
        "name": tag.name,
        "color": tag.color,
        "archived": tag.archived_at is not None,
    }


def _annotation_shapes(
    session: Session, run_id: uuid.UUID, rows: list[TranscriptAnnotation]
) -> list[dict[str, Any]]:
    """Resolve a set of stored annotations against the CURRENT render into the island
    JSON shape (camelCase): highlight spans, staleness, honest timing precision and
    seconds, live speakers, and the row metadata (colour/quote/note/tags). Shared by
    the list GET and the single-row create/patch responses so one row can never
    serialize two ways. Reads render the CORRECTED variant, exactly as the review
    surface does."""
    if not rows:
        return []
    lines = attributed_transcript(session, run_id, text=TranscriptText.CORRECTED)
    covered = load_covered_segments(session, run_id)
    anchors = [stored_anchor_from_row(row) for row in rows]
    resolved = {r.annotation_id: r for r in resolve_annotation_spans(lines, covered, anchors)}
    tags_by_id = tags_for_annotations(session, [row.id for row in rows])
    # Canonical transcript order is applied AFTER resolution — by rendered line index,
    # not the captured start_segment_index annotations_for_run sorts by — so the panel
    # and the bulk pull-quote export agree even when a split child reorders lines.
    ordered_rows = sorted(rows, key=lambda row: resolved_order_key(resolved[row.id]))
    shapes: list[dict[str, Any]] = []
    for row in ordered_rows:
        res = resolved[row.id]
        shapes.append(
            {
                "id": str(row.id),
                "anchorKind": row.anchor_kind,
                "colorIndex": row.color_index,
                "quote": row.quote_text,
                "note": row.note,
                "operator": row.operator,
                "stale": res.stale,
                "timingPrecision": res.timing_precision,
                "startSeconds": res.start_seconds,
                "endSeconds": res.end_seconds,
                "speakers": list(res.speakers),
                "spans": [
                    {"lineIndex": s.line_index, "start": s.start, "end": s.end} for s in res.spans
                ],
                "locatorLineIndex": res.locator_line_index,
                "startSegmentIndex": row.start_segment_index,
                "endSegmentIndex": row.end_segment_index,
                "tags": [_tag_shape(t) for t in tags_by_id.get(row.id, [])],
            }
        )
    return shapes


def _require_filter_tags_exist(session: Session, tag_ids: list[uuid.UUID]) -> None:
    """Fail closed (404) when a ``?tag=`` filter names a tag that does not exist — a
    mistyped or forged id (docs/annotations.md: an unknown tag is a 404, never
    silently indistinguishable from a valid filter that matched nothing)."""
    if not tag_ids:
        return
    found = set(
        session.execute(select(AnnotationTag.id).where(AnnotationTag.id.in_(tag_ids))).scalars()
    )
    missing = [t for t in tag_ids if t not in found]
    if missing:
        raise HTTPException(status_code=404, detail=f"unknown tag id {missing[0]}")


def _annotations_payload(
    session: Session, run_id: uuid.UUID, tag_ids: list[uuid.UUID]
) -> dict[str, Any]:
    """The GET /annotations body: the run's live annotations (optionally OR-filtered
    by tag) as island shapes, plus the full tag universe for the panel/picker."""
    rows = annotations_for_run(session, run_id, tag_ids=tag_ids or None)
    return {
        "annotations": _annotation_shapes(session, run_id, rows),
        "tags": [_tag_shape(t) for t in list_tags(session)],
    }


def _run_source_title(run: PipelineRun) -> str:
    """A non-blank, operator-recognizable source label for a pull-quote citation
    (issue #86): the run's sidecar title (issue #104, operator intent), else the
    acquisition-metadata title (issue #36), else a cleaned filename from the source
    path — the same display precedence the run listing uses."""
    title = title_from_snapshot(run.sidecar)
    if title is None and run.media_item.source_metadata is not None:
        title = run.media_item.source_metadata.title
    return friendly_media_label(title, run.media_item.source_path)


# Register the friendly-title helper as a Jinja global (issue #117): every console
# surface that names a run (queue, workbench, transcript, dashboard) resolves the
# title through the one precedence, so the pages never disagree. Registered here,
# after the definition, rather than in the globals block above (the function is
# not yet defined there).
templates.env.globals["run_source_title"] = _run_source_title


def _pull_quote_markdown(
    resolved: ResolvedAnnotation,
    lines: Sequence[TranscriptLine],
    *,
    source_title: str,
    tags: Sequence[str],
    note: str | None,
) -> str:
    """Assemble one highlight's pull-quote Markdown, refusing a stale or otherwise
    unresolvable highlight with a 409 ``X-Voxint-Conflict: stale`` (docs/annotations.md):
    the captured copy alone cannot reconstruct the live speaker attribution and
    per-line geometry a faithful quote needs, so the export never fabricates it from
    ``quote_text``."""
    clipped = clip_lines_for_export(resolved, lines)
    if (
        resolved.stale
        or not clipped
        or resolved.start_seconds is None
        or resolved.end_seconds is None
    ):
        raise HTTPException(
            status_code=409,
            detail="highlight is stale; refresh or re-anchor it before exporting",
            headers=_ANNOTATION_STALE_HEADERS,
        )
    return annotation_pull_quote(
        clipped,
        source_title=source_title,
        start_seconds=resolved.start_seconds,
        end_seconds=resolved.end_seconds,
        timing_precision=resolved.timing_precision,
        tags=tags,
        note=note,
    )


def _capture_payload_from_form(
    start_segment_id: uuid.UUID,
    start_offset: int,
    start_child_word_start: int | None,
    start_child_word_end: int | None,
    end_segment_id: uuid.UUID,
    end_offset: int,
    end_child_word_start: int | None,
    end_child_word_end: int | None,
    client_quote: str,
) -> CapturePayload:
    """Assemble a :class:`CapturePayload` from the flat form sextuple (x2). The service
    normalizes direction and classifies; the route never picks the anchor kind."""
    return CapturePayload(
        start=CaptureEndpoint(
            segment_id=start_segment_id,
            offset=start_offset,
            child_word_start=start_child_word_start,
            child_word_end=start_child_word_end,
        ),
        end=CaptureEndpoint(
            segment_id=end_segment_id,
            offset=end_offset,
            child_word_start=end_child_word_start,
            child_word_end=end_child_word_end,
        ),
        client_quote=client_quote,
    )


def _verify_annotation_claim(session: Session, run_id: uuid.UUID, token: uuid.UUID) -> PipelineRun:
    """Gate a run-scoped annotation write. An unknown run is a 404 (docs/annotations.md
    fail-closed taxonomy, checked before the claim so a missing run never masquerades
    as a claim conflict); a lost claim is a 409 marked ``X-Voxint-Conflict: claim`` so
    the island stops the loop and re-claims. Holds the row lock (``for_update``) so a
    concurrent re-claim serializes against the write."""
    _run_or_404(session, run_id)
    try:
        return verify_claim(session, run_id, token, for_update=True)
    except ClaimMismatchError as exc:
        raise HTTPException(
            status_code=409, detail=str(exc), headers=_CLAIM_CONFLICT_HEADERS
        ) from exc


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
    # The run's frozen pack, resolved once for every segment's correction
    # provenance (#83) — read-time, from the immutable per-run snapshot.
    rule_index = _load_run_rule_index(session, run_id)
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
        "segments": [
            _island_segment(ln, palette, rule_index) for ln in lines
        ],
    }


def _load_run_rule_index(
    session: Session, run_id: uuid.UUID
) -> DeclaredRuleIndex | None:
    """The run's frozen domain-pack snapshot resolved into a declared-rule index
    for read-time provenance (#83), or ``None`` when the run is gone or its
    snapshot is absent/corrupt.

    Reads the ``domain_pack`` snapshot column DIRECTLY and hands it to
    :func:`build_declared_rule_index` — never through ``domain_pack_from_snapshot``,
    which would degrade a NULL/corrupt snapshot to the current default pack and
    fabricate declarations this run never had.
    """
    run = session.get(PipelineRun, run_id)
    if run is None:
        return None
    return build_declared_rule_index(run.domain_pack)


def _island_segment(
    ln: TranscriptLine,
    palette: dict[str, int],
    rule_index: DeclaredRuleIndex | None = None,
) -> dict[str, Any]:
    """One transcript line as the island's per-segment shape — the ONE builder the
    hydrated props and the split-route response share, so a page reload and a live
    split can never disagree on a segment's fields.

    ``sourceSegmentId`` is the immutable PARENT id (issue #59): the verify / correct
    / split write target, identical to ``segmentId`` for an unsplit line and shared
    across a split parent's derived children. ``reviewTarget`` is true on exactly
    one line per parent — the queue entry — so the N-of-M loop counts one target
    per parent and never double-counts children.

    ``corrections`` / ``rawText`` (#83) carry deterministic domain-pack correction
    provenance and the immutable raw evidence for the compare/reset affordance.
    Both are whole-segment concerns: a split child (``word_start`` set) never
    carries them (the parent's spans address its full enhanced text, not a child
    slice), and ``correction_trace`` is ``None`` there anyway (a corrected segment
    is never split).
    """
    is_split_child = ln.word_start is not None
    # Operator edit supersedes pipeline provenance (#83): once the operator saves
    # their own text (`corrected`), the domain-pack trace's spans address the
    # PIPELINE-enhanced text, not the operator-effective text now shown — so the
    # "corrected by domain pack" marker would be stale and misleading. The client
    # clears it locally on a /text save, but the SERVER must own the rule too, or a
    # page reload (and any whole-run reconcile via /split or /relabel, which reuse
    # this builder) resurrects the stale marker. `rawText` stays exposed — the
    # compare / reset-to-raw affordance remains honest and useful after an edit.
    corrections = (
        None
        if is_split_child or ln.corrected
        else resolve_segment_provenance(
            ln.correction_trace, ln.corrector_version, rule_index
        )
    )
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
        # Deterministic domain-pack correction provenance (#83): which pack/rule
        # produced each edit, or an honest unavailable state; None when no rule
        # materially fired (or on a split child). Never a text diff — driven by the
        # persisted trace (trace_has_entries) alone.
        "corrections": corrections,
        # The immutable raw ASR text for the whole segment (#83), for the console's
        # compare / reset-to-raw affordance. None on split children (raw is a
        # whole-segment concern) and synthetic export lines.
        "rawText": None if is_split_child else ln.raw_text,
    }


def _run_island_segments(session: Session, run_id: uuid.UUID) -> list[dict[str, Any]]:
    """The run's island segment payload (issue #59) — CORRECTED variant, split
    parents expanded — for a live write to reconcile the console against server
    truth. Same builder as hydration, so a split response and a page reload agree."""
    palette = speaker_palette(_run_label_universe(session, run_id))
    lines = attributed_transcript(session, run_id, text=TranscriptText.CORRECTED)
    rule_index = _load_run_rule_index(session, run_id)
    return [_island_segment(ln, palette, rule_index) for ln in lines]


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
    folder_path: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Template context for a setup-wizard step.

    Read-only: it uses ``get_app_settings`` (never ``get_or_create``) so rendering a
    GET can't create the app_settings row. Fields prefill from the saved row layered
    over env defaults, matching how a run would resolve them; a POST that fails
    validation passes ``error=`` plus the raw submitted text via ``overrides`` so the
    operator's in-progress input survives the re-render.

    ``folder_path`` overrides the MEDIA-step browse location (the folder panel); it
    defaults to ``?path=`` from the query string. A non-HTMX folder mutation that
    fails re-renders the full page and passes the submitted ``path`` here so the
    browser stays put instead of snapping back to the root.
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
        # The finish-step summary counts registered folders; the media step itself
        # now renders them through the folder panel (issue #63), not a textarea.
        "media_folders": media_folders,
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
        # The finish step offers "Finish setup & start tutorial" (seeds if needed,
        # issue #75) alongside a plain "Finish setup"; this flag only selects the
        # already-seeded vs. seed-on-finish copy, never the redirect.
        "tutorial_available": ready_tutorial_run_id(session) is not None,
        "active_nav": "setup",
        "error": None,
        # Bounded, non-secret message when a seed-on-finish fails (issue #75).
        "tutorial_error": None,
    }
    # The folder browser (issue #63) is built only for the MEDIA step, so the
    # other five wizard screens never pay for a MEDIA_ROOT walk or a domain-pack
    # load. The untrusted ?path= is revalidated inside list_media_subdirs.
    if step is WizardStep.MEDIA:
        context.update(
            _folder_panel_context(
                session,
                settings,
                action_prefix="/setup/folders",
                csrf=context["csrf_setup"],
                path=folder_path if folder_path is not None else (
                    request.query_params.get("path") or "."
                ),
            )
        )
    context.update(overrides)
    return context


# Issue #77: when the LLM settings form turns LLM enhancement OFF, any feature that
# ``validate_effective_flags`` requires ``llm_enabled=true`` for would be stranded on
# — a combination the boot validator (config.py) rejects on restart. The LLM form
# refuses such a disable (writes nothing) rather than auto-disabling the dependent
# (#62's "never flip an unrelated setting"). The rejection copy is directed at THIS
# page and names the blocker, unlike ``_FEATURE_INVARIANT_COPY`` whose "turn it on in
# the LLM section below, or turn <feature> off" wording is written for the Features
# form and points the wrong way here. Keyed on the exact validator messages for the
# three llm-dependency invariants → the blocking feature's operator label; a drift
# test locks the keys against ``validate_effective_flags``.
_LLM_DEPENDENCY_LABELS: dict[str, str] = {
    "enrichment_names_llm_enabled requires llm_enabled=true — the "
    "LLM name pass reuses the configured enhancement endpoint": "the LLM name pass",
    "enrichment_run_assets_enabled requires llm_enabled=true — the"
    " asset generators reuse the configured enhancement endpoint": "run assets",
    "enrichment_web_research_enabled requires llm_enabled=true — the"
    " producer reuses the configured enhancement endpoint": "web-research enrichment",
}


def _join_operator_labels(labels: list[str]) -> str:
    """Grammatical, order-preserving join of one to three labels for operator copy."""
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _llm_disable_strand_error(row: AppSettings | None, settings: Settings) -> str | None:
    """Refuse to turn LLM enhancement OFF while a feature that needs it is on (#77).

    ``validate_effective_flags`` stays the single source of the invariant decision:
    the current effective flag combination is validated at its present LLM state and
    again with ``llm_enabled`` forced ``False``, and we act ONLY on the *delta* — the
    violations disabling LLM would newly introduce. So an unrelated pre-existing
    violation (e.g. a malformed web-search endpoint the fail-closed enable path left
    behind, or a dependent already stranded) never blocks an LLM save and never
    mislabels the cause; only the LLM-dependency invariants can flip here because the
    other flags are held at their current effective values. Returns an LLM-page plain
    message naming the blocker(s), or ``None`` when disabling is safe.
    """
    others: dict[str, object] = {
        "enrichment_names_enabled": resolve_effective_enrichment_names_enabled(row, settings),
        "enrichment_names_llm_enabled": resolve_effective_enrichment_names_llm_enabled(
            row, settings
        ),
        "enrichment_run_assets_enabled": resolve_effective_enrichment_run_assets_enabled(
            row, settings
        ),
        "enrichment_run_assets_autogenerate": resolve_effective_enrichment_run_assets_autogenerate(
            row, settings
        ),
        "voxint_web_research": resolve_effective_voxint_web_research(row, settings),
        "enrichment_web_research_enabled": resolve_effective_enrichment_web_research_enabled(
            row, settings
        ),
        "web_search_base_url": resolve_effective_web_search_base_url(row, settings),
    }
    before = set(
        validate_effective_flags(
            EffectiveFlags(llm_enabled=resolve_effective_llm_enabled(row, settings), **others)  # type: ignore[arg-type]
        )
    )
    new_violations = [
        message
        for message in validate_effective_flags(
            EffectiveFlags(llm_enabled=False, **others)  # type: ignore[arg-type]
        )
        if message not in before
    ]
    if not new_violations:
        return None
    labels = [
        _LLM_DEPENDENCY_LABELS[message]
        for message in new_violations
        if message in _LLM_DEPENDENCY_LABELS
    ]
    if not labels:
        # Defensive: a newly-introduced violation with no label (unreachable — only
        # the three llm-dependency invariants change when llm_enabled flips). Still
        # refuse, without naming a feature we can't identify.
        return (
            "Another feature still needs LLM enhancement. Turn it off in the Features"
            " or Sources & research section before turning LLM enhancement off."
        )
    needs = "it needs" if len(labels) == 1 else "they need"
    return (
        f"Turn off {_join_operator_labels(labels)} before turning LLM enhancement off"
        f" — {needs} the LLM. You can turn features off in the Features and Sources &"
        " research sections."
    )


# ---- Media folders + per-folder domain packs (issue #63) --------------------
# The pack <select>'s "Default" (= no mapping) option submits this explicit,
# non-empty sentinel — NOT "". FastAPI collapses both an absent field and an empty
# "" form value to None for an Optional[str] param, so "" cannot distinguish "the
# operator chose Default (clear the mapping)" from "the field was absent because the
# control was disabled (leave the mapping untouched)". A non-empty sentinel makes
# that distinction unambiguous: None ⇒ no-op, sentinel ⇒ clear, a real name ⇒ set.
_PACK_DEFAULT_SENTINEL = "__default__"

# The folder browser and the Settings "Media folders" section share these
# helpers. Every mutation serializes on the singleton app_settings row so
# overlapping htmx requests (double-clicks, multiple tabs) cannot lose an update
# or leave a folder_domain_packs mapping whose key is not a registered folder —
# the feature's core invariant: set(folder_domain_packs) ⊆ set(media_folders).
# ARRAY/JSON columns are not Mutable-tracked, so every write reassigns a fresh
# object to flag the column dirty.


def _locked_app_settings(session: Session, settings: Settings) -> AppSettings:
    """The singleton app_settings row, created if absent and locked FOR UPDATE.

    ``get_or_create`` handles the first-insert race; the follow-up
    ``SELECT … FOR UPDATE`` then locks and rereads so two overlapping folder
    mutations serialize (the second blocks until the first commits, then sees its
    result). On PostgreSQL this is a real row lock; on SQLite SQLAlchemy renders no
    locking clause (single-writer anyway), so the tests that assert serialization
    run against the production database.

    ``populate_existing=True`` is load-bearing: ``get_or_create`` has already loaded
    the singleton into this Session's identity map, so a plain ``FOR UPDATE`` reread
    would take the lock but return the *cached* instance with its pre-lock attribute
    values — silently defeating the reread and re-opening the lost-update window the
    lock exists to close. The option forces the locked row's freshly-read columns to
    overwrite the identity-mapped instance, so ``row.media_folders`` reflects the
    post-lock committed state on every path (not only when the lock happened to
    block on a concurrent writer).
    """
    get_or_create(session, llm_enabled_default=settings.llm_enabled)
    return session.execute(
        select(AppSettings)
        .where(AppSettings.id == APP_SETTINGS_SINGLETON_ID)
        .with_for_update()
        .execution_options(populate_existing=True)
    ).scalar_one()


def _add_media_folder(session: Session, settings: Settings, raw_path: str) -> str | None:
    """Register one folder under MEDIA_ROOT. Returns an error message, else ``None``.

    Only the SUBMITTED path is validated (via ``normalize_media_folders`` — resolve
    + containment + reserved-tree + existing-dir), so a stale/removed *existing*
    folder can never block a new add. The canonical result is merged into the
    locked row's ``media_folders`` (dedup, ``MAX_MEDIA_FOLDERS`` cap). Re-adding a
    folder already registered is an idempotent no-op.
    """
    row = _locked_app_settings(session, settings)
    try:
        normalized = normalize_media_folders([raw_path], settings.media_root)
    except SetupValidationError as exc:
        return str(exc)
    if not normalized:
        return "Choose a folder to add."
    folder = normalized[0]
    existing = list(row.media_folders or [])
    if folder in existing:
        return None  # idempotent
    if len(existing) + 1 > MAX_MEDIA_FOLDERS:
        return f"You can register at most {MAX_MEDIA_FOLDERS} folders."
    row.media_folders = [*existing, folder]
    return None


def _remove_media_folder(session: Session, settings: Settings, folder: str) -> str | None:
    """Unregister a folder and prune its domain-pack mapping. Idempotent.

    Operates only on stored identities (never touches the filesystem): a ``folder``
    not currently registered is a no-op. Removal also drops
    ``folder_domain_packs[folder]`` so no orphan mapping survives.
    """
    row = _locked_app_settings(session, settings)
    existing = list(row.media_folders or [])
    if folder not in existing:
        return None  # already gone — idempotent
    row.media_folders = [f for f in existing if f != folder]
    mapping = dict(row.folder_domain_packs or {})
    if mapping.pop(folder, None) is not None:
        row.folder_domain_packs = mapping
    return None


def _set_folder_pack(
    session: Session, settings: Settings, folder: str, pack: str | None
) -> str | None:
    """Assign (or clear) a registered folder's domain pack. Last write wins.

    ``pack is None`` means the ``pack`` field was *absent* from the submission — a
    no-op, never a clear. This matters when the registry is down: the panel disables
    the ``<select>`` (and the Set button), and a disabled control submits nothing, so
    an accidental submit must not be read as "select Default" and wipe the stored
    mapping. ``pack == _PACK_DEFAULT_SENTINEL`` is the explicit "Default" choice and
    removes the mapping. Any other non-empty pack must resolve in
    ``available_domain_packs`` (an unknown name or a registry-wide ``DomainPackError``
    writes nothing and surfaces the message). The folder must already be registered —
    enforcing the mapping-key ⊆ media_folders invariant on write.
    """
    if pack is None:
        return None  # field absent (e.g. disabled select) — never touch the mapping
    row = _locked_app_settings(session, settings)
    if folder not in (row.media_folders or []):
        return "That folder is not registered."
    mapping = dict(row.folder_domain_packs or {})
    if pack == _PACK_DEFAULT_SENTINEL:
        if mapping.pop(folder, None) is not None:
            row.folder_domain_packs = mapping
        return None
    try:
        available = available_domain_packs(settings)
    except DomainPackError:
        return (
            "Domain packs can't be listed right now — check your DOMAIN_PACKS_DIR / "
            "DOMAIN_PACK_PATH configuration."
        )
    if pack not in available:
        return f"Unknown domain pack: {pack}."
    mapping[folder] = pack
    row.folder_domain_packs = mapping
    return None


def _apply_folder_mutation(
    session: Session,
    settings: Settings,
    *,
    action: str,
    folder: str,
    pack: str | None,
) -> str | None:
    """Dispatch a folder-panel mutation to the right persist helper.

    ``action`` is the consolidated route's verb (one POST per mount instead of
    three) — an unknown value is a client error (422), never a silent no-op.
    """
    if action == "add":
        return _add_media_folder(session, settings, folder)
    if action == "remove":
        return _remove_media_folder(session, settings, folder)
    if action == "pack":
        return _set_folder_pack(session, settings, folder, pack)
    raise HTTPException(status_code=422, detail=f"unknown folder action {action!r}")


def _folder_missing(root: Path, folder: str) -> bool:
    """True if a registered ``folder`` no longer resolves to a directory in the root.

    ``root`` is already ``resolve()``-d. Mirrors the browse/add containment posture:
    a path that escapes the root (symlink swap), fails to resolve (embedded NUL), or
    is not an existing directory reads as missing — never as a present directory
    sitting outside MEDIA_ROOT.
    """
    try:
        resolved = (root / folder).resolve()
        return not (resolved.is_relative_to(root) and resolved.is_dir())
    except (ValueError, OSError):
        return True


def _folder_panel_context(
    session: Session,
    settings: Settings,
    *,
    action_prefix: str,
    csrf: str,
    path: str,
) -> dict[str, Any]:
    """Template context for the shared folder-panel fragment (issue #63).

    ``action_prefix`` is ``/setup/folders`` or ``/settings/folders`` and ``csrf``
    the matching token, so one fragment serves both the wizard (CSRF_SETUP) and the
    Settings page (CSRF_SETTINGS). ``path`` is the untrusted current browse
    location, revalidated by ``list_media_subdirs``. ``available_domain_packs`` is
    loaded once; a registry-wide ``DomainPackError`` disables the pack selects with
    an honest message rather than crashing the page, and a stored pack no longer
    available is flagged so the select can show it as "(unavailable)" instead of a
    false "Default".
    """
    row = get_app_settings(session)
    registered = list(row.media_folders) if row and row.media_folders else []
    mapping = dict(row.folder_domain_packs) if row and row.folder_domain_packs else {}
    listing = list_media_subdirs(settings.media_root, path, set(registered))
    packs_available = True
    pack_names: list[str] = []
    try:
        pack_names = sorted(available_domain_packs(settings))
    except DomainPackError:
        packs_available = False
    # Full-page navigation base for the no-JS fallback (htmx enhances these into
    # #folder-panel swaps). Exactly two mounts, so deriving the prefix here keeps
    # the call sites clean.
    is_setup = action_prefix == "/setup/folders"
    if is_setup:
        nav_prefix, nav_suffix, nav_base = "/setup?step=media&path=", "", "/setup"
    else:
        nav_prefix, nav_suffix, nav_base = "/settings?path=", "#folders", "/settings"
    root = settings.media_root.resolve()
    folder_rows: list[dict[str, Any]] = []
    for folder in registered:
        current_pack = mapping.get(folder, "")
        folder_rows.append(
            {
                "rel": folder,
                "pack": current_pack,
                # A stored pack the current registry doesn't offer (absent from a
                # healthy registry, or a registry that can't be listed at all) is
                # shown explicitly as its own selected "(unavailable)" option — never
                # a false "Default".
                "pack_unavailable": bool(current_pack) and current_pack not in pack_names,
                # A registered folder that has since vanished on disk is flagged
                # (still removable) — honest state, not a silent drop. Resolve +
                # contain (not a bare ``is_dir``) to match the browse/add posture: a
                # symlink or path that now escapes the root reads as missing rather
                # than as a present directory outside MEDIA_ROOT.
                "missing": _folder_missing(root, folder),
            }
        )
    return {
        "folders_browse_action": f"{action_prefix}/browse",
        "folders_mutate_action": action_prefix,
        "folders_nav_prefix": nav_prefix,
        "folders_nav_suffix": nav_suffix,
        "folders_nav_base": nav_base,
        "folders_wizard": is_setup,
        "folders_csrf": csrf,
        "folder_listing": listing,
        "folder_rows": folder_rows,
        "folder_pack_names": pack_names,
        "folder_packs_available": packs_available,
        # The "Default" <option> submits this explicit sentinel (see the constant) so
        # a disabled/absent select is never misread as "clear the mapping".
        "folder_pack_default_value": _PACK_DEFAULT_SENTINEL,
        "folder_error": None,
    }


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
    * **Stranded-dependent disable** (issue #77 — turning LLM off while a feature that
      needs it is effectively on) returns a plain message and writes NOTHING (no
      ``get_or_create``), so LLM stays on rather than the form auto-disabling an
      unrelated feature (#62). See :func:`_llm_disable_strand_error`.
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
    # Issue #77: refuse a deliberate disable that would strand a feature depending on
    # LLM, BEFORE any mutation — write nothing (no get_or_create), keep LLM on rather
    # than auto-disabling the dependent (#62). Scope is deliberate-disable-only: the
    # fail-closed *enable* path below (requested-on but no usable key/budget) forces
    # llm_enabled=False and so can ITSELF leave a dependent stranded, yet it is left
    # unguarded on purpose — it already returns the operator's real problem (the
    # key/budget error), keeps the valid key they typed (#46), and fixing that key
    # re-enables LLM and un-strands the dependent. Guarding it would have to either
    # drop that key or bury the key error under a strand message. Known residual (#77).
    if not enabled:
        strand_error = _llm_disable_strand_error(get_app_settings(session), settings)
        if strand_error is not None:
            return strand_error
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
        # Issue #67: a keyless enable is legitimate when the bundled local model is
        # the active endpoint — resolve it from the just-created row (the operator
        # may have turned the bundle on in Features first) so the keyless audience
        # can actually flip the master LLM switch. BYO-only jobs stay key-gated.
        bundled_active = llm_bundled_active(row, settings)
        try:
            validate_llm_enable(effective_key, settings, bundled_active=bundled_active)
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
        "llm_bundled_enabled",
        "Use the bundled local model",
        "Route transcript enhancement and run-asset summaries + entities to the"
        " bundled local model, so they work with no external API key. It powers"
        " ONLY those — topics, web research, and LLM name suggestions still need a"
        " BYO endpoint and key. Requires LLM enhancement to be on, and the bundled"
        " model service to be running (compose.llm.yaml).",
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

# Operator-plain copy for the invariant violations the settings sections surface
# (Features #62 + Sources & research #76). validate_effective_flags is the SINGLE
# source of WHICH combinations are invalid, but its messages name the flag
# identifiers (enrichment_run_assets_enabled, web_search_base_url, …) — the exact
# jargon this arc exists to keep out of a non-technical operator's way. So the
# settings boundaries translate the reachable messages to plain language, while the
# config boot validator keeps the identifier-bearing strings (a .env editor wants
# the variable name). Keyed on the exact shared message; an un-mapped message falls
# through to the original. A drift test locks every reachable key so a reworded
# invariant can never silently fall back to jargon here.
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
    # Sources & research section (issue #76). The producer + base-URL invariants.
    "enrichment_web_research_enabled requires voxint_web_research=true"
    " — the producer's only egress is the controlled retrieval tools": (
        "Web-research enrichment needs Web research turned on above — the"
        " producer's only way out to the network is through it."
    ),
    "enrichment_web_research_enabled requires llm_enabled=true — the"
    " producer reuses the configured enhancement endpoint": (
        "Web-research enrichment needs LLM transcript enhancement turned on. Turn"
        " it on in the LLM section, or turn web-research enrichment off."
    ),
    "voxint_web_research=true requires web_search_base_url — the"
    " searxng provider has no default endpoint": (
        "Web research needs a search provider endpoint. Enter one below, or turn"
        " Web research off."
    ),
    "web_search_base_url must not contain whitespace or backslashes": (
        "The search provider endpoint can't contain spaces or backslashes — check"
        " for a stray character."
    ),
    "web_search_base_url is malformed": (
        "The search provider endpoint isn't a valid web address — check for a typo"
        " (for example an invalid port)."
    ),
    "web_search_base_url must be an absolute http(s) URL": (
        "The search provider endpoint must be a full web address starting with"
        " http:// or https://."
    ),
    "web_search_base_url must not embed credentials — use web_search_api_key": (
        "Don't put a username or password in the endpoint — use the API key field"
        " below instead."
    ),
    "web_search_base_url must be a bare endpoint (no query/fragment)": (
        "Enter just the endpoint address — no “?query” or “#fragment” at the end."
    ),
}


# Issue #61: plain-language remediation for the wizard SERVICES step's readiness
# checks (``voxint doctor`` surfaced in the browser). Keyed by a coarse CATEGORY, not
# the raw diagnostics check name, because the three model services carry distinct
# names (transcription / diarization / speaker embedding) that share one fix. Shown
# only when a check is not ``ready``. The model copy is warm-up-aware: a reachable
# service whose model is still loading reads as ``failed`` (the pipeline genuinely
# can't run yet), so the text names "starting" so it isn't misread as a crash.
_DOCTOR_REMEDIATION: dict[str, str] = {
    "database": (
        "Start Postgres, then re-check. Voxint keeps every run and its transcript"
        " here — nothing can be submitted until it's up."
    ),
    "redis": (
        "Start Redis, then re-check. It's the task queue — submissions wait here to"
        " be picked up, so the pipeline can't run without it."
    ),
    "models": (
        "Start the model services — the GPU overlay (compose.gpu.yaml) or, with no"
        " NVIDIA GPU, the CPU overlay (compose.cpu.yaml); see the README quickstart."
        " If one just started it may still be loading its model — re-check in a"
        " moment."
    ),
    "llm": (
        "LLM transcript enhancement is on but the endpoint didn't answer (or rejected"
        " the check) — check the LLM section in Settings. Transcription and diarization"
        " still run; enhancement is simply skipped until the endpoint is reachable."
    ),
    # Fallback for any diagnostics check not explicitly categorized below — a neutral
    # "look at this dependency" rather than wrongly steering the operator at the model
    # services. No current check lands here (see _doctor_category), but a future one
    # gets honest generic copy instead of an empty string or a misleading fix.
    "other": (
        "This dependency isn't ready — check its configuration in Settings or the"
        " README, then re-check."
    ),
}


def _doctor_category(name: str) -> str:
    """Map a diagnostics :class:`~voxint.diagnostics.CheckResult` name to a
    ``_DOCTOR_REMEDIATION`` category. Total by construction — an unrecognized name (a
    future check) falls through to the neutral ``other`` copy, never a KeyError and
    never the wrong (model-services) remediation."""
    if name == "postgres":
        return "database"
    if name == "redis":
        return "redis"
    if name in {"transcription", "diarization", "speaker embedding"}:
        return "models"
    if name == "llm endpoint":
        return "llm"
    return "other"


def _doctor_checks(request: Request, session: Session) -> list[dict[str, Any]]:
    """Run the wizard SERVICES readiness checks (issue #61) and shape them for the
    template. Self-contains every dependency failure (DB / redis / model / LLM), so
    it never raises into the request — the caller can render a 200 regardless."""
    settings: Settings = request.app.state.settings
    engine = cast(Engine, session.get_bind())
    with httpx.Client(timeout=httpx.Timeout(settings.health_probe_timeout_seconds)) as client:
        results = run_diagnostics(
            settings, engine, http_client=client, include_hf_token=False
        )
    return [
        {
            "name": r.name,
            "state": check_state(r),
            "detail": r.detail,
            "remediation": _DOCTOR_REMEDIATION[_doctor_category(r.name)],
        }
        for r in results
    ]


def _minimal_services_context(request: Request, settings: Settings) -> dict[str, Any]:
    """DB-free context for the SERVICES step when the ``app_settings`` read itself
    fails — a down or unmigrated Postgres makes ``_setup_context``'s row + tutorial
    queries raise. Showing that (a failed ``postgres`` readiness row) is the whole
    point of this step, so it must still render rather than 500. Carries only the keys
    the SERVICES branch + the shared nav/layout touch; the other steps' DB-derived
    fields are absent (the Jinja env uses lenient ``Undefined`` and those fields are
    referenced only inside other steps' branches). The caller attaches the checks."""
    return {
        "request": request,
        "step": WizardStep.SERVICES,
        "steps": STEP_ORDER,
        "step_index": STEP_ORDER.index(WizardStep.SERVICES),
        "next_step": next_step(WizardStep.SERVICES),
        "csrf_setup": mint_csrf_token(request.app.state.csrf_secret, CSRF_SETUP),
        "media_root": str(settings.media_root),
        "active_nav": "setup",
        "error": None,
    }


def _services_step_response(request: Request, session: Session) -> Response:
    """Render the wizard SERVICES step with the full doctor readiness checks (#61).

    The checks run first — they self-contain every failure, so this GET is a 200 even
    with every dependency down — then the page context is built, falling back to a
    DB-free context when the ``app_settings`` read raises (Postgres down/unmigrated) so
    the operator still sees the failed-postgres row instead of a 500."""
    settings: Settings = request.app.state.settings
    checks = _doctor_checks(request, session)
    try:
        context = _setup_context(request, session, WizardStep.SERVICES)
    except SQLAlchemyError:
        # The poisoned session must be rolled back so the trailing commit in
        # _get_session is a harmless no-op rather than re-raising.
        session.rollback()
        context = _minimal_services_context(request, settings)
    context["doctor_checks"] = checks
    return templates.TemplateResponse(request, "setup.html", context)


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
    replace, and the real form always submits every radio in ``_FEATURE_FLAG_META``
    (the POST route must declare a matching ``Form`` param for each, or Starlette
    drops the field and this loop silently reverts it — see ``settings_features``).

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


# The two live-read flags the Settings "Semantic search" section exposes as
# tri-state toggles (issue #121). Kept out of _FEATURE_FLAG_META on purpose: the
# semantic pair depends on nothing else and validates through its OWN self-contained
# invariant (semantic_index_flags_ok), never the EffectiveFlags web, matching the
# app_settings design note that these flags stay out of EffectiveFlags.
_SEMANTIC_FLAG_NAMES: tuple[str, ...] = (
    "semantic_index_enabled",
    "semantic_index_autogenerate",
)


def _persist_semantic_index(
    session: Session,
    settings: Settings,
    *,
    submitted: dict[str, str],
) -> list[str]:
    """Apply the Semantic search section's two tri-state toggles atomically (#121).

    Mirrors :func:`_persist_feature_flags`' candidate → validate → one-mutation
    shape, but validates the semantic pair's OWN self-contained invariant
    (:func:`semantic_index_flags_ok`: autogenerate rides on the feature, so it
    requires the feature enabled) rather than the EffectiveFlags web. The effective
    combination (candidate over env default) is checked BEFORE any get_or_create, so
    a rejected save writes NOTHING (the API session commits on the 200 error
    re-render). An unexpected choice is rejected, never coerced, for the same reason
    the Features section rejects one: silently mapping it would drop an override or
    disable a feature. Returns operator-plain error messages (empty ⇒ success); the
    caller commits on success.
    """
    candidates: dict[str, bool | None] = {}
    for name in _SEMANTIC_FLAG_NAMES:
        choice = submitted.get(name, "inherit")
        if choice not in _FEATURE_FLAG_CHOICES:
            return [
                "Unrecognized semantic-search setting — choose On, Off, or Use"
                " installation setting."
            ]
        candidates[name] = None if choice == "inherit" else (choice == "on")
    row = get_app_settings(session)

    def _effective(name: str) -> bool:
        candidate = candidates[name]
        return bool(getattr(settings, name)) if candidate is None else candidate

    # semantic_index_flags_ok is the single source of WHICH combination is invalid;
    # its message names the flag identifiers (for a .env editor), so the operator
    # boundary translates the one reachable violation to plain language.
    if (
        semantic_index_flags_ok(
            enabled=_effective("semantic_index_enabled"),
            autogenerate=_effective("semantic_index_autogenerate"),
        )
        is not None
    ):
        return [
            "Turn semantic search on before auto-indexing new runs — the automatic"
            " step only runs the feature it rides on."
        ]
    row = get_or_create(session, llm_enabled_default=settings.llm_enabled)
    for name, value in candidates.items():
        setattr(row, name, value)
    return []


def _persist_web_research(
    session: Session,
    settings: Settings,
    *,
    submitted_master: str,
    submitted_producer: str,
    raw_base_url: str,
    raw_key: str,
    remove_key: bool,
    raw_domains: str,
) -> list[str]:
    """Apply the Sources & research section as ONE deliberate mutation (issue #76).

    The web-research capability's five operator controls — the master toggle
    (``voxint_web_research``) + producer toggle (``enrichment_web_research_enabled``)
    as tri-state on/off/inherit, the provider ``web_search_base_url`` (string
    override) + ``web_search_api_key`` (secret), and the ``source_authority_domains``
    editor — jointly configure one capability, so they save atomically: the whole
    form is a candidate that is validated against EVERY error source before the row
    is touched, and ANY violation (out-of-set choice, cross-flag invariant, a
    malformed endpoint, a malformed domain, or the contradictory key remove+replace)
    writes NOTHING (no ``get_or_create`` — the API session commits on the 200 error
    re-render, so the absence of a write is the guarantee). Returns the list of
    operator-plain error messages (empty ⇒ success); the caller commits on success.

    The flags NOT edited here (``llm_enabled`` + the names/run-assets flags) are
    resolved at their CURRENT effective value so a dependency invariant fires against
    real system state (enabling the producer while LLM is off is rejected) and this
    section never flips an unrelated setting. The API key is a credential: it never
    leaves this function and no returned message interpolates it.
    """
    row = get_app_settings(session)
    errors: list[str] = []

    # Tri-state toggles → True / False / None (inherit). An unrecognized choice (a
    # stale client / hand-crafted POST — never the shipped radios) is rejected, not
    # coerced (coercing would silently disable or drop an override), mirroring
    # _persist_feature_flags.
    if (
        submitted_master not in _FEATURE_FLAG_CHOICES
        or submitted_producer not in _FEATURE_FLAG_CHOICES
    ):
        return ["Unrecognized setting — choose On, Off, or Use installation setting."]
    cand_master = None if submitted_master == "inherit" else submitted_master == "on"
    cand_producer = None if submitted_producer == "inherit" else submitted_producer == "on"

    # String override candidates (base URL + domains): blank → NULL (inherit); a
    # value that merely echoes the env default → NULL too, so saving never silently
    # pins the env value onto the row (issue #46 tri-state precedent). Stored
    # verbatim-stripped; the runtime parser/validator re-reads the effective value.
    stripped_base = raw_base_url.strip()
    if not stripped_base:
        cand_base: str | None = None
    elif stripped_base == settings.web_search_base_url:
        cand_base = None
    else:
        cand_base = stripped_base
    stripped_domains = raw_domains.strip()
    if not stripped_domains:
        cand_domains: str | None = None
    elif stripped_domains == settings.source_authority_domains.strip():
        cand_domains = None
    else:
        cand_domains = stripped_domains

    # Secret candidate (the _persist_llm_settings precedent): blank = keep stored
    # (never prefilled), remove = NULL (revert to env), typed = the new value. The
    # contradictory remove+replacement combination is rejected.
    typed_key = bool(raw_key.strip())
    try:
        new_key = normalize_web_search_api_key(raw_key)
    except SetupValidationError as exc:
        new_key = None
        errors.append(str(exc))
    # Detect the contradiction from the RAW submission, so remove + a *malformed*
    # replacement still reports it in the same pass (a normalize failure nulls
    # new_key, which would otherwise hide the contradiction until a second attempt).
    if remove_key and typed_key:
        errors.append(
            "Choose either a new web-search API key or “remove saved key”, not both."
        )
    if remove_key:
        cand_key: str | None = None
    elif new_key is not None:
        cand_key = new_key
    else:
        cand_key = row.web_search_api_key if row is not None else None

    # Domain-format errors (strict — rejects exactly what the runtime parser drops).
    # Validate only what would actually be STORED: a blank or env-echoed submission
    # collapses to NULL (inherit env, permissive at runtime), so there is nothing to
    # reject — an operator who leaves the field alone is never shown a domain error.
    if cand_domains is not None:
        errors.extend(validate_authority_domains(cand_domains))

    # Cross-flag invariants over the effective (candidate-over-env) combination. The
    # effective base URL is validated HERE by validate_effective_flags when the
    # master toggle is effectively on (its voxint_web_research ⇒ valid-base-url rule).
    effective_master = bool(settings.voxint_web_research) if cand_master is None else cand_master
    effective_producer = (
        bool(settings.enrichment_web_research_enabled) if cand_producer is None else cand_producer
    )
    effective_base = cand_base if cand_base is not None else settings.web_search_base_url
    errors.extend(
        validate_effective_flags(
            EffectiveFlags(
                # Not edited in this section — resolved at the current effective value
                # so a dependency (producer ⇒ llm) fires against real state.
                llm_enabled=resolve_effective_llm_enabled(row, settings),
                enrichment_names_enabled=resolve_effective_enrichment_names_enabled(row, settings),
                enrichment_names_llm_enabled=resolve_effective_enrichment_names_llm_enabled(
                    row, settings
                ),
                enrichment_run_assets_enabled=resolve_effective_enrichment_run_assets_enabled(
                    row, settings
                ),
                enrichment_run_assets_autogenerate=(
                    resolve_effective_enrichment_run_assets_autogenerate(row, settings)
                ),
                # Edited here — the candidate over env default.
                voxint_web_research=effective_master,
                enrichment_web_research_enabled=effective_producer,
                web_search_base_url=effective_base,
            )
        )
    )
    # Standalone base-URL validation covers the complementary case only: an explicit
    # override saved WHILE the master toggle is off (validate_effective_flags checks
    # the base URL only when master is on, so this catches a latent-malformed
    # override that would break the instant master flips on). The two branches are
    # mutually exclusive, so no message is ever produced twice.
    if not effective_master and cand_base is not None:
        url_error = validate_web_search_base_url(cand_base)
        if url_error is not None:
            errors.append(url_error)

    if errors:
        # Translate identifier-bearing invariant messages to operator-plain copy;
        # nothing is written.
        return [_FEATURE_INVARIANT_COPY.get(message, message) for message in errors]

    # Valid → one mutation. get_or_create only now, so a rejected save writes nothing.
    row = get_or_create(session, llm_enabled_default=settings.llm_enabled)
    row.voxint_web_research = cand_master
    row.enrichment_web_research_enabled = cand_producer
    row.web_search_base_url = cand_base
    row.web_search_api_key = cand_key
    row.source_authority_domains = cand_domains
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
    finish controls). The adjudicate→check_words→export next-links carry the
    verified claim ``token`` so the workbench and transcript stepper stay writable;
    the export link never does.
    """
    step = parse_tutorial_step(request.query_params.get("tutorial"))
    if step is None or STEP_PAGE[step] is not page:
        return None
    tutorial_run_id = ready_tutorial_run_id(session)
    if tutorial_run_id is None:
        return None
    # Run-scoped pages must be showing THE tutorial run; the queue page carries no
    # run_id and only needs the tutorial run to exist (checked above).
    if (
        page in (TutorialPage.RUN_DETAIL, TutorialPage.WORKBENCH, TutorialPage.TRANSCRIPT)
        and run_id != tutorial_run_id
    ):
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
            # Step 1 → Step 2: hand off to the transcript stepper (issue #117 Phase
            # B), carrying the live claim token so verify/edit stay enabled there.
            banner["next_href"] = (
                f"/review/{tutorial_run_id}/transcript?token={token}&tutorial=check_words"
            )
            banner["next_label"] = "Continue to checking the words →"
        else:
            # No live claim on this tab — offer to (re)claim and continue rather
            # than a dead next-link that would land on a read-only workbench.
            banner["claim_run_id"] = tutorial_run_id
            banner["csrf_claim"] = mint_csrf_token(secret, CSRF_CLAIM)
    elif step is TutorialStep.CHECK_WORDS:
        if token is not None:
            # Step 2 → export: stay on the transcript page (both steps share it),
            # keeping the claim token so a returning tab is still writable.
            banner["next_href"] = (
                f"/review/{tutorial_run_id}/transcript?token={token}&tutorial=export"
            )
            banner["next_label"] = "I've checked the words →"
        else:
            # A stale/absent token here means the workbench claim is gone; recover
            # by re-claiming (which re-enters the walkthrough on run identity)
            # rather than offering a dead read-only next-link.
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
        if step is WizardStep.SERVICES:
            # The services step surfaces the full `voxint doctor` readiness checks
            # (issue #61) — a few-second network op we don't pay on other steps.
            # include_hf_token=False drops the Hugging Face check: the default install
            # runs on vendored weights, so it's noise AND a live huggingface.co call
            # this step has no reason to make. _services_step_response renders a 200
            # even when Postgres itself is down (that's exactly what it must show).
            return _services_step_response(request, session)
        context = _setup_context(request, session, step)
        return templates.TemplateResponse(request, "setup.html", context)

    def _setup_redirect(step: WizardStep) -> RedirectResponse:
        return RedirectResponse(f"/setup?step={step.value}", status_code=303)

    # Folder registration on the media step is the browser panel (issue #63):
    # POST /setup/folders, not a bulk textarea. The old POST /setup/media route
    # was removed with the textarea.

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
        # A freeze-time domain-pack collision (issue #84) / unresolvable pack (issue
        # #11) aborts the batch with a plain-language 422 rather than a raw 500 — no
        # run is committed (the commit is a single call below), so nothing is stranded.
        try:
            run_ids = [
                run.id
                for path in result.candidates
                if (run := submit_media_item_if_new(session, path)) is not None
            ]
        except DomainPackError as exc:
            raise HTTPException(
                status_code=422, detail=_submit_domain_pack_detail(exc)
            ) from exc
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

    # ---- Folder browser + per-folder domain packs (issue #63) --------------
    # Two routes per mount: a read-only browse GET (no CSRF — authenticated,
    # bounded, never creates the row) and one mutate POST carrying an action verb.
    # An HX request gets the panel fragment back; a plain request degrades to a
    # full-page redirect on success, or a full-page re-render carrying the error
    # inline on failure (mirrors _roster_response) — never a silent reload that
    # discards the message and looks like the mutation succeeded.

    def _folder_panel_response(
        request: Request,
        session: Session,
        settings: Settings,
        *,
        action_prefix: str,
        csrf_action: str,
        path: str,
        error: str | None,
        redirect_url: str,
        error_page: Callable[[str], Response],
    ) -> Response:
        if not request.headers.get("HX-Request"):
            if error is not None:
                # No-JS failure: re-render the whole page with the message inline
                # (and the browse position preserved), not a 303 that drops it.
                return error_page(error)
            return RedirectResponse(redirect_url, status_code=303)
        csrf = mint_csrf_token(request.app.state.csrf_secret, csrf_action)
        context = _folder_panel_context(
            session, settings, action_prefix=action_prefix, csrf=csrf, path=path
        )
        context["folder_error"] = error
        response = templates.TemplateResponse(
            request, "fragments/folder_panel.html", {"request": request, **context}
        )
        # Authenticated fragment: keep it out of any shared/browser cache.
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/setup/folders/browse", include_in_schema=False)
    def setup_folders_browse(
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        path: Annotated[str, Query(max_length=4096)] = ".",
    ) -> Response:
        settings: Settings = request.app.state.settings
        csrf = mint_csrf_token(request.app.state.csrf_secret, CSRF_SETUP)
        context = _folder_panel_context(
            session, settings, action_prefix="/setup/folders", csrf=csrf, path=path
        )
        response = templates.TemplateResponse(
            request, "fragments/folder_panel.html", {"request": request, **context}
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.post("/setup/folders", include_in_schema=False)
    def setup_folders(
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        action: Annotated[str, Form(max_length=16)] = "",
        folder: Annotated[str, Form(max_length=4096)] = "",
        pack: Annotated[str | None, Form(max_length=200)] = None,
        path: Annotated[str, Form(max_length=4096)] = ".",
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> Response:
        _require_csrf(request, CSRF_SETUP, csrf_token)
        settings: Settings = request.app.state.settings
        error = _apply_folder_mutation(
            session, settings, action=action, folder=folder, pack=pack
        )
        if error is None:
            session.commit()
        else:
            # A 200 re-render would otherwise commit the get_or_create insert /
            # the FOR UPDATE lock's transaction — roll back so a failed mutation
            # writes nothing (mirrors the wizard's other error re-renders).
            session.rollback()
        redirect = "/setup?" + urlencode({"step": "media", "path": path})

        def _error_page(message: str) -> Response:
            return templates.TemplateResponse(
                request,
                "setup.html",
                _setup_context(
                    request, session, WizardStep.MEDIA, folder_path=path,
                    folder_error=message,
                ),
            )

        return _folder_panel_response(
            request,
            session,
            settings,
            action_prefix="/setup/folders",
            csrf_action=CSRF_SETUP,
            path=path,
            error=error,
            redirect_url=redirect,
            error_page=_error_page,
        )

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
        start_tutorial: Annotated[str | None, Form()] = None,
    ) -> Response:
        _require_csrf(request, CSRF_SETUP, csrf_token)
        settings: Settings = request.app.state.settings
        # ``start_tutorial`` is the operator's explicit intent (the primary Finish
        # button), NOT the seed mechanism: it both seeds-if-needed and drives the
        # redirect, so a plain "Finish setup" never launches a tutorial and its
        # label never lies (issue #75). Seed BEFORE completing onboarding so a
        # storage/asset failure aborts the whole request with nothing committed.
        # Exact "1" (the button's value) — a crafted start_tutorial=0/false is not
        # an intent to start.
        wants_tutorial = start_tutorial == "1"
        seeded_run_id: uuid.UUID | None = None
        if wants_tutorial:
            seeded_run_id, error = _try_seed_tutorial(session, settings)
            if error is not None:
                session.rollback()
                return templates.TemplateResponse(
                    request,
                    "setup.html",
                    _setup_context(
                        request, session, WizardStep.FINISH, tutorial_error=error
                    ),
                )
        complete_onboarding(session, llm_enabled_default=settings.llm_enabled)
        # Commit explicitly before the redirect so the request that follows cannot
        # observe stale onboarding state (the gate re-reads per request).
        session.commit()
        # Launch the guided tutorial only when the operator asked for it and only
        # AFTER onboarding commits: a pre-onboarding link to /runs/{id}?tutorial=run
        # would hit the protected gate and bounce back to /setup. Redirect off the
        # seeder's own returned id (idempotent — the existing run when already
        # seeded) rather than a re-read, so a successful seed can never silently
        # fall through to /review.
        if wants_tutorial and seeded_run_id is not None:
            return RedirectResponse(
                f"/runs/{seeded_run_id}?tutorial=run", status_code=303
            )
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

    @protected.get("/search")
    def search(
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        q: str | None = None,
    ) -> Response:
        # The ranked semantic "Meaning" mode (issue #121): a distinct surface from
        # the chronological /runs browse, reading the embedding index the spine
        # builds. search_passages needs the session FACTORY, not this request's
        # session: it opens its own two short sessions (a settings gate, then one
        # read-only REPEATABLE READ snapshot that wraps all ranking arms so a
        # concurrent publish cannot straddle them). SessionDep still runs first,
        # so the factory is initialized and the basic-auth + onboarding gates on
        # the `protected` router have already passed. The feature/weights/indexing
        # state comes back on the page object, which the template renders honestly.
        settings: Settings = request.app.state.settings
        page = search_passages(
            request.app.state.session_factory,
            settings=settings,
            query=q or "",
        )
        return templates.TemplateResponse(
            request,
            "search.html",
            {
                "request": request,
                "page": page,
                "query": page.query,
                # Meaning search pairs with Runs (reached from its toggle), so the
                # Runs nav item stays lit rather than leaving the nav orphaned.
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
        except DomainPackError as exc:
            # Freeze-time snapshot collision (issue #84) / unresolvable pack (issue
            # #11): surface a plain-language 422, not the raw 500 the bare raise gave.
            raise HTTPException(
                status_code=422, detail=_submit_domain_pack_detail(exc)
            ) from exc
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
        except DomainPackError as exc:
            # Freeze-time snapshot collision (issue #84) / unresolvable pack (issue
            # #11): plain-language 422 rather than a raw 500. The snapshot is frozen
            # before any row/source_path write, so nothing is stranded.
            raise HTTPException(
                status_code=422, detail=_submit_domain_pack_detail(exc)
            ) from exc
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
                # The frozen sidecar title (issue #104), shown as the run's
                # display name above the raw path. Tolerant read: None when the
                # run has no sidecar (or a tampered one).
                "sidecar_title": title_from_snapshot(run.sidecar),
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
        read: bool = False,
        timestamps: bool = True,
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
        if read:
            # Read mode (issue #65): a timestamp-optional, shareable reading view
            # rendered purely server-side from the SAME presentation seam and the
            # SAME paragraph grouping the Markdown export uses — no island, no
            # second transcript truth. Grouping and timestamp formatting stay in
            # Python; the template only lays out the supplied rows. Jinja
            # autoescape (not the markdown-specific `_md_escape`) makes hostile
            # transcript text render literally in the HTML view.
            read_rows = [
                {
                    "speaker": para.speaker,
                    "lines": para.text.split("\n"),
                    "timespan": (
                        format_timespan(para.start_seconds, para.end_seconds)
                        if timestamps
                        else None
                    ),
                }
                for para in paragraphize_transcript(lines)
            ]
            return templates.TemplateResponse(
                request,
                "transcript.html",
                {
                    "request": request,
                    "run": run,
                    "lines": lines,
                    "read": True,
                    "read_rows": read_rows,
                    "read_timestamps": timestamps,
                    "text": variant,
                    "variants": list(TranscriptText),
                    "active_nav": "runs",
                },
            )
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
                "read": False,
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
        # Declared-rule reconciliation (#83): the run-level "declared but never
        # fired" panel. Computed ONCE per page load by replaying the frozen pack
        # over every segment's IMMUTABLE raw_text (one row per segment — never the
        # split-expanded lines, which would double-count a rule's applied segments).
        # [] when the snapshot is unavailable or declares no corrections, so the
        # panel simply does not render. Skip the replay (and the raw-text query)
        # entirely for an unclaimed/read-only view — the panel lives inside the
        # writable block and never renders there — and when the pack declares no
        # rules (run_reconciliation would short-circuit to [] anyway).
        rule_index = _load_run_rule_index(session, run_id) if token is not None else None
        if rule_index is not None and rule_index.rules:
            raw_texts = (
                session.execute(
                    select(TranscriptSegment.raw_text).where(
                        TranscriptSegment.pipeline_run_id == run_id
                    )
                )
                .scalars()
                .all()
            )
            island_props["reconciliation"] = run_reconciliation(rule_index, raw_texts)
        else:
            island_props["reconciliation"] = []
        # Operator annotation layer (issue #86): the annotations to render + the tag
        # universe + a tag-CRUD CSRF token + the server caps. Hydrated for the
        # review-stepper only (annotation authoring is claim-gated); the read-only
        # transcript-player never gets these. The list mirrors GET /annotations so a
        # write's JSON reconciles against the same shape the page hydrated with.
        annotations_payload = _annotations_payload(session, run_id, [])
        island_props["annotations"] = annotations_payload["annotations"]
        island_props["annotationTags"] = annotations_payload["tags"]
        island_props["annotationLimits"] = _annotation_limits()
        island_props["tagCsrf"] = mint_csrf_token(
            request.app.state.csrf_secret, CSRF_ANNOTATION_TAGS
        )
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
                # Guided-tutorial banner (issue #117 Phase B): CHECK_WORDS and EXPORT
                # both bind this transcript page. The banner lives above the body
                # (base.html), outside the island, so a stepper re-render never
                # clobbers it. The claim token flows through so the export step's
                # onward link stays writable.
                "tutorial": _tutorial_banner(
                    request,
                    session,
                    page=TutorialPage.TRANSCRIPT,
                    run_id=run_id,
                    token=token,
                ),
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
        run = _run_or_404(session, run_id)
        _reject_if_archived(run)
        # Stable across this operation: FAILED -> QUEUED keeps the stage, and
        # QUEUED can leave only for RUNNING at that same stage or CANCELLED.
        failed_stage = Stage(run.current_stage) if run.current_stage else None
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
        return _run_redirect(
            run_id, published=_publish_or_defer(run_id, stage=failed_stage)
        )

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
        # Finding D4: the export's URL fields are reduced to host-only, matching
        # the run-detail UI's provenance_host policy — the full page/uploader/
        # channel URLs (and raw's webpage_url) are the acquisition surface the
        # operator should not re-share by handing off an export. Descriptive
        # metadata (title, uploader/channel names, tags, description) is the
        # operator's own data and stays. `raw` is already an allowlisted scalar
        # subset (no signed URLs), so only its known URL keys need reducing.
        source_metadata = (
            {
                "source_kind": snapshot.source_kind,
                "title": snapshot.title,
                "uploader": snapshot.uploader,
                "uploader_url": provenance_host(snapshot.uploader_url),
                "channel": snapshot.channel,
                "channel_url": provenance_host(snapshot.channel_url),
                "description": snapshot.description,
                "upload_date": (snapshot.upload_date.isoformat() if snapshot.upload_date else None),
                "duration_seconds": snapshot.duration_seconds,
                "tags": snapshot.tags,
                "canonical_url": provenance_host(snapshot.canonical_url),
                "extractor": snapshot.extractor,
                "extractor_version": snapshot.extractor_version,
                "raw": _export_raw_host_only(snapshot.raw),
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
            # v2 (finding D4): source_metadata URL fields are host-only. v1 emitted
            # full uploader/channel/canonical URLs and raw.webpage_url verbatim.
            "schema_version": 2,
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
            if _segment_is_corrected(session, segment_id):
                reason = "this segment has an operator correction; clear it to split"
            elif trace_has_entries(segment.correction_trace):
                reason = "a domain-pack correction was applied here; splitting is disabled"
            else:
                reason = "no aligned word timings for this segment (or its text was enhanced)"
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
        # reading copy (issue #52). Only txt and md honor the flag.
        return _export_transcript(
            run_id, session, TranscriptFormat.TXT, text, timestamps=timestamps
        )

    @protected.get("/review/{run_id}/export.md")
    def export_transcript_md(
        run_id: uuid.UUID,
        operator: OperatorDep,
        session: SessionDep,
        text: str | None = None,
        timestamps: bool = True,
    ) -> Response:
        # Readable Markdown (issue #65): ## speaker headings + merged blockquotes.
        # ?timestamps=false drops the per-paragraph time range for a clean copy.
        return _export_transcript(
            run_id, session, TranscriptFormat.MARKDOWN, text, timestamps=timestamps
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

    # ---- Operator annotation layer (issue #86) --------------------------------
    # Thin handlers over voxint.adjudication.annotations: the service owns all
    # coordinate math, classification, idempotency, and staleness; routes only
    # parse the wire shape, gate auth/claim/CSRF, and map AnnotationError to HTTP
    # (docs/annotations.md). Reads need onboarding only; run-scoped writes need the
    # live review claim; global tag writes are CSRF-gated like run notes.

    @protected.get("/review/{run_id}/annotations")
    def list_annotations(
        run_id: uuid.UUID,
        operator: OperatorDep,
        session: SessionDep,
        tag: Annotated[list[uuid.UUID] | None, Query()] = None,
    ) -> JSONResponse:
        """The run's live annotations (transcript order) resolved against the current
        render, plus the tag universe. Onboarding-auth only — no claim, so a reviewer
        can read annotations without holding the slot. Repeated ``?tag=`` is an
        OR-union filter, identically in the panel and exports; an unknown tag id in
        the filter fails closed (404)."""
        _run_or_404(session, run_id)
        tag_ids = tag or []
        _require_filter_tags_exist(session, tag_ids)
        return JSONResponse(_annotations_payload(session, run_id, tag_ids))

    @protected.get("/review/{run_id}/annotations/export.md")
    def export_annotations_md(
        run_id: uuid.UUID,
        operator: OperatorDep,
        session: SessionDep,
        tag: Annotated[list[uuid.UUID] | None, Query()] = None,
    ) -> Response:
        """All (filtered) highlights as Markdown pull-quotes in canonical transcript
        order (issue #86), joined by a thematic-break separator. Onboarding-auth only,
        NO claim — Copy is a read, available in a read-only review tab. Repeated
        ``?tag=`` is the same OR-union filter as the panel; an unknown tag id is 404.
        Fails ATOMICALLY with 409 ``X-Voxint-Conflict: stale`` if ANY matched highlight
        is stale (it is never silently omitted). An empty match is an empty body (200)."""
        run = _run_or_404(session, run_id)
        tag_ids = tag or []
        _require_filter_tags_exist(session, tag_ids)
        rows = annotations_for_run(session, run_id, tag_ids=tag_ids or None)
        if not rows:
            return Response(content="", media_type=ANNOTATION_MEDIA_TYPES["md"])
        lines = attributed_transcript(session, run_id, text=TranscriptText.CORRECTED)
        covered = load_covered_segments(session, run_id)
        resolved = {
            r.annotation_id: r
            for r in resolve_annotation_spans(
                lines, covered, [stored_anchor_from_row(row) for row in rows]
            )
        }
        tags_by_id = tags_for_annotations(session, [row.id for row in rows])
        source_title = _run_source_title(run)
        ordered = sorted(rows, key=lambda row: resolved_order_key(resolved[row.id]))
        quotes = [
            _pull_quote_markdown(
                resolved[row.id],
                lines,
                source_title=source_title,
                tags=[t.name for t in tags_by_id.get(row.id, [])],
                note=row.note,
            )
            for row in ordered
        ]
        return Response(
            content=ANNOTATION_BULK_SEPARATOR.join(quotes),
            media_type=ANNOTATION_MEDIA_TYPES["md"],
        )

    @protected.get("/review/{run_id}/annotations/{annotation_id}/export.md")
    def export_annotation_md(
        run_id: uuid.UUID,
        annotation_id: uuid.UUID,
        operator: OperatorDep,
        session: SessionDep,
    ) -> Response:
        """One highlight as a Markdown pull-quote (issue #86). Onboarding-auth only,
        NO claim. A foreign, forged, or soft-deleted id is 404 (fail closed). A stale
        highlight is refused 409 ``X-Voxint-Conflict: stale`` — the operator refreshes
        or re-anchors it first."""
        run = _run_or_404(session, run_id)
        try:
            row = live_annotation_or_404(session, run_id, annotation_id)
        except AnnotationError as exc:
            raise _annotation_http_error(exc) from exc
        lines = attributed_transcript(session, run_id, text=TranscriptText.CORRECTED)
        covered = load_covered_segments(session, run_id)
        resolved = resolve_annotation_spans(lines, covered, [stored_anchor_from_row(row)])[0]
        tags = [t.name for t in tags_for_annotations(session, [row.id]).get(row.id, [])]
        markdown = _pull_quote_markdown(
            resolved,
            lines,
            source_title=_run_source_title(run),
            tags=tags,
            note=row.note,
        )
        return Response(content=markdown, media_type=ANNOTATION_MEDIA_TYPES["md"])

    @protected.post("/review/{run_id}/annotations/export/live.md")
    def export_live_pull_quote(
        run_id: uuid.UUID,
        operator: OperatorDep,
        session: SessionDep,
        start_segment_id: Annotated[uuid.UUID, Form()],
        start_offset: Annotated[int, Form()],
        end_segment_id: Annotated[uuid.UUID, Form()],
        end_offset: Annotated[int, Form()],
        client_quote: Annotated[str, Form()],
        start_child_word_start: Annotated[int | None, Form()] = None,
        start_child_word_end: Annotated[int | None, Form()] = None,
        end_child_word_start: Annotated[int | None, Form()] = None,
        end_child_word_end: Annotated[int | None, Form()] = None,
        note: Annotated[str | None, Form()] = None,
        tags: Annotated[list[uuid.UUID] | None, Form()] = None,
    ) -> Response:
        """A live pull-quote for an UNSAVED selection (issue #86, docs/annotations.md):
        classify + validate exactly as create, but persist NOTHING and return the
        Markdown. Onboarding-auth only — no claim, no nonce, no CSRF, because nothing is
        written. Same caps/validation as create (422 on a bad anchor or cap; 409 stale on
        a drifted client quote). The optional ``note``/``tags`` are echoed into the quote
        trailer, never stored; unknown tag ids are 404."""
        run = _run_or_404(session, run_id)
        payload = _capture_payload_from_form(
            start_segment_id,
            start_offset,
            start_child_word_start,
            start_child_word_end,
            end_segment_id,
            end_offset,
            end_child_word_start,
            end_child_word_end,
            client_quote,
        )
        try:
            tag_names = resolve_tag_names(session, list(tags) if tags else [])
            normalized_note = normalize_note(note)
            derived, covered = derive_live_anchor(session, run_id, payload)
        except AnnotationError as exc:
            raise _annotation_http_error(exc) from exc
        lines = attributed_transcript(session, run_id, text=TranscriptText.CORRECTED)
        anchor = stored_anchor_from_derived(derived, uuid.uuid4())
        resolved = resolve_annotation_spans(lines, covered, [anchor])[0]
        markdown = _pull_quote_markdown(
            resolved,
            lines,
            source_title=_run_source_title(run),
            tags=tag_names,
            note=normalized_note,
        )
        return Response(content=markdown, media_type=ANNOTATION_MEDIA_TYPES["md"])

    @protected.post("/review/{run_id}/annotations")
    def create_annotation(
        run_id: uuid.UUID,
        session: SessionDep,
        operator: OperatorDep,
        token: Annotated[uuid.UUID, Form()],
        nonce: Annotated[str, Form(min_length=8, max_length=64)],
        start_segment_id: Annotated[uuid.UUID, Form()],
        start_offset: Annotated[int, Form()],
        end_segment_id: Annotated[uuid.UUID, Form()],
        end_offset: Annotated[int, Form()],
        client_quote: Annotated[str, Form()],
        color_index: Annotated[int, Form()],
        start_child_word_start: Annotated[int | None, Form()] = None,
        start_child_word_end: Annotated[int | None, Form()] = None,
        end_child_word_start: Annotated[int | None, Form()] = None,
        end_child_word_end: Annotated[int | None, Form()] = None,
        note: Annotated[str | None, Form()] = None,
        tags: Annotated[list[uuid.UUID] | None, Form()] = None,
    ) -> JSONResponse:
        """Create an annotation (the sole create path). Claim-gated and idempotent by
        the client ``nonce``: a same-payload replay returns the original row, a
        different payload is a 409 idempotency conflict. The server classifies the
        anchor kind and derives the quote/hash/seconds; the client never picks the
        kind. Returns the created annotation's island shape (201)."""
        run = _verify_annotation_claim(session, run_id, token)
        _reject_if_archived(run)
        payload = _capture_payload_from_form(
            start_segment_id,
            start_offset,
            start_child_word_start,
            start_child_word_end,
            end_segment_id,
            end_offset,
            end_child_word_start,
            end_child_word_end,
            client_quote,
        )
        try:
            row = capture_annotation(
                session,
                run_id=run_id,
                payload=payload,
                operator=operator,
                nonce=nonce,
                color_index=color_index,
                note=note,
                tag_ids=list(tags) if tags else None,
            )
        except AnnotationError as exc:
            raise _annotation_http_error(exc) from exc
        return JSONResponse(_annotation_shapes(session, run_id, [row])[0], status_code=201)

    @protected.patch("/review/{run_id}/annotations/{annotation_id}")
    def patch_annotation(
        run_id: uuid.UUID,
        annotation_id: uuid.UUID,
        session: SessionDep,
        operator: OperatorDep,
        token: Annotated[uuid.UUID, Form()],
        op: Annotated[str, Form()],
        color_index: Annotated[int | None, Form()] = None,
        note: Annotated[str | None, Form()] = None,
        tags: Annotated[list[uuid.UUID] | None, Form()] = None,
        start_segment_id: Annotated[uuid.UUID | None, Form()] = None,
        start_offset: Annotated[int | None, Form()] = None,
        end_segment_id: Annotated[uuid.UUID | None, Form()] = None,
        end_offset: Annotated[int | None, Form()] = None,
        start_child_word_start: Annotated[int | None, Form()] = None,
        start_child_word_end: Annotated[int | None, Form()] = None,
        end_child_word_start: Annotated[int | None, Form()] = None,
        end_child_word_end: Annotated[int | None, Form()] = None,
        client_quote: Annotated[str | None, Form()] = None,
    ) -> JSONResponse:
        """Mutate an existing annotation. ``op`` is one of three mutually-exclusive
        operations (docs/annotations.md): ``edit`` replaces metadata (colour/note/
        tags), ``refresh`` re-derives the quote/hash/seconds when the anchor still
        deterministically identifies its span, and ``reanchor`` atomically replaces
        the anchor from a fresh capture payload. Claim-gated; a deleted or foreign id
        is 404. Returns the updated annotation's island shape."""
        run = _verify_annotation_claim(session, run_id, token)
        _reject_if_archived(run)
        try:
            if op == "edit":
                if color_index is None:
                    raise HTTPException(status_code=422, detail="edit requires color_index")
                row = update_annotation(
                    session,
                    run_id=run_id,
                    annotation_id=annotation_id,
                    color_index=color_index,
                    note=note,
                    tag_ids=list(tags) if tags else None,
                )
            elif op == "refresh":
                row = refresh_annotation(session, run_id=run_id, annotation_id=annotation_id)
            elif op == "reanchor":
                if (
                    start_segment_id is None
                    or start_offset is None
                    or end_segment_id is None
                    or end_offset is None
                    or client_quote is None
                ):
                    raise HTTPException(
                        status_code=422,
                        detail="reanchor requires a full capture payload",
                    )
                payload = _capture_payload_from_form(
                    start_segment_id,
                    start_offset,
                    start_child_word_start,
                    start_child_word_end,
                    end_segment_id,
                    end_offset,
                    end_child_word_start,
                    end_child_word_end,
                    client_quote,
                )
                row = reanchor_annotation(
                    session,
                    run_id=run_id,
                    annotation_id=annotation_id,
                    payload=payload,
                )
            else:
                raise HTTPException(
                    status_code=422,
                    detail=f"unknown op {op!r}; expected edit, refresh, or reanchor",
                )
        except AnnotationError as exc:
            raise _annotation_http_error(exc) from exc
        return JSONResponse(_annotation_shapes(session, run_id, [row])[0])

    @protected.delete("/review/{run_id}/annotations/{annotation_id}")
    def delete_annotation(
        run_id: uuid.UUID,
        annotation_id: uuid.UUID,
        session: SessionDep,
        operator: OperatorDep,
        token: Annotated[uuid.UUID, Form()],
    ) -> Response:
        """Soft-delete an annotation (idempotent): a repeat DELETE of an already-
        deleted row is a no-op 204, never a 404 — the row still exists and a create
        replay still finds it. An unknown/foreign id is 404. Claim-gated."""
        run = _verify_annotation_claim(session, run_id, token)
        _reject_if_archived(run)
        try:
            soft_delete_annotation(session, run_id=run_id, annotation_id=annotation_id)
        except AnnotationError as exc:
            raise _annotation_http_error(exc) from exc
        return Response(status_code=204)

    @protected.get("/annotations/tags")
    def list_annotation_tags(operator: OperatorDep, session: SessionDep) -> JSONResponse:
        """The global tag universe (all tags, archived included), in display order.
        Onboarding-auth only; tags are not run-scoped."""
        return JSONResponse({"tags": [_tag_shape(t) for t in list_tags(session)]})

    @protected.post("/annotations/tags")
    def create_annotation_tag(
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        name: Annotated[str, Form()],
        color: Annotated[int, Form()],
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> JSONResponse:
        """Create a global tag (name + palette colour). CSRF-gated like run notes —
        tag writes have no run or claim context. A normalized-name duplicate is a 409;
        a blank/over-cap name or bad colour is a 422. Returns the created tag (201)."""
        _require_csrf(request, CSRF_ANNOTATION_TAGS, csrf_token)
        try:
            tag = create_tag(session, name=name, color=color)
        except AnnotationError as exc:
            raise _annotation_http_error(exc) from exc
        return JSONResponse(_tag_shape(tag), status_code=201)

    @protected.patch("/annotations/tags/{tag_id}")
    def update_annotation_tag(
        tag_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        name: Annotated[str | None, Form()] = None,
        color: Annotated[int | None, Form()] = None,
        archived: Annotated[bool | None, Form()] = None,
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> JSONResponse:
        """Rename / recolour / archive / restore a tag. Each field is independently
        optional (absent leaves it untouched; ``archived`` is a tri-state). CSRF-gated.
        A rename colliding with a different tag is a 409; an unknown id is 404."""
        _require_csrf(request, CSRF_ANNOTATION_TAGS, csrf_token)
        try:
            tag = update_tag(session, tag_id=tag_id, name=name, color=color, archived=archived)
        except AnnotationError as exc:
            raise _annotation_http_error(exc) from exc
        return JSONResponse(_tag_shape(tag))

    # ---- Prometheus metrics ----------------------------------------------------
    # Read-only aggregate exposition on the *protected* router: Prometheus scrapes
    # it with basic_auth, so the "everything but /healthz authenticates" invariant
    # holds without a new flag or token path. The one windowed series
    # (voxint_runs_created_24h) bakes its window into the metric name.

    def _resource_snapshot(request: Request) -> ResourceSnapshot:
        """The cached hardware snapshot for a render, guarded to never raise.

        ``/metrics`` and the dashboard/resource pages read this cached value
        (never ``force``): a 15s poll across tabs shares one probe, and a probe
        failure degrades to an empty snapshot rather than breaking the page.
        """
        try:
            return collect_resource_status(request.app.state.settings)
        except Exception:
            # Telemetry is advisory; a probe failure must never break a render.
            return ResourceSnapshot(gpus=(), services=(), collected_age_seconds=0.0)

    @protected.get("/metrics")
    def metrics(request: Request, operator: OperatorDep, session: SessionDep) -> Response:
        now = datetime.now(UTC)
        stats = collect_stats(session, since=now - DEFAULT_WINDOW, now=now)
        # Append the hardware gauges from the same cached snapshot the dashboard
        # and resource page render, so a scrape and the UI cannot disagree.
        body = render_prometheus(stats) + render_resource_prometheus(
            _resource_snapshot(request)
        )
        return Response(
            content=body,
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
        # The 15s htmx poll re-requests this route and swaps ONLY the metrics
        # fragment; the task-card queries (issue #117 Phase C) feed the static full
        # page, never that fragment, so skip them on the poll rather than paying for
        # two queries whose results would be discarded.
        is_htmx = bool(request.headers.get("HX-Request"))
        context = {
            "request": request,
            "stats": stats,
            # Curated hardware strip (W3): rendered inside the htmx-swapped
            # fragment so it refreshes on the same 15s poll as the run figures.
            "resource_strip": build_resource_strip(_resource_snapshot(request)),
            "active_nav": "dashboard",
            # Iterate the enum (not the sparse status_counts map) so the status
            # table renders in a stable order and zero-fills empty statuses, the
            # same contract format_stats_text/render_prometheus hold.
            "run_statuses": list(RunStatus),
            # Carry the accepted window through the 15s htmx poll so a custom
            # ?since= isn't lost on the first refresh. Only echo a value we
            # actually honored (an invalid one falls back to the default, so we
            # drop it from the poll URL too).
            "since_param": "" if since_invalid or not raw_since else raw_since,
            "since_invalid": since_invalid,
        }
        if not is_htmx:
            # The count of runs actually eligible for review, sharing the queue's
            # own predicate (issue #117). It powers the static "Continue review (N)"
            # task card; review_backlog_count derives from adjudication_queue and
            # cannot drift from it. It is deliberately the ONLY copy of this number
            # on the page — no live stat-card duplicate that could contradict it.
            context["review_backlog"] = review_backlog_count(session)
            # The "Last finished run" task card: newest run by terminal-stage
            # completion, None when nothing has finished (honest empty state).
            context["last_completed"] = latest_completed_run(session)
        template = "fragments/dashboard_metrics.html" if is_htmx else "dashboard.html"
        return templates.TemplateResponse(request, template, context)

    # ---- Hardware resource page (hardware-aware W3) -----------------------------
    # The fuller live view behind the dashboard strip: the aggregated GPU card
    # (utilization/VRAM/temperature labeled INSTANTANEOUS, peak temp + throttle
    # events labeled cumulative-since-restart), per-service admission, and the
    # curated warnings. Reads the same cached snapshot as /metrics and the strip,
    # so the three cannot disagree. An htmx poll swaps just the fragment, like
    # /dashboard. Protected like every non-/healthz page.

    @protected.get("/resources")
    def resources(request: Request, operator: OperatorDep) -> Response:
        snapshot = _resource_snapshot(request)
        context = {
            "request": request,
            "snapshot": snapshot,
            "resource_strip": build_resource_strip(snapshot),
            "active_nav": "resources",
        }
        template = (
            "fragments/resource_status.html"
            if request.headers.get("HX-Request")
            else "resources.html"
        )
        return templates.TemplateResponse(request, template, context)

    # ---- Settings + guided-tutorial lifecycle (issue #3, slice 6) --------------
    # The persistent, re-runnable entry point: re-open the setup wizard, and
    # start / replay / complete the guided tutorial. All @protected (an
    # un-onboarded operator is bounced to /setup by the gate). The two POSTs verify
    # CSRF_SETTINGS and 409 when no tutorial run is available, so a stray token can
    # never "complete" or "replay" an unseeded tutorial.

    def _settings_context(
        request: Request,
        session: Session,
        folder_path: str | None = None,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Shared context for the settings page (GET render + POST re-render).

        Carries the effective LLM state (issue #10) — enablement over env, the
        effective endpoint, and whether an effective key is present and where it
        comes from — never the key value — plus the Features-section tri-state flag
        rows (issue #62). ``overrides`` lets a POST re-render carry a section
        ``*_error`` and (for Features) ``features_submitted``, the operator's
        submitted radio selections, so a rejected save re-renders their choices.

        ``folder_path`` overrides the folder-panel browse location (defaults to
        ``?path=`` from the query string) so a non-HTMX folder mutation that fails
        can re-render the full page at the submitted path instead of the root.
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
        # Semantic search section (issue #121): two tri-state rows (the feature +
        # its autogenerate rider). On an invariant-rejected save, render the
        # operator's submitted choices back (``semantic_index_submitted``); otherwise
        # the stored raw tri-state.
        semantic_submitted: dict[str, str] | None = overrides.pop(
            "semantic_index_submitted", None
        )
        # Sources & research section (issue #76). On an invariant/format-rejected
        # save, render the operator's submitted choices back (``web_research_submitted``
        # — the four non-secret fields only, never the key); otherwise the stored raw
        # tri-state / override values.
        wr_submitted: dict[str, str] | None = overrides.pop("web_research_submitted", None)
        wr_base_value, wr_base_default = str_flag_form_field(row, settings, "web_search_base_url")
        wr_domains_value, wr_domains_default = str_flag_form_field(
            row, settings, "source_authority_domains"
        )
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
            # Bounded, non-secret message when a UI-triggered tutorial seed fails
            # (issue #75); None on a normal render. Overridable via ``overrides``.
            "tutorial_error": None,
            "feature_flags": feature_flags,
            "features_errors": [],
            # Semantic search (issue #121): the two tri-state toggles (raw stored
            # tri-state, or submitted choice on re-render), their env defaults for the
            # "inherit" label, and whether the embedding weights are actually
            # installed — an honest note when the feature is on but weights are absent,
            # since enabling it then cannot answer a query.
            "semantic_index_enabled_state": (
                semantic_submitted.get("semantic_index_enabled", "inherit")
                if semantic_submitted is not None
                else feature_flag_state(row, "semantic_index_enabled")
            ),
            "semantic_index_enabled_env_default": bool(settings.semantic_index_enabled),
            "semantic_index_autogenerate_state": (
                semantic_submitted.get("semantic_index_autogenerate", "inherit")
                if semantic_submitted is not None
                else feature_flag_state(row, "semantic_index_autogenerate")
            ),
            "semantic_index_autogenerate_env_default": bool(
                settings.semantic_index_autogenerate
            ),
            "semantic_index_weights_available": minilm_artifacts_available(),
            "semantic_index_errors": [],
            # Sources & research (issue #76): the two web-research toggles (raw
            # tri-state, or submitted choice on re-render), the endpoint override +
            # env placeholder, the credential status (present + source, never the
            # value), and the authority-domains override + env placeholder.
            "web_research_master_state": (
                wr_submitted.get("voxint_web_research", "inherit")
                if wr_submitted is not None
                else feature_flag_state(row, "voxint_web_research")
            ),
            "web_research_master_env_default": bool(settings.voxint_web_research),
            "web_research_producer_state": (
                wr_submitted.get("enrichment_web_research_enabled", "inherit")
                if wr_submitted is not None
                else feature_flag_state(row, "enrichment_web_research_enabled")
            ),
            "web_research_producer_env_default": bool(settings.enrichment_web_research_enabled),
            "web_search_base_url": (
                wr_submitted.get("web_search_base_url", "")
                if wr_submitted is not None
                else wr_base_value
            ),
            "web_search_base_url_default": wr_base_default,
            "web_search_key_present": bool(resolve_effective_web_search_api_key(row, settings)),
            "web_search_key_source": effective_web_search_key_source(row, settings),
            "source_authority_domains": (
                wr_submitted.get("source_authority_domains", "")
                if wr_submitted is not None
                else wr_domains_value
            ),
            "source_authority_domains_default": wr_domains_default,
            "web_research_errors": [],
        }
        # Media folders + per-folder domain packs (issue #63). Always present on the
        # Settings page; the untrusted ?path= is revalidated in list_media_subdirs.
        context.update(
            _folder_panel_context(
                session,
                settings,
                action_prefix="/settings/folders",
                csrf=context["csrf_settings"],
                path=folder_path if folder_path is not None else (
                    request.query_params.get("path") or "."
                ),
            )
        )
        # Watch-folder ingest (issue #60): the tri-state toggle beside the folder
        # panel, its env default (for the "inherit" label), the EFFECTIVE gate (drives
        # the on/off wording), and the latest sweep summary for the status line.
        context["watch_folder_state"] = feature_flag_state(row, "watch_folder_enabled")
        context["watch_folder_env_default"] = bool(settings.watch_folder_enabled)
        context["watch_folder_effective"] = resolve_effective_watch_folder_enabled(row, settings)
        last_sweep = row.watch_folder_last_sweep if row is not None else None
        context["watch_folder_last_sweep"] = last_sweep
        # Parse the stored ISO timestamp into a datetime so the template can format it
        # (the JSON column holds a string); None if never run or malformed.
        checked_at: datetime | None = None
        if isinstance(last_sweep, dict) and last_sweep.get("completed_at"):
            try:
                checked_at = datetime.fromisoformat(str(last_sweep["completed_at"]))
            except ValueError:
                checked_at = None
        context["watch_folder_checked_at"] = checked_at
        context["watch_folder_error"] = None
        # Corrections authoring (issue #84): the operator's current rules (for the
        # no-JS read-only fallback) plus the JSON props the corrections-editor
        # island hydrates from — the rules, the CSRF-guarded save action, and the
        # #80 bounds so the client can hint before the server (authoritative) gate.
        stored_corrections = (
            list(row.corrections) if row is not None and row.corrections else []
        )
        context["corrections"] = stored_corrections
        context["corrections_props"] = {
            "rules": stored_corrections,
            "action": "/settings/corrections",
            "csrfToken": context["csrf_settings"],
            "limits": {
                "maxRules": MAX_RULES_PER_PACK,
                "maxMatchChars": MAX_MATCH_CHARS,
                "maxReplacementChars": MAX_REPLACEMENT_CHARS,
            },
        }
        # Initialized like every sibling error (llm_error, watch_folder_error) so the
        # template never leans on Jinja's lenient Undefined; overridden on a rejected save.
        context["corrections_error"] = None
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
        llm_bundled_enabled: Annotated[str, Form()] = "inherit",
        ytdlp_enabled: Annotated[str, Form()] = "inherit",
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> Response:
        _require_csrf(request, CSRF_SETTINGS, csrf_token)
        settings: Settings = request.app.state.settings
        # Every flag the Features section RENDERS (_FEATURE_FLAG_META) must have a
        # matching Form param here and a key in `submitted`: a rendered radio whose
        # name is not declared is silently dropped by Starlette, and
        # _persist_feature_flags then reads it as "inherit" and clears any stored
        # override. test_features_route_declares_every_flag guards this pairing.
        submitted = {
            "enrichment_names_enabled": enrichment_names_enabled,
            "enrichment_names_llm_enabled": enrichment_names_llm_enabled,
            "enrichment_run_assets_enabled": enrichment_run_assets_enabled,
            "enrichment_run_assets_autogenerate": enrichment_run_assets_autogenerate,
            "llm_bundled_enabled": llm_bundled_enabled,
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

    @protected.post("/settings/semantic")
    def settings_semantic(
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        semantic_index_enabled: Annotated[str, Form()] = "inherit",
        semantic_index_autogenerate: Annotated[str, Form()] = "inherit",
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> Response:
        """Save the Semantic search section's two tri-state toggles (issue #121).

        Scoped and CSRF-gated like every settings section. Both rendered radios have
        a matching Form param here (a rendered radio whose name is undeclared is
        dropped by Starlette and then read as "inherit", silently clearing an
        override — test_semantic_route_declares_every_flag guards the pairing). On the
        one reachable invariant violation (autogenerate on with the feature off)
        nothing is written and the operator's choices re-render with a plain message.
        """
        _require_csrf(request, CSRF_SETTINGS, csrf_token)
        settings: Settings = request.app.state.settings
        submitted = {
            "semantic_index_enabled": semantic_index_enabled,
            "semantic_index_autogenerate": semantic_index_autogenerate,
        }
        errors = _persist_semantic_index(session, settings, submitted=submitted)
        if errors:
            return templates.TemplateResponse(
                request,
                "settings.html",
                _settings_context(
                    request,
                    session,
                    semantic_index_errors=errors,
                    semantic_index_submitted=submitted,
                ),
            )
        session.commit()
        return RedirectResponse("/settings", status_code=303)

    @protected.post("/settings/corrections")
    def settings_corrections(
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        rules: Annotated[str, Form()],
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> Response:
        """Replace the operator's console-authored correction rules (issue #84).

        The corrections-editor island owns add/edit/remove/reorder client-side and
        submits the FULL ordered list as a JSON string in the ``rules`` form field
        (replace-all semantics, like vocabulary). The whole set is validated through
        the SAME #80 gate a pack gets — bounds, NUL/control-char rejection, unique
        ids, boundary-aware idempotence — plus a union check against the current
        default pack, so a rule the console accepts is one the frozen-per-run
        pipeline can apply. On any violation NOTHING is written: the island (Accept:
        application/json) gets a 422 with the plain-language message and the
        offending row; a JS-off submit re-renders the page with the message.
        """
        _require_csrf(request, CSRF_SETTINGS, csrf_token)
        settings: Settings = request.app.state.settings
        wants_json = "application/json" in (request.headers.get("accept") or "")
        try:
            raw_items = json.loads(rules)
        except json.JSONDecodeError:
            message = "The corrections payload was not valid JSON."
            if wants_json:
                return JSONResponse(
                    {"ok": False, "error": message, "row": None}, status_code=422
                )
            return templates.TemplateResponse(
                request,
                "settings.html",
                _settings_context(request, session, corrections_error=message),
                status_code=422,
            )
        # Candidate → validate (field + cross-rule + default-pack union) → ONE
        # mutation. get_or_create only after validation, so a rejected save (which
        # the API session would commit on a 200/422 render) writes nothing.
        # The union check needs the default pack; if the registry is unreadable
        # (bad DOMAIN_PACKS_DIR / DOMAIN_PACK_PATH) degrade like the folder panel
        # (_set_folder_pack) — refuse with guidance, never 500. Skipping the union
        # silently would be worse (it would let a colliding rule save).
        try:
            pack_corrections = (
                default_domain_pack(settings).to_mapping().get("corrections")
            )
        except DomainPackError:
            message = (
                "Domain packs can't be loaded right now, so corrections can't be "
                "validated — check your domain-pack configuration and try again."
            )
            if wants_json:
                return JSONResponse(
                    {"ok": False, "error": message, "row": None}, status_code=422
                )
            return templates.TemplateResponse(
                request,
                "settings.html",
                _settings_context(request, session, corrections_error=message),
                status_code=422,
            )
        try:
            normalized = normalize_operator_corrections(
                raw_items, pack_corrections=pack_corrections
            )
        except OperatorCorrectionError as exc:
            if wants_json:
                return JSONResponse(
                    {"ok": False, "error": exc.message, "row": exc.row},
                    status_code=422,
                )
            return templates.TemplateResponse(
                request,
                "settings.html",
                _settings_context(request, session, corrections_error=exc.message),
                status_code=422,
            )
        row = get_or_create(session, llm_enabled_default=settings.llm_enabled)
        row.corrections = normalized
        session.commit()
        if wants_json:
            return JSONResponse({"ok": True, "corrections": normalized})
        return RedirectResponse("/settings", status_code=303)

    @protected.post("/settings/watch-folder")
    def settings_watch_folder(
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        watch_folder_enabled: Annotated[str, Form()] = "inherit",
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> Response:
        """Set the watch-folder ingest runtime override (issue #60).

        A single tri-state toggle beside the media-folders panel: On/Off/inherit map
        to the nullable ``app_settings.watch_folder_enabled`` column as
        True/False/None (None = inherit the installation default). There is no
        cross-flag invariant, so a valid choice writes the one column and redirects;
        an unrecognized value (a stale client / hand-crafted POST) is rejected rather
        than silently coerced, mirroring ``_persist_feature_flags``.
        """
        _require_csrf(request, CSRF_SETTINGS, csrf_token)
        settings: Settings = request.app.state.settings
        if watch_folder_enabled not in _FEATURE_FLAG_CHOICES:
            return templates.TemplateResponse(
                request,
                "settings.html",
                _settings_context(
                    request,
                    session,
                    watch_folder_error=(
                        "Unrecognized watch-folder setting — choose On, Off, or Use "
                        "installation setting."
                    ),
                ),
            )
        row = get_or_create(session, llm_enabled_default=settings.llm_enabled)
        row.watch_folder_enabled = (
            None if watch_folder_enabled == "inherit" else (watch_folder_enabled == "on")
        )
        session.commit()
        return RedirectResponse("/settings", status_code=303)

    @protected.post("/settings/web-research")
    def settings_web_research(
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        voxint_web_research: Annotated[str, Form()] = "inherit",
        enrichment_web_research_enabled: Annotated[str, Form()] = "inherit",
        web_search_base_url: Annotated[str, Form()] = "",
        web_search_api_key: Annotated[str, Form()] = "",
        remove_web_search_api_key: Annotated[bool, Form()] = False,
        source_authority_domains: Annotated[str, Form()] = "",
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> Response:
        _require_csrf(request, CSRF_SETTINGS, csrf_token)
        settings: Settings = request.app.state.settings
        # Candidate → validate (all error sources) → ONE mutation. On any violation
        # nothing is written and the operator's choices are re-rendered with the
        # plain-language message(s) (issue #76). The re-render dict carries the four
        # NON-SECRET fields only — the API key is never echoed back.
        errors = _persist_web_research(
            session,
            settings,
            submitted_master=voxint_web_research,
            submitted_producer=enrichment_web_research_enabled,
            raw_base_url=web_search_base_url,
            raw_key=web_search_api_key,
            remove_key=remove_web_search_api_key,
            raw_domains=source_authority_domains,
        )
        if errors:
            submitted = {
                "voxint_web_research": voxint_web_research,
                "enrichment_web_research_enabled": enrichment_web_research_enabled,
                "web_search_base_url": web_search_base_url,
                "source_authority_domains": source_authority_domains,
            }
            return templates.TemplateResponse(
                request,
                "settings.html",
                _settings_context(
                    request, session, web_research_errors=errors, web_research_submitted=submitted
                ),
            )
        session.commit()
        return RedirectResponse("/settings", status_code=303)

    # ---- Settings → Media folders + domain packs (issue #63) ---------------
    # The same folder browser as the wizard, mounted on the protected router with
    # CSRF_SETTINGS. Shares _folder_panel_response / _folder_panel_context.

    @protected.get("/settings/folders/browse")
    def settings_folders_browse(
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        path: Annotated[str, Query(max_length=4096)] = ".",
    ) -> Response:
        settings: Settings = request.app.state.settings
        csrf = mint_csrf_token(request.app.state.csrf_secret, CSRF_SETTINGS)
        context = _folder_panel_context(
            session, settings, action_prefix="/settings/folders", csrf=csrf, path=path
        )
        response = templates.TemplateResponse(
            request, "fragments/folder_panel.html", {"request": request, **context}
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @protected.post("/settings/folders")
    def settings_folders(
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        action: Annotated[str, Form(max_length=16)] = "",
        folder: Annotated[str, Form(max_length=4096)] = "",
        pack: Annotated[str | None, Form(max_length=200)] = None,
        path: Annotated[str, Form(max_length=4096)] = ".",
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> Response:
        _require_csrf(request, CSRF_SETTINGS, csrf_token)
        settings: Settings = request.app.state.settings
        error = _apply_folder_mutation(
            session, settings, action=action, folder=folder, pack=pack
        )
        if error is None:
            session.commit()
        else:
            session.rollback()
        redirect = "/settings?" + urlencode({"path": path}) + "#folders"

        def _error_page(message: str) -> Response:
            return templates.TemplateResponse(
                request,
                "settings.html",
                _settings_context(request, session, folder_path=path, folder_error=message),
            )

        return _folder_panel_response(
            request,
            session,
            settings,
            action_prefix="/settings/folders",
            csrf_action=CSRF_SETTINGS,
            path=path,
            error=error,
            redirect_url=redirect,
            error_page=_error_page,
        )

    @protected.post("/settings/tutorial/seed")
    def tutorial_seed(
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> Response:
        _require_csrf(request, CSRF_SETTINGS, csrf_token)
        # Idempotent seed from the UI so a non-technical operator never needs the
        # CLI (issue #75). On a classified storage/asset failure, roll back the
        # flushed partial rows (the request session commits on any 200) and
        # re-render the page with bounded, non-secret guidance — writing nothing.
        settings: Settings = request.app.state.settings
        run_id, error = _try_seed_tutorial(session, settings)
        if error is not None:
            session.rollback()
            return templates.TemplateResponse(
                request,
                "settings.html",
                _settings_context(request, session, tutorial_error=error),
            )
        session.commit()
        return RedirectResponse(f"/runs/{run_id}?tutorial=run", status_code=303)

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
