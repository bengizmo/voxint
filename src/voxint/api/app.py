"""FastAPI application: the review console (queue, workbench, media), health.

Adjudication is post-hoc: only COMPLETED runs appear in the queue, and nothing
here touches the pipeline state machine. Every route except ``/healthz`` sits
behind single-operator basic auth; mutations additionally require the live
claim token, and each rendered form carries a fresh server-issued nonce that
becomes the ledger idempotency key — an htmx retry of the same form is a
harmless replay, while a new submission is a new decision (corrections are
appends; the newest ruling per label wins at read time).
"""

import logging
import secrets
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, BinaryIO, cast

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.templating import Jinja2Templates
from sqlalchemy import exists, select
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
    CSRF_CLAIM,
    CSRF_FETCH,
    CSRF_REQUEUE,
    CSRF_ROSTER_ARCHIVE,
    CSRF_ROSTER_EMBEDDING_DELETE,
    CSRF_ROSTER_MERGE,
    CSRF_ROSTER_RENAME,
    CSRF_ROSTER_RESTORE,
    CSRF_SETTINGS,
    CSRF_SETUP,
    CSRF_SUBMIT,
    mint_csrf_token,
    verify_csrf_token,
)
from voxint.api.health_probe import probe_services
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
    normalize_llm_base_url,
    normalize_llm_model,
    normalize_media_folders,
    normalize_vocabulary,
    parse_step,
    scan_media_folders,
    validate_llm_enable,
)
from voxint.app_settings import (
    clear_tutorial_completion,
    complete_onboarding,
    get_app_settings,
    get_or_create,
    is_onboarded,
    mark_tutorial_complete,
    ready_tutorial_run_id,
)
from voxint.config import Settings, get_settings, llm_budget_fits_stage_lease
from voxint.db.models import (
    Decision,
    PipelineRun,
    RunStatus,
    Speaker,
    StageRun,
    TranscriptSegment,
)
from voxint.db.session import build_engine, build_session_factory, session_scope
from voxint.ingest import (
    MissingStageError,
    RunNotFailedError,
    RunNotFoundError,
    UploadConflictError,
    UploadTooLargeError,
    UploadValidationError,
    UrlValidationError,
    requeue_failed_run,
    submit_media_item_if_new,
    submit_upload,
    submit_url,
)
from voxint.media.redaction import provenance_host
from voxint.media.serving import (
    MediaGate,
    MediaNotServableError,
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
_MEDIA_CHUNK_BYTES = 256 * 1024
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


def _run_or_404(session: Session, run_id: uuid.UUID) -> PipelineRun:
    run = session.get(PipelineRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no run {run_id}")
    return run


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
    return {
        "request": request,
        "run": run,
        "states": states,
        "previews": _label_previews(
            session, run.id, states, settings.review_preview_segments
        ),
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
        # LLM step: the row's enablement over env, its (non-secret) overrides, and
        # whether the env carries a key / a lease-fitting budget — never the key.
        "llm_enabled": bool(row.llm_enabled) if row is not None else settings.llm_enabled,
        "llm_base_url": (row.llm_base_url if row and row.llm_base_url else settings.llm_base_url),
        "llm_model": (row.llm_model if row and row.llm_model else settings.llm_model),
        "llm_key_present": bool(settings.llm_api_key.strip()),
        "llm_budget_ok": llm_budget_fits_stage_lease(settings),
        # The finish step launches the tutorial iff it has been seeded; otherwise
        # it prints the `voxint tutorial seed` note and does a plain /review finish.
        "tutorial_available": ready_tutorial_run_id(session) is not None,
        "active_nav": "setup",
        "error": None,
    }
    context.update(overrides)
    return context


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


def _roster_context(
    request: Request, session: Session, error: str | None = None
) -> dict[str, Any]:
    """Template context for the roster page and its htmx fragment."""
    overview = roster_overview(session)
    secret = request.app.state.csrf_secret
    return {
        "request": request,
        "overview": overview,
        "voiceprints": {
            entry.speaker.id: voiceprint_bars(entry.embeddings)
            for entry in overview.active
        },
        "roster_error": error,
        "active_nav": "speakers",
        "csrf_rename": mint_csrf_token(secret, CSRF_ROSTER_RENAME),
        "csrf_merge": mint_csrf_token(secret, CSRF_ROSTER_MERGE),
        "csrf_archive": mint_csrf_token(secret, CSRF_ROSTER_ARCHIVE),
        "csrf_restore": mint_csrf_token(secret, CSRF_ROSTER_RESTORE),
        "csrf_embedding_delete": mint_csrf_token(
            secret, CSRF_ROSTER_EMBEDDING_DELETE
        ),
    }


def _roster_response(
    request: Request, session: Session, error: str | None = None
) -> Response:
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
            folders = normalize_media_folders(
                media_folders.splitlines(), settings.media_root
            )
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

    def _scan_response(
        request: Request, session: Session, result: ScanResult
    ) -> Response:
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
                    "csrf_setup": mint_csrf_token(
                        request.app.state.csrf_secret, CSRF_SETUP
                    ),
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
                    "csrf_setup": mint_csrf_token(
                        request.app.state.csrf_secret, CSRF_SETUP
                    ),
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
        csrf_token: Annotated[str | None, Form()] = None,
    ) -> Response:
        _require_csrf(request, CSRF_SETUP, csrf_token)
        settings: Settings = request.app.state.settings

        def _rerender(error: str) -> Response:
            return templates.TemplateResponse(
                request,
                "setup.html",
                _setup_context(
                    request,
                    session,
                    WizardStep.LLM,
                    error=error,
                    llm_enabled=enabled,
                    llm_base_url=llm_base_url or settings.llm_base_url,
                    llm_model=llm_model or settings.llm_model,
                ),
            )

        # Format-validate the (optional) overrides first; a malformed value is a
        # form error that changes nothing (a prior valid config stays intact).
        try:
            base_url = normalize_llm_base_url(llm_base_url)
            model = normalize_llm_model(llm_model)
        except SetupValidationError as exc:
            return _rerender(str(exc))
        # Enabling has two hard guards (env key present + budget fits the lease). On
        # failure we FAIL CLOSED: persist llm_enabled=False (plus the parsed
        # overrides) and show why, so an un-enablable LLM can never be left on.
        if enabled:
            try:
                validate_llm_enable(settings)
            except SetupValidationError as exc:
                row = get_or_create(session, llm_enabled_default=settings.llm_enabled)
                row.llm_enabled = False
                row.llm_base_url = base_url
                row.llm_model = model
                return _rerender(str(exc))
        row = get_or_create(session, llm_enabled_default=settings.llm_enabled)
        row.llm_enabled = enabled
        row.llm_base_url = base_url
        row.llm_model = model
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
    ) -> Response:
        settings: Settings = request.app.state.settings
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
        )
        next_url = (
            runs_url(
                status=status_filter,
                review=review_filter,
                filters=search_filters,
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
                "csrf_submit": mint_csrf_token(
                    request.app.state.csrf_secret, CSRF_SUBMIT
                ),
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
        transcript_available = bool(
            session.scalar(
                select(exists().where(TranscriptSegment.pipeline_run_id == run_id))
            )
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
                "transcript_available": transcript_available,
                # Read as a bare boolean (never echoed): a submit/requeue whose
                # enqueue was deferred by a broker outage redirects here with it.
                "enqueue_deferred": request.query_params.get("enqueue") == "deferred",
                # CSRF token for the requeue form (rendered only when FAILED).
                "csrf_requeue": mint_csrf_token(
                    request.app.state.csrf_secret, CSRF_REQUEUE
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
        return templates.TemplateResponse(
            request,
            "transcript.html",
            {
                "request": request,
                "run": run,
                "lines": attributed_transcript(session, run_id, text=variant),
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

    @protected.get("/review")
    def review_queue(
        request: Request, operator: OperatorDep, session: SessionDep
    ) -> Response:
        return templates.TemplateResponse(
            request,
            "queue.html",
            {
                "request": request,
                "entries": adjudication_queue(session),
                "operator": operator,
                # CSRF token for the per-row claim forms.
                "csrf_claim": mint_csrf_token(request.app.state.csrf_secret, CSRF_CLAIM),
                "tutorial": _tutorial_banner(
                    request, session, page=TutorialPage.REVIEW_QUEUE
                ),
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
        _run_or_404(session, run_id)
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
        return RedirectResponse(
            f"/review/{run_id}?token={token}{suffix}", status_code=303
        )

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
                select(Speaker)
                .where(Speaker.id == speaker_id)
                .with_for_update(read=True)
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

    @protected.get("/review/{run_id}/export.txt")
    def export_transcript(
        run_id: uuid.UUID, operator: OperatorDep, session: SessionDep
    ) -> PlainTextResponse:
        _run_or_404(session, run_id)
        lines = attributed_transcript(session, run_id, text=TranscriptText.ENHANCED)
        body = "\n".join(
            f"[{line.start_seconds:9.2f} {line.end_seconds:9.2f}]"
            f" {line.speaker}: {line.text}"
            for line in lines
        )
        return PlainTextResponse(body + ("\n" if lines else ""))

    # ---- Settings + guided-tutorial lifecycle (issue #3, slice 6) --------------
    # The persistent, re-runnable entry point: re-open the setup wizard, and
    # start / replay / complete the guided tutorial. All @protected (an
    # un-onboarded operator is bounced to /setup by the gate). The two POSTs verify
    # CSRF_SETTINGS and 409 when no tutorial run is available, so a stray token can
    # never "complete" or "replay" an unseeded tutorial.

    @protected.get("/settings")
    def settings_page(
        request: Request, operator: OperatorDep, session: SessionDep
    ) -> Response:
        tutorial_run = ready_tutorial_run_id(session)
        row = get_app_settings(session)
        return templates.TemplateResponse(
            request,
            "settings.html",
            {
                "request": request,
                "tutorial_available": tutorial_run is not None,
                "tutorial_run_id": tutorial_run,
                "tutorial_completed_at": row.tutorial_completed_at if row else None,
                # Completion celebration after POST /settings/tutorial/complete —
                # shown ONLY when the tutorial is genuinely completed, so a spoofed
                # or bookmarked ?tutorial=done on an unseeded/incomplete tutorial
                # does not falsely claim completion.
                "tutorial_done": (
                    request.query_params.get("tutorial") == "done"
                    and row is not None
                    and row.tutorial_completed_at is not None
                ),
                "csrf_settings": mint_csrf_token(
                    request.app.state.csrf_secret, CSRF_SETTINGS
                ),
                "active_nav": "settings",
            },
        )

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
    def speakers_page(
        request: Request, operator: OperatorDep, session: SessionDep
    ) -> Response:
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

    @protected.get("/media/{run_id}")
    @protected.head("/media/{run_id}")
    def media(
        run_id: uuid.UUID, request: Request, operator: OperatorDep, session: SessionDep
    ) -> Response:
        settings: Settings = request.app.state.settings
        _run_or_404(session, run_id)
        try:
            path = normalized_audio_path(session, run_id, settings.media_root)
        except StageDataError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        gate = _get_media_gate(request)
        try:
            fh, size = gate.open_for_serving(path)
        except MediaNotServableError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        try:
            byte_range = parse_range(request.headers.get("range"), size)
        except RangeNotSatisfiableError:
            fh.close()
            return Response(
                status_code=416, headers={"Content-Range": f"bytes */{size}"}
            )
        headers = {"Accept-Ranges": "bytes", "Content-Type": "audio/wav"}
        if byte_range is None:
            status, start, length = 200, 0, size
        else:
            status, start, length = 206, byte_range.start, byte_range.length
            headers["Content-Range"] = (
                f"bytes {byte_range.start}-{byte_range.end}/{size}"
            )
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
