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
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.responses import (
    FileResponse,
    Response,
)
from starlette.datastructures import MutableHeaders
from starlette.requests import ClientDisconnect
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from voxint import __version__
from voxint.api.routers.deps import (
    _APP_ASSET_MEDIA_TYPES,
    _APP_ASSETS_DIR,
    OperatorDep,
    _looks_hashed,
)
from voxint.api.routers.legacy_review import router as review_router
from voxint.api.routers.legacy_review import transcript_router as review_transcript_router
from voxint.api.routers.legacy_runs import (
    actions_router as runs_actions_router,
)
from voxint.api.routers.legacy_runs import (
    core_router as runs_core_router,
)
from voxint.api.routers.legacy_runs import (
    dashboards_router as runs_dashboards_router,
)
from voxint.api.routers.legacy_runs import (
    tail_router as runs_tail_router,
)
from voxint.api.routers.settings import router as settings_router
from voxint.api.routers.settings import setup_router
from voxint.api.routers.speakers import router as speakers_router
from voxint.config import Settings, get_settings

logger = logging.getLogger(__name__)

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


def _register_routes(app: FastAPI) -> None:
    # Two registrars: `app` carries the onboarding-gate-EXEMPT routes (liveness,
    # the static asset routes, and the setup wizard's setup_router); `console`
    # aggregates every per-area router, and each of those declares its own
    # router-level require_onboarded dependency (routers/deps.py). The gate
    # rides the family router, not this aggregator, because an outer router's
    # dependencies do not appear in a nested route's dependant tree on this
    # FastAPI, which is where the characterization contract reads gating from.
    # Exemption stays structural: a route is exempt iff it reaches the app
    # outside a gated family router (the route-inventory and onboarding-gate
    # tests guard against a slip). New console routes go on their area router.
    console = APIRouter()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    # ---- First-run setup wizard (issue #3): moved to routers/settings.py
    # (setup_router, registered on `app` so the onboarding gate exempts it).
    app.include_router(setup_router)

    # ---- Index + run submission/browsing/transcript: moved to
    # routers/legacy_runs.py; included here to keep registration order.
    console.include_router(runs_core_router)

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

    console.include_router(review_transcript_router)

    # ---- Run actions (requeue/cancel/archive/notes/export): moved to
    # routers/legacy_runs.py; included here to keep registration order.
    console.include_router(runs_actions_router)

    # ---- Review queue, workbench, annotations (issues #16/#86): moved to
    # routers/legacy_review.py; included here to keep registration order.
    console.include_router(review_router)

    # ---- Metrics, dashboard, resources: moved to routers/legacy_runs.py;
    # included here to keep registration order.
    console.include_router(runs_dashboards_router)

    # ---- Settings + guided-tutorial lifecycle: moved to routers/settings.py;
    # included here to keep registration order.
    console.include_router(settings_router)

    # ---- Speaker roster + web research (issue #7 / #42): moved to
    # routers/speakers.py; included here to keep registration order.
    console.include_router(speakers_router)

    # ---- Run assets, translation, media streaming: moved to
    # routers/legacy_runs.py; included here to keep registration order.
    console.include_router(runs_tail_router)

    app.include_router(console)


app = create_app()
