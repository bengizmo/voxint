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
from collections.abc import AsyncIterator, Iterable, Iterator
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    Response,
)
from fastapi.routing import APIRoute
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import ClientDisconnect
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from voxint import __version__
from voxint.api.routers.activity import router as activity_router
from voxint.api.routers.deps import (
    _APP_ASSET_MEDIA_TYPES,
    _APP_ASSETS_DIR,
    OperatorDep,
    _configure_template_loader,
    _get_session,
    _looks_hashed,
    _plugin_template_dirs,
    _verify_plugin_csrf,
    require_onboarded,
    templates,
)
from voxint.api.routers.editor import router as editor_router
from voxint.api.routers.home import router as home_router
from voxint.api.routers.jobs import router as jobs_router
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
from voxint.api.routers.media import router as media_router
from voxint.api.routers.projects import router as projects_router
from voxint.api.routers.settings import _settings_context, setup_router
from voxint.api.routers.settings import router as settings_router
from voxint.api.routers.speakers import router as speakers_router
from voxint.config import Settings, get_settings
from voxint.plugins import PluginError, PluginRegistry, load_plugins
from voxint.plugins.boot import validate_boot
from voxint.plugins.deps import PluginRouteDeps
from voxint.plugins.registry import find_route_collisions

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

        token_path = _is_token_sensitive_path(scope.get("path", ""))

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                _apply_security_headers(
                    MutableHeaders(scope=message), token_path=token_path
                )
            await send(message)

        await self._app(scope, receive, send_with_headers)


_MEDIA_DETAIL_RE = re.compile(
    r"^/media/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/editor(?:\?|$)",
    re.IGNORECASE,
)


def _is_token_sensitive_path(path: str) -> bool:
    """True for route families that may carry a claim token in the URL or response.

    Matched paths get ``Cache-Control: no-store`` so token-bearing responses are
    never written to a browser or proxy cache. Currently two families:

    * ``/review`` and all descendants (the legacy review flow).
    * ``/media/{uuid}`` and descendants (the editor detail page, #156).

    Library-level ``/media`` routes (listing, upload, assign, rerun, archive) are
    excluded -- they never carry tokens and should remain cacheable by the browser.
    """
    if path.startswith("/review"):
        return True
    return bool(_MEDIA_DETAIL_RE.match(path))


def _apply_security_headers(headers: MutableHeaders, *, token_path: bool) -> None:
    """Stamp the console's baseline security headers (idempotent via ``setdefault``).

    Shared by ``_SecurityHeadersMiddleware`` and the 500 handler so the policy has
    one definition.

    * ``Referrer-Policy: no-referrer`` and ``Cache-Control: no-store`` (on
      token-sensitive paths) contain the URL-borne claim token (finding D1).
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
    if token_path:
        headers.setdefault("cache-control", "no-store")


def _wants_html(request: Request) -> bool:
    """Return whether the request's Accept header includes HTML.

    Intentionally simple: ``*/*`` and missing Accept fall through to JSON,
    which is what scripted clients and curl expect."""
    return "text/html" in request.headers.get("accept", "").lower()


def _error_page_values(status_code: int, detail: Any) -> tuple[str, str]:
    """Return client-safe title and detail text for an HTML error page."""
    if status_code >= 500:
        return "Something went wrong", "Internal Server Error"
    if status_code == 404:
        return (
            "Page not found",
            "The page you were looking for does not exist or has been moved.",
        )
    if status_code == 403:
        return "Forbidden", str(detail)
    return "Request error", str(detail)


def _html_error_response(request: Request, status_code: int, detail: Any) -> Response:
    """Render the standalone error template, with a resilient text fallback."""
    title, safe_detail = _error_page_values(status_code, detail)
    try:
        response: HTMLResponse = templates.TemplateResponse(
            request,
            "error.html",
            {
                "status_code": status_code,
                "title": title,
                "detail": safe_detail,
            },
            status_code=status_code,
        )
        return response
    except Exception:
        logger.error("failed to render HTML error page", exc_info=True)
        return Response(safe_detail, status_code=status_code, media_type="text/plain")


async def _http_exception_handler(request: Request, exc: HTTPException) -> Response:
    """Render HTTP errors as HTML for browsers and JSON for API clients.

    Non-error status codes (< 400, e.g. 204 with hx-redirect) are returned
    with the original detail and no content-negotiation so the default
    Starlette behavior is preserved."""
    if exc.status_code < 400:
        response = Response(
            status_code=exc.status_code,
            headers=exc.headers,
        )
        _apply_security_headers(
            response.headers, token_path=_is_token_sensitive_path(request.url.path)
        )
        return response
    detail: Any = "Internal Server Error" if exc.status_code >= 500 else exc.detail
    if _wants_html(request):
        response = _html_error_response(request, exc.status_code, detail)
    else:
        response = Response(
            json.dumps({"detail": detail}),
            status_code=exc.status_code,
            media_type="application/json",
        )
    if exc.headers and exc.status_code < 500:
        response.headers.update(exc.headers)
    _apply_security_headers(
        response.headers, token_path=_is_token_sensitive_path(request.url.path)
    )
    return response


async def _security_headers_on_error(request: Request, exc: Exception) -> Response:
    """Content-negotiate an unhandled 500 and re-apply the D1 headers.

    Starlette's ``ServerErrorMiddleware`` wraps the whole app *outside* the
    user-added ``_SecurityHeadersMiddleware``, so a truly-unhandled exception's
    500 would otherwise skip the header stamp. Registering this as the ``Exception``
    handler closes that gap. ``ServerErrorMiddleware`` still re-raises after sending
    this response, so it does not mask exceptions from the server or tests."""
    if _wants_html(request):
        response = _html_error_response(request, 500, "Internal Server Error")
    else:
        response = Response(
            json.dumps({"detail": "Internal Server Error"}),
            status_code=500,
            media_type="application/json",
        )
    _apply_security_headers(
        response.headers, token_path=_is_token_sensitive_path(request.url.path)
    )
    return response


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run one-shot startup tasks that need an initialized app state."""
    try:
        from voxint.db.session import build_engine, build_session_factory, session_scope
        from voxint.ingest.service import reconcile_orphaned_incoming

        resolved: Settings = app.state.settings
        factory = app.state.session_factory
        if factory is None:
            factory = build_session_factory(build_engine(resolved.database_url))
            app.state.session_factory = factory
        with session_scope(factory) as session:
            removed = reconcile_orphaned_incoming(session, resolved.media_root)
            if removed:
                logger.info(
                    "reconciled %d orphaned incoming file(s)", len(removed)
                )
    except Exception:
        logger.warning(
            "startup reconciler skipped (database unavailable)", exc_info=True
        )
    yield


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
        lifespan=_lifespan,
    )
    resolved = settings or get_settings()
    app.state.settings = resolved
    # Plugin framework (issue #138): build + validate the registry once at app
    # startup and store it on app.state for the route / settings / template seams.
    # Boot fails loud on a malformed builtin or an env-sourced plugin invariant
    # violation (validate_boot), matching the worker's import-time fail-loud. The
    # template loader is (re)configured from it so plugin templates render
    # namespaced as <plugin_id>/foo.html. Dormant while BUILTIN is empty: no
    # plugins, no contributions, and the loader stays the pristine core loader.
    registry = load_plugins(resolved)
    validate_boot(registry, settings=resolved)
    app.state.plugins = registry
    _configure_template_loader(_plugin_template_dirs(registry))
    # CSRF signing secret: the configured value when set, otherwise a secret
    # auto-generated and persisted to the data directory on first run (so open
    # forms survive a restart and multiple workers share the same secret).
    if resolved.csrf_secret:
        app.state.csrf_secret = resolved.csrf_secret
    else:
        from voxint.api.csrf import load_or_create_csrf_secret

        app.state.csrf_secret = load_or_create_csrf_secret(resolved.media_root)
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
    # Starlette raises its base HTTPException for router-generated errors such
    # as a nonexistent path; FastAPI's HTTPException is a subclass used by routes.
    app.add_exception_handler(
        StarletteHTTPException, _http_exception_handler  # type: ignore[arg-type]
    )
    app.add_exception_handler(HTTPException, _http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _security_headers_on_error)
    _register_routes(app)
    return app


def _iter_api_routes(routes: Iterable[object]) -> Iterator[APIRoute]:
    """Yield every :class:`APIRoute` reachable from ``routes``, descending mounts.

    FastAPI mounts an included ``APIRouter`` as a sub-route that exposes the
    original router on ``original_router`` rather than flattening its routes, so
    each per-area router under the ``console`` aggregator — and each plugin
    router included alongside them (#138) — sits nested. A single-level walk
    misses those routes entirely, which is why the plugin-vs-core collision
    guard and the route-inventory contract test both recurse here. Non-APIRoute
    mounts without an ``original_router`` (static-file mounts) yield nothing.
    """
    for route in routes:
        if isinstance(route, APIRoute):
            yield route
        else:
            sub = getattr(route, "original_router", None)
            if sub is not None:
                yield from _iter_api_routes(sub.routes)


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

    # ---- Home (Console 2.0 P1, #152): the landing page at /. Registered
    # first among the console families so the root route sits early in the
    # match/inventory order, where the old index redirect lived.
    console.include_router(home_router)

    # ---- Media library (Console 2.0 P2a, #153): the /media file listing.
    # Always registered so the route inventory is stable across the dark-ship
    # flip; the router's require_media_enabled gate 404s until the flag is on.
    console.include_router(media_router)

    # ---- Media detail / editor (Console 2.0 P3a, #156): the /media/{id} page.
    # Registered immediately after media_router so the UUID path parameter does
    # not shadow the library's named action routes (/media/submit etc.). Same
    # area gate (require_media_enabled) on the router.
    console.include_router(editor_router)

    # ---- Projects (Console 2.0 P2b, #153): the /projects list + detail pages.
    # Always registered (stable route inventory); require_projects_enabled 404s
    # them until the flag is on. Registering /projects is what flips
    # app.state.projects_routed, so the sidebar's Projects link appears only once
    # these pages exist AND console_projects_enabled is set.
    console.include_router(projects_router)

    # ---- Run submission/browsing/transcript: moved to
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

    # ---- Jobs area (Console 2.0 P5, #160): /jobs + /jobs/{run_id}. Slotted
    # after speakers and before the legacy runs tail so the new pages register
    # ahead of the /runs child endpoints they will eventually absorb, keeping the
    # legacy surface grouped last. Dark-shipped: registered unconditionally (no
    # area gate), discovery gated by console_jobs_enabled in the sidebar.
    console.include_router(jobs_router)

    # ---- Activity poll endpoint (Console 2.0 P7, #162): /activity/events.
    # Registered next to Jobs (its badge + toast links target /jobs). Dark-ship:
    # always registered so the route inventory is stable; the handler 404s until
    # console_activity_enabled is on.
    console.include_router(activity_router)

    # ---- Run assets, translation, media streaming: moved to
    # routers/legacy_runs.py; included here to keep registration order.
    console.include_router(runs_tail_router)

    # Plugin routers (issue #138): build each active plugin's router with the
    # capped deps bundle, reject any (path, method) claimed by two plugins BEFORE
    # mounting (the route-inventory contract test additionally catches a plugin
    # colliding with a CORE route), then mount each behind a gated wrapper so
    # plugin routes run behind require_onboarded like every console route. The
    # wrapper is needed because the `console` aggregator is ungated (P0b: the
    # gate rides each per-area router); on main's pre-P0b layout the gated
    # `protected` router provided this at runtime. URLs are frozen contract, so
    # no prefix is forced. Empty registry => nothing built, nothing mounted,
    # byte-identical route table.
    registry: PluginRegistry = app.state.plugins
    plugin_deps = PluginRouteDeps(
        templates=templates,
        get_session=_get_session,
        verify_csrf=_verify_plugin_csrf,
        render_settings_page=lambda request, session: templates.TemplateResponse(
            request, "settings/settings.html", _settings_context(request, session)
        ),
    )
    built_routers: list[APIRouter] = []
    routes_by_plugin: dict[str, list[tuple[str, str]]] = {}
    for plugin in registry.plugins:
        router = plugin.build_router(plugin_deps)
        if router is None:
            continue
        built_routers.append(router)
        routes_by_plugin[plugin.manifest.id] = [
            (getattr(route, "path", ""), method)
            for route in router.routes
            for method in (getattr(route, "methods", None) or ())
        ]
    collisions = find_route_collisions(routes_by_plugin)
    if collisions:
        raise PluginError("plugin route collisions: " + "; ".join(collisions))
    # Plugin-vs-core collisions (issue #138): a plugin claiming a (path, method)
    # a core route already owns fails boot too, mirroring the worker's
    # _CORE_TASK_NAMES guard. find_route_collisions above only catches
    # plugin-vs-plugin; without this a converted plugin (#139+) that squats a core
    # route would mount an unreachable shadow that route dispatch order hides.
    # Core routes are the exempt app-level routes plus every route already on
    # `console` (mounted below, so not yet on `app`).
    core_pairs = {
        (route.path, method)
        for route in (
            *_iter_api_routes(app.routes),
            *_iter_api_routes(console.routes),
        )
        for method in (route.methods or ())
    }
    core_conflicts = sorted(
        f"{method.upper()} {path}: claimed by plugin {plugin_id} and a core route"
        for plugin_id, routes in routes_by_plugin.items()
        for path, method in routes
        if (path, method.upper()) in core_pairs
    )
    if core_conflicts:
        raise PluginError(
            "plugin route collisions with core: " + "; ".join(core_conflicts)
        )
    if built_routers:
        gated_plugins = APIRouter(dependencies=[Depends(require_onboarded)])
        for router in built_routers:
            gated_plugins.include_router(router)
        console.include_router(gated_plugins)

    app.include_router(console)

    # Console area-flag guard (#152 review): the sidebar and Home render a
    # dark-shipped area's links only when its flag is on AND its routes exist,
    # so flipping CONSOLE_PROJECTS_ENABLED before the projects phase lands can
    # never advertise a guaranteed 404. Computed once here (the route table is
    # fixed after startup); the shell context processor reads it per request.
    app.state.projects_routed = any(
        route.path == "/projects" for route in _iter_api_routes(app.routes)
    )
    # Jobs (#160) dark-ships routed-but-undiscovered: /jobs always registers, so
    # this stamp is always true. The shell reads flag AND stamp (mirroring
    # projects), so flipping CONSOLE_JOBS_ENABLED alone surfaces the sidebar Jobs
    # link — the dark-ship activation switch, not a route-existence guard.
    app.state.jobs_routed = any(
        route.path == "/jobs" for route in _iter_api_routes(app.routes)
    )
    # Media (#154) dark-ships routed-but-undiscovered: /media always registers, so
    # this stamp is always true. The shell reads flag AND stamp (mirroring jobs), so
    # flipping CONSOLE_MEDIA_ENABLED alone points the sidebar Media link and the
    # "Add media" quick action at /media — the dark-ship activation switch.
    app.state.media_routed = any(
        route.path == "/media" for route in _iter_api_routes(app.routes)
    )
    # Activity (#162) dark-ships routed-but-undiscovered: /activity/events always
    # registers, so this stamp is always true. shell.activity_enabled ANDs the
    # flag, this stamp, AND shell.jobs_enabled (the badge lives on the Jobs entry
    # and the toast links target /jobs), so activity is never surfaced while Jobs
    # discovery is off.
    app.state.activity_routed = any(
        route.path == "/activity/events" for route in _iter_api_routes(app.routes)
    )


app = create_app()
