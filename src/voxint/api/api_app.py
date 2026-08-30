"""Public REST API sub-application, mounted at ``/api/v1/``.

Separate from the console app so it can carry its own OpenAPI schema, bearer-only
authentication, and guaranteed-JSON exception handlers without leaking console
routes into the public contract or vice versa.
"""

import logging
import secrets
from collections.abc import Iterator
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session as SASession
from starlette.exceptions import HTTPException as StarletteHTTPException

from voxint.api.auth import AuthContext
from voxint.api_keys import TOKEN_PREFIX, verify_api_key
from voxint.config import Settings
from voxint.db.session import build_engine, build_session_factory, session_scope

logger = logging.getLogger(__name__)

API_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Error envelope
# ---------------------------------------------------------------------------

def _error_body(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"code": code, "message": message}
    if details:
        body["details"] = details
    return {"error": body}


async def _api_exception_handler(
    request: Request, exc: HTTPException | StarletteHTTPException
) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    code = getattr(exc, "error_code", None) or _status_to_code(exc.status_code)
    headers = getattr(exc, "headers", None) or {}
    headers["Cache-Control"] = "no-store"
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(code, detail),
        headers=headers,
    )


def _status_to_code(status: int) -> str:
    return {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        405: "method_not_allowed",
        409: "conflict",
        413: "payload_too_large",
        422: "validation_error",
        429: "rate_limited",
        500: "internal_error",
    }.get(status, "error")


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def _get_api_session(request: Request) -> Iterator[SASession]:
    factory = request.app.state.session_factory
    if factory is None:
        factory = build_session_factory(
            build_engine(request.app.state.settings.database_url)
        )
        request.app.state.session_factory = factory
    with session_scope(factory) as session:
        yield session


ApiSessionDep = Annotated[SASession, Depends(_get_api_session)]


def _resolve_bearer(request: Request, session: ApiSessionDep) -> AuthContext:
    settings: Settings = request.app.state.settings

    # Environment key (bootstrap convenience, not DB-backed)
    auth_header = request.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        raise HTTPException(
            status_code=401,
            detail="authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = auth_header[7:].strip()
    if not token:
        raise HTTPException(
            status_code=401,
            detail="authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check env key first
    env_key = settings.voxint_api_key
    if env_key and secrets.compare_digest(token, env_key):
        return AuthContext(user_id=None, username="api:env", role="admin")

    # DB-backed key
    if not token.startswith(TOKEN_PREFIX):
        raise HTTPException(
            status_code=401,
            detail="invalid api key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    result = verify_api_key(session, token, multi_user=settings.voxint_multi_user)
    if result is None:
        raise HTTPException(
            status_code=401,
            detail="invalid api key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    key_row, user = result
    if user is not None:
        return AuthContext(
            user_id=user.id, username=user.username, role=user.role
        )
    return AuthContext(
        user_id=None, username=f"api:{key_row.name}", role="admin"
    )


ApiKeyDep = Annotated[AuthContext, Depends(_resolve_bearer)]


def _require_api_admin(identity: ApiKeyDep) -> AuthContext:
    if identity.role != "admin":
        raise HTTPException(status_code=403, detail="admin role required")
    return identity


ApiAdminDep = Annotated[AuthContext, Depends(_require_api_admin)]


# ---------------------------------------------------------------------------
# Sub-app factory
# ---------------------------------------------------------------------------

def create_api_app() -> FastAPI:
    from voxint import __version__

    api = FastAPI(
        title="Voxint API",
        version=API_VERSION,
        description=(
            "Public REST API for the Voxint audio-intelligence pipeline. "
            f"Application version {__version__}."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    api.add_exception_handler(
        StarletteHTTPException, _api_exception_handler  # type: ignore[arg-type]
    )
    api.add_exception_handler(
        HTTPException, _api_exception_handler  # type: ignore[arg-type]
    )

    _register_api_routes(api)
    return api


def _register_api_routes(api: FastAPI) -> None:
    from voxint.api.routers.api_v1.keys import router as keys_router
    from voxint.api.routers.api_v1.media import router as media_router
    from voxint.api.routers.api_v1.runs import router as runs_router
    from voxint.api.routers.api_v1.status import router as status_router
    from voxint.api.routers.api_v1.transcript import router as transcript_router

    api.include_router(status_router)
    api.include_router(keys_router)
    api.include_router(media_router)
    api.include_router(runs_router)
    api.include_router(transcript_router)
