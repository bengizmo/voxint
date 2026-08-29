"""Login and logout routes for multi-user mode.

Registered on the app (not the console aggregator) so they are exempt from
the onboarding gate, matching the structural exemption pattern of ``/healthz``
and ``setup_router``. When ``voxint_multi_user`` is false, both routes return
404 so the route inventory is stable but the form is unreachable.
"""

from typing import Annotated, Any
from urllib.parse import urlparse

from fastapi import APIRouter, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse

from voxint.api.auth import (
    SESSION_COOKIE,
    cleanup_expired_sessions,
    create_session,
    delete_session,
    load_session_user,
)
from voxint.api.csrf import CSRF_LOGIN, CSRF_LOGOUT, mint_csrf_token, verify_csrf_token
from voxint.api.routers.deps import SessionDep, templates
from voxint.config import Settings
from voxint.users import authenticate

router = APIRouter()


def _login_context(
    request: Request,
    *,
    error: str = "",
    submitted_username: str = "",
    next_url: str = "/",
) -> dict[str, Any]:
    csrf_secret: str = request.app.state.csrf_secret
    return {
        "request": request,
        "csrf_token": mint_csrf_token(csrf_secret, CSRF_LOGIN),
        "error": error,
        "submitted_username": submitted_username,
        "next_url": next_url,
    }


def _validate_next(next_url: str | None) -> str:
    if not next_url:
        return "/"
    if not next_url.startswith("/"):
        return "/"
    if next_url.startswith("//") or next_url.startswith("/\\"):
        return "/"
    if "\\" in next_url or any(ord(c) < 0x20 for c in next_url):
        return "/"
    parsed = urlparse(next_url)
    if parsed.scheme or parsed.netloc:
        return "/"
    return next_url


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request) -> Response:
    settings: Settings = request.app.state.settings
    if not settings.voxint_multi_user:
        raise HTTPException(status_code=404, detail="not found")
    next_url = _validate_next(request.query_params.get("next"))
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        session_factory = request.app.state.session_factory
        if session_factory is not None:
            from voxint.db.session import session_scope

            with session_scope(session_factory) as db:
                if load_session_user(db, token) is not None:
                    return RedirectResponse(url=next_url, status_code=303)
    return templates.TemplateResponse(
        request, "auth/login.html", _login_context(request, next_url=next_url)
    )


@router.post("/login")
def login_submit(
    request: Request,
    session: SessionDep,
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
    csrf_token: Annotated[str, Form()],
    next: Annotated[str, Form(alias="next")] = "/",
) -> Response:
    settings: Settings = request.app.state.settings
    if not settings.voxint_multi_user:
        raise HTTPException(status_code=404, detail="not found")

    csrf_secret: str = request.app.state.csrf_secret
    if not verify_csrf_token(csrf_secret, CSRF_LOGIN, csrf_token):
        return templates.TemplateResponse(
            request,
            "auth/login.html",
            _login_context(
                request,
                error="Session expired. Please try again.",
                submitted_username=username,
                next_url=_validate_next(next),
            ),
            status_code=403,
        )

    user = authenticate(session, username=username.strip().lower(), password=password)
    if user is None:
        return templates.TemplateResponse(
            request,
            "auth/login.html",
            _login_context(
                request,
                error="Invalid credentials.",
                submitted_username=username,
                next_url=_validate_next(next),
            ),
            status_code=401,
        )

    cleanup_expired_sessions(session)

    from voxint.api.auth import new_session_token

    token = new_session_token()
    create_session(
        session,
        user_id=user.id,
        token=token,
        ttl_seconds=settings.voxint_session_ttl_seconds,
    )

    next_url = _validate_next(next)
    response = RedirectResponse(url=next_url, status_code=303)
    secure = request.url.scheme == "https" or request.headers.get(
        "x-forwarded-proto"
    ) == "https"
    response.set_cookie(
        SESSION_COOKIE,
        value=token,
        httponly=True,
        samesite="lax",
        path="/",
        secure=secure,
        max_age=settings.voxint_session_ttl_seconds,
    )
    return response


@router.post("/logout")
def logout(
    request: Request,
    session: SessionDep,
    csrf_token: Annotated[str, Form()],
) -> Response:
    settings: Settings = request.app.state.settings
    if not settings.voxint_multi_user:
        raise HTTPException(status_code=404, detail="not found")

    csrf_secret: str = request.app.state.csrf_secret
    if not verify_csrf_token(csrf_secret, CSRF_LOGOUT, csrf_token):
        raise HTTPException(status_code=403, detail="invalid CSRF token")

    token = request.cookies.get(SESSION_COOKIE)
    if token:
        delete_session(session, token)

    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response
