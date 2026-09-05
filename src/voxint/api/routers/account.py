"""Self-service account management routes for multi-user mode.

Registered on the app directly (not the console aggregator) so that viewer
password changes are not blocked by the console's viewer_write_guard. Each
handler checks voxint_multi_user and returns 404/redirect in single-operator
mode (where no User rows exist to change).
"""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse

from voxint.api.auth import (
    SESSION_COOKIE,
    create_session,
    new_session_token,
)
from voxint.api.csrf import CSRF_ACCOUNT_PASSWORD, mint_csrf_token, verify_csrf_token
from voxint.api.routers.deps import CurrentUserDep, SessionDep, templates
from voxint.config import Settings
from voxint.db.models import User
from voxint.users import change_own_password

logger = logging.getLogger(__name__)

router = APIRouter()


def _password_context(
    request: Request,
    *,
    error: str = "",
    ok: str = "",
) -> dict[str, Any]:
    csrf_secret: str = request.app.state.csrf_secret
    return {
        "request": request,
        "csrf_token": mint_csrf_token(csrf_secret, CSRF_ACCOUNT_PASSWORD),
        "error": error,
        "ok": ok,
        "active_nav": "account",
    }


@router.get("/account/password", response_class=HTMLResponse)
def password_page(request: Request, identity: CurrentUserDep) -> Response:
    settings: Settings = request.app.state.settings
    if not settings.voxint_multi_user:
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(
        request,
        "account/password.html",
        _password_context(
            request,
            ok="Password changed" if request.query_params.get("ok") else "",
        ),
    )


@router.post("/account/password")
def password_submit(
    request: Request,
    session: SessionDep,
    identity: CurrentUserDep,
    current_password: Annotated[str, Form()],
    new_password: Annotated[str, Form()],
    new_password_confirm: Annotated[str, Form()],
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    settings: Settings = request.app.state.settings
    if not settings.voxint_multi_user:
        raise HTTPException(status_code=404, detail="not found")

    csrf_secret: str = request.app.state.csrf_secret
    if not verify_csrf_token(csrf_secret, CSRF_ACCOUNT_PASSWORD, csrf_token):
        return templates.TemplateResponse(
            request,
            "account/password.html",
            _password_context(
                request,
                error="This form is invalid or expired. Please try again.",
            ),
            status_code=403,
        )

    if new_password != new_password_confirm:
        return templates.TemplateResponse(
            request,
            "account/password.html",
            _password_context(request, error="New passwords do not match."),
            status_code=400,
        )

    user = session.get(User, identity.user_id)
    if user is None or user.disabled_at is not None:
        raise HTTPException(status_code=403, detail="account unavailable")

    try:
        changed = change_own_password(
            session,
            user,
            current_password=current_password,
            new_password=new_password,
        )
    except ValueError as exc:
        return templates.TemplateResponse(
            request,
            "account/password.html",
            _password_context(request, error=str(exc)),
            status_code=400,
        )

    if not changed:
        return templates.TemplateResponse(
            request,
            "account/password.html",
            _password_context(request, error="Current password is incorrect."),
            status_code=400,
        )

    token = new_session_token()
    create_session(
        session,
        user_id=user.id,
        token=token,
        ttl_seconds=settings.voxint_session_ttl_seconds,
    )
    session.commit()

    logger.info("user %s changed their password", identity.username)

    response = RedirectResponse(
        "/account/password?ok=1",
        status_code=303,
    )
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
