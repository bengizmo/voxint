"""Single-operator HTTP Basic auth for the review UI.

Identity comes exclusively from verified credentials — never from a header or
form field a client could set. Every route except ``/healthz`` (including
htmx fragments and media) hangs off this dependency. Comparison is
constant-time on both fields regardless of which one mismatches.
"""

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from voxint.config import Settings

_basic = HTTPBasic(realm="voxint-review")


def require_operator(
    request: Request,
    credentials: Annotated[HTTPBasicCredentials, Depends(_basic)],
) -> str:
    """Authenticated operator name, or a 401 challenge."""
    settings: Settings = request.app.state.settings
    user_ok = secrets.compare_digest(
        credentials.username.encode(), settings.voxint_user.encode()
    )
    password_ok = secrets.compare_digest(
        credentials.password.encode(), settings.voxint_password.encode()
    )
    if not (user_ok and password_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid credentials",
            headers={"WWW-Authenticate": 'Basic realm="voxint-review"'},
        )
    return credentials.username
