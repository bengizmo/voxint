"""Authentication for the review UI: single-operator Basic or multi-user sessions.

Single-operator mode (default): HTTP Basic against env credentials, exactly as
before. Multi-user mode (``VOXINT_MULTI_USER=true``): DB-backed opaque session
cookies with a login form. The mode branch lives in ``_resolve_identity()`` in
``deps.py``; this module provides the pure helpers both paths share.
"""

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from voxint.config import Settings
from voxint.db.models import AuthSession, User

SESSION_COOKIE = "voxint_session"


@dataclass(frozen=True, slots=True)
class AuthContext:
    user_id: uuid.UUID | None
    username: str
    role: str


def verify_basic_credentials(settings: Settings, username: str, password: str) -> bool:
    user_ok = secrets.compare_digest(
        username.encode(), settings.voxint_user.encode()
    )
    password_ok = secrets.compare_digest(
        password.encode(), settings.voxint_password.encode()
    )
    return user_ok and password_ok


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def hash_session_token(token: str) -> bytes:
    return hashlib.sha256(token.encode("ascii")).digest()


def create_session(
    db: Session,
    *,
    user_id: uuid.UUID,
    token: str,
    ttl_seconds: int,
) -> AuthSession:
    now = datetime.now(UTC)
    row = AuthSession(
        id=uuid.uuid4(),
        user_id=user_id,
        token_hash=hash_session_token(token),
        created_at=now,
        expires_at=now + timedelta(seconds=ttl_seconds),
    )
    db.add(row)
    return row


def load_session_user(
    db: Session,
    token: str,
) -> User | None:
    token_hash = hash_session_token(token)
    now = datetime.now(UTC)
    row = db.execute(
        select(AuthSession)
        .where(
            AuthSession.token_hash == token_hash,
            AuthSession.expires_at > now,
        )
    ).scalar_one_or_none()
    if row is None:
        return None
    user = db.get(User, row.user_id)
    if user is None or user.disabled_at is not None:
        return None
    return user


def delete_session(db: Session, token: str) -> None:
    db.execute(
        delete(AuthSession).where(
            AuthSession.token_hash == hash_session_token(token)
        )
    )


def cleanup_expired_sessions(db: Session) -> int:
    now = datetime.now(UTC)
    result = db.execute(
        delete(AuthSession).where(AuthSession.expires_at <= now)
    )
    return result.rowcount  # type: ignore[return-value]
