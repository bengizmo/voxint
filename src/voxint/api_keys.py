"""API key service: create, verify, revoke, list.

Keys authenticate against ``/api/v1/`` via Bearer tokens.  The plaintext
token is returned exactly once on creation and never stored; the database
holds only the SHA-256 digest (same primitive as session tokens -- high-entropy
random material, so argon2 would add latency for no security gain).
"""

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.db.models import ApiKey, User

TOKEN_PREFIX = "vxint_"
_PREFIX_DISPLAY_LEN = 12
_LAST_USED_THROTTLE_SECONDS = 60


@dataclass(frozen=True, slots=True)
class CreatedKey:
    id: uuid.UUID
    name: str
    key_prefix: str
    plaintext_token: str


def _hash_token(token: str) -> bytes:
    return hashlib.sha256(token.encode("ascii")).digest()


def create_api_key(
    db: Session,
    *,
    name: str,
    user_id: uuid.UUID | None = None,
) -> CreatedKey:
    name = name.strip()
    if not name:
        raise ValueError("key name must not be blank")

    raw = secrets.token_urlsafe(32)
    plaintext = f"{TOKEN_PREFIX}{raw}"
    prefix = raw[:_PREFIX_DISPLAY_LEN]

    row = ApiKey(
        id=uuid.uuid4(),
        user_id=user_id,
        name=name,
        key_hash=_hash_token(plaintext),
        key_prefix=prefix,
    )
    db.add(row)
    db.flush()
    return CreatedKey(
        id=row.id,
        name=row.name,
        key_prefix=prefix,
        plaintext_token=plaintext,
    )


def verify_api_key(
    db: Session,
    token: str,
    *,
    multi_user: bool,
) -> tuple[ApiKey, User | None] | None:
    """Verify a bearer token and return the key row (+ owning user if any).

    Returns ``None`` on any auth failure (invalid, expired, revoked, disabled
    user, wrong mode).  Throttle-updates ``last_used_at`` to avoid WAL churn
    on hot poll paths.
    """
    if not token.startswith(TOKEN_PREFIX):
        return None

    token_hash = _hash_token(token)
    row = db.execute(
        select(ApiKey).where(ApiKey.key_hash == token_hash)
    ).scalar_one_or_none()
    if row is None:
        return None
    if row.revoked_at is not None:
        return None

    now = datetime.now(UTC)
    if row.expires_at is not None and row.expires_at <= now:
        return None

    user: User | None = None
    if multi_user:
        if row.user_id is None:
            pass
        else:
            user = db.get(User, row.user_id)
            if user is None or user.disabled_at is not None:
                return None
    else:
        if row.user_id is not None:
            return None

    if (
        row.last_used_at is None
        or (now - row.last_used_at).total_seconds() > _LAST_USED_THROTTLE_SECONDS
    ):
        row.last_used_at = now

    return row, user


def revoke_api_key(db: Session, key_id: uuid.UUID) -> ApiKey | None:
    row = db.get(ApiKey, key_id)
    if row is None:
        return None
    if row.revoked_at is None:
        row.revoked_at = datetime.now(UTC)
    return row


def list_api_keys(
    db: Session,
    *,
    user_id: uuid.UUID | None = None,
    include_revoked: bool = False,
) -> list[ApiKey]:
    q = select(ApiKey).order_by(ApiKey.created_at.desc())
    if user_id is not None:
        q = q.where(ApiKey.user_id == user_id)
    if not include_revoked:
        q = q.where(ApiKey.revoked_at.is_(None))
    return list(db.scalars(q))
