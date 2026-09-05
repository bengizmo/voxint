import re
import uuid
from datetime import UTC, datetime

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from voxint.db.models import AuthSession, User, UserRole

_PASSWORD_HASHER = PasswordHasher()
_USERNAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_MAX_PASSWORD_BYTES = 1024
_DUMMY_HASH: str = (
    "$argon2id$v=19$m=65536,t=3,p=4$Gwsswm7h9t/4Z/Y9USuYpA$"
    "JTWaLrqL7JFNDA6yngvUok1GasS+Y+W4sAa3Qe3Ee/s"
)


def _validate_password(password: str) -> None:
    if not password:
        raise ValueError("password must not be empty")
    if len(password.encode("utf-8")) > _MAX_PASSWORD_BYTES:
        raise ValueError("password must not exceed 1024 UTF-8 bytes")


def hash_password(password: str) -> str:
    _validate_password(password)
    return _PASSWORD_HASHER.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        _validate_password(password)
    except ValueError:
        return False
    try:
        return _PASSWORD_HASHER.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def create_user(
    session: Session,
    *,
    username: str,
    password: str,
    role: UserRole = UserRole.REVIEWER,
) -> User:
    if ":" in username or _USERNAME_PATTERN.fullmatch(username) is None:
        raise ValueError("invalid username")
    _validate_password(password)

    if session.scalar(select(func.count()).select_from(User)) == 0:
        role = UserRole.ADMIN

    user = User(
        id=uuid.uuid4(),
        username=username,
        password_hash=hash_password(password),
        role=role.value,
    )
    session.add(user)
    session.flush()
    return user


def authenticate(
    session: Session, *, username: str, password: str
) -> User | None:
    user = session.scalar(select(User).where(User.username == username))
    password_hash = user.password_hash if user is not None else _DUMMY_HASH
    if not verify_password(password, password_hash):
        return None
    if user is None or user.disabled_at is not None:
        return None
    return user


def list_users(session: Session) -> list[User]:
    return list(session.scalars(select(User).order_by(User.username)).all())


def _assert_not_last_admin(session: Session, user: User, action: str) -> None:
    """Raise if ``user`` is the only active admin.

    Locks all active admin rows with ``FOR UPDATE`` so two concurrent
    demotions/disables cannot both pass the check and leave zero active
    admins (TOCTOU). The lock is held until the caller's transaction commits
    or rolls back. SQLite (unit tests) ignores ``with_for_update`` silently,
    so this is safe in both engines.
    """
    if user.role != UserRole.ADMIN.value or user.disabled_at is not None:
        return
    active_admins = session.scalar(
        select(func.count())
        .select_from(
            select(User.id)
            .where(
                User.role == UserRole.ADMIN.value,
                User.disabled_at.is_(None),
            )
            .with_for_update()
            .subquery()
        )
    )
    if active_admins == 1:
        raise ValueError(f"cannot {action} the last active admin")


def set_role(session: Session, user: User, new_role: UserRole) -> None:
    if new_role is not UserRole.ADMIN:
        _assert_not_last_admin(session, user, "downgrade")
    if user.role != new_role.value:
        user.role = new_role.value
        session.execute(delete(AuthSession).where(AuthSession.user_id == user.id))


def set_disabled(session: Session, user: User, *, disabled: bool) -> None:
    if disabled:
        _assert_not_last_admin(session, user, "disable")
        user.disabled_at = datetime.now(UTC)
        session.execute(delete(AuthSession).where(AuthSession.user_id == user.id))
    else:
        user.disabled_at = None


def reset_password(session: Session, user: User, new_password: str) -> None:
    user.password_hash = hash_password(new_password)
    session.execute(delete(AuthSession).where(AuthSession.user_id == user.id))


def change_own_password(
    session: Session,
    user: User,
    *,
    current_password: str,
    new_password: str,
) -> bool:
    if not verify_password(current_password, user.password_hash):
        return False
    reset_password(session, user, new_password)
    return True
