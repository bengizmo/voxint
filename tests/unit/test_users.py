import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import CheckConstraint, create_engine, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from voxint.db.models import AuthSession, Base, User
from voxint.users import (
    UserRole,
    authenticate,
    create_user,
    hash_password,
    list_users,
    reset_password,
    set_disabled,
    set_role,
    verify_password,
)


@pytest.fixture()
def session() -> Iterator[Session]:
    engine = create_engine("sqlite://")
    postgres_only_checks = [
        constraint
        for constraint in User.__table__.constraints
        if isinstance(constraint, CheckConstraint)
        and constraint.name
        in {
            "users_username_nonempty_check",
            "users_username_format_check",
            "users_username_no_colon_check",
        }
    ]
    # SQLite cannot compile the PostgreSQL operators in these model constraints.
    for constraint in postgres_only_checks:
        User.__table__.constraints.remove(constraint)
    try:
        Base.metadata.create_all(
            engine, tables=[User.__table__, AuthSession.__table__]
        )
    finally:
        for constraint in postgres_only_checks:
            User.__table__.append_constraint(constraint)
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as db_session:
        yield db_session
    engine.dispose()


def test_hash_password_round_trip() -> None:
    password_hash = hash_password("secret")

    assert verify_password("secret", password_hash)
    assert password_hash != "secret"


def test_verify_password_rejects_wrong_password() -> None:
    assert not verify_password("wrong", hash_password("secret"))


def test_hash_password_rejects_oversized_password() -> None:
    with pytest.raises(ValueError):
        hash_password("a" * 1025)


def test_verify_password_rejects_oversized_password() -> None:
    assert not verify_password("a" * 1025, hash_password("secret"))


def test_verify_password_rejects_tampered_hash() -> None:
    password_hash = hash_password("secret")

    assert not verify_password("secret", password_hash[:-1] + "x")


def test_empty_password_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        hash_password("")


def test_create_user_hashes_password(session: Session) -> None:
    user = create_user(session, username="reviewer", password="secret")

    assert user.username == "reviewer"
    assert user.password_hash != "secret"
    assert verify_password("secret", user.password_hash)


def test_create_user_forces_first_user_to_admin(session: Session) -> None:
    user = create_user(
        session,
        username="first",
        password="secret",
        role=UserRole.REVIEWER,
    )

    assert user.role == UserRole.ADMIN.value


def test_create_user_respects_role_for_second_user(session: Session) -> None:
    create_user(session, username="first", password="secret")

    user = create_user(
        session,
        username="second",
        password="secret",
        role=UserRole.REVIEWER,
    )

    assert user.role == UserRole.REVIEWER.value


@pytest.mark.parametrize(
    "username",
    ["Uppercase", "has:colon", "", "a" * 65, ".leading", "_leading"],
)
def test_create_user_rejects_invalid_username(session: Session, username: str) -> None:
    with pytest.raises(ValueError):
        create_user(session, username=username, password="secret")


@pytest.mark.parametrize(
    "username",
    ["lowercase", "user123", "user.name", "user-name", "user_name"],
)
def test_create_user_accepts_valid_username(session: Session, username: str) -> None:
    assert create_user(session, username=username, password="secret").username == username


def test_create_user_rejects_oversized_password(session: Session) -> None:
    with pytest.raises(ValueError):
        create_user(session, username="reviewer", password="a" * 1025)


def test_create_user_rejects_duplicate_username(session: Session) -> None:
    create_user(session, username="duplicate", password="secret")

    with pytest.raises(IntegrityError):
        create_user(session, username="duplicate", password="secret")


def test_authenticate_returns_user_for_correct_credentials(
    session: Session,
) -> None:
    user = create_user(session, username="reviewer", password="secret")

    assert authenticate(session, username="reviewer", password="secret") is user


def test_authenticate_rejects_wrong_password(session: Session) -> None:
    create_user(session, username="reviewer", password="secret")

    assert authenticate(session, username="reviewer", password="wrong") is None


def test_authenticate_rejects_unknown_username(session: Session) -> None:
    assert authenticate(session, username="unknown", password="secret") is None


def test_authenticate_rejects_disabled_user(session: Session) -> None:
    user = create_user(session, username="reviewer", password="secret")
    user.disabled_at = datetime.now(UTC)
    session.flush()

    assert authenticate(session, username="reviewer", password="secret") is None


def test_list_users_empty(session: Session) -> None:
    assert list_users(session) == []


def test_list_users_ordered_by_username(session: Session) -> None:
    create_user(session, username="charlie", password="secret")
    create_user(session, username="alice", password="secret")
    create_user(session, username="bob", password="secret")

    assert [user.username for user in list_users(session)] == [
        "alice",
        "bob",
        "charlie",
    ]


def test_set_role_changes_role(session: Session) -> None:
    create_user(session, username="admin", password="secret")
    user = create_user(session, username="reviewer", password="secret")

    set_role(session, user, UserRole.ADMIN)

    assert user.role == UserRole.ADMIN.value


def test_set_role_purges_sessions_on_change(session: Session) -> None:
    create_user(session, username="admin", password="secret")
    user = create_user(session, username="reviewer", password="secret")
    auth_session = AuthSession(
        id=uuid.uuid4(),
        user_id=user.id,
        token_hash=b"fake-hash-bytes",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    session.add(auth_session)
    session.flush()

    set_role(session, user, UserRole.ADMIN)

    count = session.scalar(
        select(func.count())
        .select_from(AuthSession)
        .where(AuthSession.user_id == user.id)
    )
    assert count == 0


def test_set_role_noop_same_role(session: Session) -> None:
    user = create_user(session, username="admin", password="secret")
    auth_session = AuthSession(
        id=uuid.uuid4(),
        user_id=user.id,
        token_hash=b"fake-hash-bytes",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    session.add(auth_session)
    session.flush()

    set_role(session, user, UserRole.ADMIN)

    count = session.scalar(
        select(func.count())
        .select_from(AuthSession)
        .where(AuthSession.user_id == user.id)
    )
    assert count == 1


def test_set_role_refuses_last_admin_downgrade(session: Session) -> None:
    user = create_user(session, username="admin", password="secret")

    with pytest.raises(ValueError):
        set_role(session, user, UserRole.REVIEWER)

    assert user.role == UserRole.ADMIN.value


def test_set_role_allows_downgrade_with_two_admins(session: Session) -> None:
    user = create_user(session, username="admin-one", password="secret")
    create_user(
        session,
        username="admin-two",
        password="secret",
        role=UserRole.ADMIN,
    )

    set_role(session, user, UserRole.REVIEWER)

    assert user.role == UserRole.REVIEWER.value


def test_set_disabled_disables_user(session: Session) -> None:
    create_user(session, username="admin", password="secret")
    user = create_user(session, username="reviewer", password="secret")
    auth_session = AuthSession(
        id=uuid.uuid4(),
        user_id=user.id,
        token_hash=b"fake-hash-bytes",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    session.add(auth_session)
    session.flush()

    set_disabled(session, user, disabled=True)

    assert user.disabled_at is not None
    count = session.scalar(
        select(func.count())
        .select_from(AuthSession)
        .where(AuthSession.user_id == user.id)
    )
    assert count == 0


def test_set_disabled_refuses_last_admin(session: Session) -> None:
    user = create_user(session, username="admin", password="secret")

    with pytest.raises(ValueError):
        set_disabled(session, user, disabled=True)

    assert user.disabled_at is None


def test_set_disabled_allows_with_two_admins(session: Session) -> None:
    user = create_user(session, username="admin-one", password="secret")
    create_user(
        session,
        username="admin-two",
        password="secret",
        role=UserRole.ADMIN,
    )

    set_disabled(session, user, disabled=True)

    assert user.disabled_at is not None


def test_set_disabled_enables_user(session: Session) -> None:
    create_user(session, username="admin", password="secret")
    user = create_user(session, username="reviewer", password="secret")
    user.disabled_at = datetime.now(UTC)
    session.flush()

    set_disabled(session, user, disabled=False)

    assert user.disabled_at is None


def test_set_disabled_enable_does_not_purge_sessions(session: Session) -> None:
    create_user(session, username="admin", password="secret")
    user = create_user(session, username="reviewer", password="secret")
    user.disabled_at = datetime.now(UTC)
    auth_session = AuthSession(
        id=uuid.uuid4(),
        user_id=user.id,
        token_hash=b"fake-hash-bytes",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    session.add(auth_session)
    session.flush()

    set_disabled(session, user, disabled=False)

    count = session.scalar(
        select(func.count())
        .select_from(AuthSession)
        .where(AuthSession.user_id == user.id)
    )
    assert count == 1


def test_reset_password_updates_hash(session: Session) -> None:
    user = create_user(session, username="admin", password="old-password")
    old_hash = user.password_hash

    reset_password(session, user, "new-password")

    assert user.password_hash != old_hash
    assert verify_password("new-password", user.password_hash)


def test_reset_password_purges_sessions(session: Session) -> None:
    user = create_user(session, username="admin", password="old-password")
    auth_session = AuthSession(
        id=uuid.uuid4(),
        user_id=user.id,
        token_hash=b"fake-hash-bytes",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    session.add(auth_session)
    session.flush()

    reset_password(session, user, "new-password")

    count = session.scalar(
        select(func.count())
        .select_from(AuthSession)
        .where(AuthSession.user_id == user.id)
    )
    assert count == 0


def test_reset_password_rejects_oversized(session: Session) -> None:
    user = create_user(session, username="admin", password="secret")

    with pytest.raises(ValueError):
        reset_password(session, user, "a" * 1025)
