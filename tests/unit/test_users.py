from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from sqlalchemy import CheckConstraint, create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from voxint.db.models import Base, User
from voxint.users import (
    UserRole,
    authenticate,
    create_user,
    hash_password,
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
        Base.metadata.create_all(engine, tables=[User.__table__])
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


def test_empty_password_round_trip() -> None:
    assert verify_password("", hash_password(""))


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
