"""Unit tests for the API key service layer."""

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from voxint.api_keys import (
    TOKEN_PREFIX,
    create_api_key,
    list_api_keys,
    revoke_api_key,
    verify_api_key,
)
from voxint.db.models import ApiKey, Base, User


@pytest.fixture()
def session() -> Iterator[Session]:
    engine = create_engine("sqlite://")
    from sqlalchemy import CheckConstraint as CC

    pg_checks: list[tuple[Any, list[CC]]] = []
    for table in (User.__table__, ApiKey.__table__):
        removed = [
            c
            for c in table.constraints
            if isinstance(c, CC)
            and c.name
            and (
                "format_check" in c.name
                or "nonempty_check" in c.name
                or "no_colon_check" in c.name
            )
        ]
        for c in removed:
            table.constraints.remove(c)
        pg_checks.append((table, removed))
    try:
        Base.metadata.create_all(
            engine, tables=[User.__table__, ApiKey.__table__]
        )
    finally:
        for table, checks in pg_checks:
            for c in checks:
                table.append_constraint(c)
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as db_session:
        yield db_session
    engine.dispose()


def _make_user(session: Session, *, role: str = "admin") -> User:
    user = User(
        id=uuid.uuid4(),
        username="testuser",
        password_hash="$argon2id$v=19$m=1,t=1,p=1$c2FsdA$dummy",
        role=role,
    )
    session.add(user)
    session.flush()
    return user


class TestCreateApiKey:
    def test_creates_key_with_prefix(self, session: Session) -> None:
        result = create_api_key(session, name="test key")
        assert result.plaintext_token.startswith(TOKEN_PREFIX)
        assert result.name == "test key"
        assert len(result.key_prefix) == 12

    def test_rejects_blank_name(self, session: Session) -> None:
        with pytest.raises(ValueError, match="blank"):
            create_api_key(session, name="")

    def test_rejects_whitespace_name(self, session: Session) -> None:
        with pytest.raises(ValueError, match="blank"):
            create_api_key(session, name="   ")

    def test_stores_hash_not_plaintext(self, session: Session) -> None:
        result = create_api_key(session, name="test")
        row = session.get(ApiKey, result.id)
        assert row is not None
        assert result.plaintext_token.encode("ascii") != row.key_hash

    def test_user_id_nullable(self, session: Session) -> None:
        result = create_api_key(session, name="instance key")
        row = session.get(ApiKey, result.id)
        assert row is not None
        assert row.user_id is None

    def test_user_id_set(self, session: Session) -> None:
        user = _make_user(session)
        result = create_api_key(session, name="user key", user_id=user.id)
        row = session.get(ApiKey, result.id)
        assert row is not None
        assert row.user_id == user.id


class TestVerifyApiKey:
    def test_valid_key_single_op(self, session: Session) -> None:
        created = create_api_key(session, name="test")
        session.flush()
        result = verify_api_key(session, created.plaintext_token, multi_user=False)
        assert result is not None
        key_row, user = result
        assert key_row.id == created.id
        assert user is None

    def test_valid_key_multi_user(self, session: Session) -> None:
        user = _make_user(session)
        created = create_api_key(session, name="test", user_id=user.id)
        session.flush()
        result = verify_api_key(session, created.plaintext_token, multi_user=True)
        assert result is not None
        key_row, resolved_user = result
        assert key_row.id == created.id
        assert resolved_user is not None
        assert resolved_user.id == user.id

    def test_rejects_bad_token(self, session: Session) -> None:
        assert verify_api_key(session, "vxint_bogus", multi_user=False) is None

    def test_rejects_non_prefixed(self, session: Session) -> None:
        assert verify_api_key(session, "not_a_key", multi_user=False) is None

    def test_rejects_revoked_key(self, session: Session) -> None:
        created = create_api_key(session, name="test")
        session.flush()
        revoke_api_key(session, created.id)
        session.flush()
        assert verify_api_key(session, created.plaintext_token, multi_user=False) is None

    def test_rejects_expired_key(self, session: Session) -> None:
        created = create_api_key(session, name="test")
        session.flush()
        row = session.get(ApiKey, created.id)
        assert row is not None
        row.expires_at = datetime.now(UTC) - timedelta(hours=1)
        session.flush()
        assert verify_api_key(session, created.plaintext_token, multi_user=False) is None

    def test_rejects_user_key_in_single_op(self, session: Session) -> None:
        user = _make_user(session)
        created = create_api_key(session, name="test", user_id=user.id)
        session.flush()
        assert verify_api_key(session, created.plaintext_token, multi_user=False) is None

    def test_rejects_disabled_user(self, session: Session) -> None:
        user = _make_user(session)
        created = create_api_key(session, name="test", user_id=user.id)
        session.flush()
        user.disabled_at = datetime.now(UTC)
        session.flush()
        assert verify_api_key(session, created.plaintext_token, multi_user=True) is None

    def test_updates_last_used_at(self, session: Session) -> None:
        created = create_api_key(session, name="test")
        session.flush()
        row = session.get(ApiKey, created.id)
        assert row is not None
        assert row.last_used_at is None
        verify_api_key(session, created.plaintext_token, multi_user=False)
        assert row.last_used_at is not None

    def test_instance_key_in_multi_user(self, session: Session) -> None:
        created = create_api_key(session, name="instance")
        session.flush()
        result = verify_api_key(session, created.plaintext_token, multi_user=True)
        assert result is not None
        key_row, user = result
        assert key_row.id == created.id
        assert user is None


class TestRevokeApiKey:
    def test_revokes_existing_key(self, session: Session) -> None:
        created = create_api_key(session, name="test")
        session.flush()
        row = revoke_api_key(session, created.id)
        assert row is not None
        assert row.revoked_at is not None

    def test_idempotent_revoke(self, session: Session) -> None:
        created = create_api_key(session, name="test")
        session.flush()
        revoke_api_key(session, created.id)
        first_revoked = session.get(ApiKey, created.id)
        assert first_revoked is not None
        ts = first_revoked.revoked_at
        revoke_api_key(session, created.id)
        assert first_revoked.revoked_at == ts

    def test_revoke_nonexistent(self, session: Session) -> None:
        assert revoke_api_key(session, uuid.uuid4()) is None


class TestListApiKeys:
    def test_lists_active_keys(self, session: Session) -> None:
        create_api_key(session, name="key1")
        create_api_key(session, name="key2")
        session.flush()
        keys = list_api_keys(session)
        assert len(keys) == 2

    def test_excludes_revoked_by_default(self, session: Session) -> None:
        created = create_api_key(session, name="key1")
        create_api_key(session, name="key2")
        session.flush()
        revoke_api_key(session, created.id)
        session.flush()
        keys = list_api_keys(session)
        assert len(keys) == 1
        assert keys[0].name == "key2"

    def test_includes_revoked_when_asked(self, session: Session) -> None:
        created = create_api_key(session, name="key1")
        create_api_key(session, name="key2")
        session.flush()
        revoke_api_key(session, created.id)
        session.flush()
        keys = list_api_keys(session, include_revoked=True)
        assert len(keys) == 2
