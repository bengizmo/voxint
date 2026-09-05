"""Integration tests for self-service password change (#364).

Exercises the /account/password GET and POST flow: happy path with session
rotation, wrong-current-password rejection, confirmation mismatch, CSRF
validation, viewer access, and single-operator 404.
"""

import uuid

from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.auth import SESSION_COOKIE
from voxint.api.csrf import (
    CSRF_ACCOUNT_PASSWORD,
    CSRF_LOGIN,
    CSRF_USERS,
    mint_csrf_token,
)
from voxint.config import Settings
from voxint.db.models import AuthSession, User, UserRole
from voxint.users import create_user, verify_password

_CSRF_SECRET = "account-password-test-csrf-secret"
_ADMIN_PW = "admin-pass-123"
_REVIEWER_PW = "reviewer-pass-456"
_VIEWER_PW = "viewer-pass-789"


def _make_settings(**overrides: object) -> Settings:
    return Settings(
        voxint_multi_user=True,
        voxint_user="ignored",
        voxint_password="ignored",
        csrf_secret=_CSRF_SECRET,
        **overrides,  # type: ignore[arg-type]
    )


def _seed_users(
    session_factory: sessionmaker[Session],
) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    with session_factory() as db:
        admin = create_user(  # pragma: allowlist secret
            db, username="admin", password=_ADMIN_PW
        )
        reviewer = create_user(  # pragma: allowlist secret
            db, username="reviewer", password=_REVIEWER_PW, role=UserRole.REVIEWER
        )
        viewer = create_user(  # pragma: allowlist secret
            db, username="viewer", password=_VIEWER_PW, role=UserRole.VIEWER
        )
        db.commit()
        return admin.id, reviewer.id, viewer.id


def _make_client(session_factory: sessionmaker[Session], **kw: object) -> TestClient:
    settings = _make_settings(**kw)
    app = create_app(settings=settings, session_factory=session_factory)
    return TestClient(app, follow_redirects=False)


def _login(client: TestClient, username: str, password: str) -> TestClient:
    csrf = mint_csrf_token(_CSRF_SECRET, CSRF_LOGIN)
    resp = client.post(
        "/login",
        data={
            "username": username,
            "password": password,
            "csrf_token": csrf,
            "next": "/",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303, f"login failed: {resp.status_code}"
    assert SESSION_COOKIE in client.cookies
    return client


def _csrf() -> str:
    return mint_csrf_token(_CSRF_SECRET, CSRF_ACCOUNT_PASSWORD)


def _post_password(
    client: TestClient,
    current: str,
    new: str,
    confirm: str | None = None,
    csrf_token: str | None = None,
) -> object:
    return client.post(
        "/account/password",
        data={
            "current_password": current,
            "new_password": new,
            "new_password_confirm": confirm if confirm is not None else new,
            "csrf_token": csrf_token if csrf_token is not None else _csrf(),
        },
        follow_redirects=False,
    )


class TestHappyPath:
    def test_get_renders_form(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        seed_onboarded(session_factory)
        _seed_users(session_factory)
        client = _make_client(session_factory)
        _login(client, "admin", _ADMIN_PW)

        resp = client.get("/account/password", headers={"accept": "text/html"})
        assert resp.status_code == 200
        assert 'name="current_password"' in resp.text
        assert 'name="new_password"' in resp.text

    def test_password_change_succeeds(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        seed_onboarded(session_factory)
        admin_id, _, _ = _seed_users(session_factory)
        client = _make_client(session_factory)
        _login(client, "admin", _ADMIN_PW)

        resp = _post_password(client, _ADMIN_PW, "brand-new-pw")
        assert resp.status_code == 303
        assert "ok=1" in resp.headers["location"]

        with session_factory() as db:
            user = db.get(User, admin_id)
            assert user is not None
            assert verify_password("brand-new-pw", user.password_hash)
            assert not verify_password(_ADMIN_PW, user.password_hash)

    def test_success_message_on_redirect(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        seed_onboarded(session_factory)
        _seed_users(session_factory)
        client = _make_client(session_factory)
        _login(client, "admin", _ADMIN_PW)

        resp = _post_password(client, _ADMIN_PW, "new-pw")
        assert resp.status_code == 303
        follow = client.get(
            resp.headers["location"], headers={"accept": "text/html"}
        )
        assert follow.status_code == 200
        assert "Password changed" in follow.text
        assert "notice--ok" in follow.text


class TestSessionRotation:
    def test_current_session_rotated(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        seed_onboarded(session_factory)
        _seed_users(session_factory)
        client = _make_client(session_factory)
        _login(client, "admin", _ADMIN_PW)
        old_cookie = client.cookies[SESSION_COOKIE]

        _post_password(client, _ADMIN_PW, "new-pw")
        new_cookie = client.cookies[SESSION_COOKIE]

        assert new_cookie != old_cookie
        resp = client.get("/account/password", headers={"accept": "text/html"})
        assert resp.status_code == 200

    def test_other_sessions_invalidated(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        seed_onboarded(session_factory)
        admin_id, _, _ = _seed_users(session_factory)
        app_settings = _make_settings()
        app = create_app(settings=app_settings, session_factory=session_factory)
        client_a = TestClient(app, follow_redirects=False)
        client_b = TestClient(app, follow_redirects=False)

        _login(client_a, "admin", _ADMIN_PW)
        _login(client_b, "admin", _ADMIN_PW)

        with session_factory() as db:
            count_before = db.scalar(
                select(func.count())
                .select_from(AuthSession)
                .where(AuthSession.user_id == admin_id)
            )
        assert count_before == 2

        _post_password(client_a, _ADMIN_PW, "new-pw")

        with session_factory() as db:
            count_after = db.scalar(
                select(func.count())
                .select_from(AuthSession)
                .where(AuthSession.user_id == admin_id)
            )
        assert count_after == 1

        resp_b = client_b.get("/account/password", headers={"accept": "text/html"})
        assert resp_b.status_code == 303
        assert "/login" in resp_b.headers["location"]


class TestValidation:
    def test_wrong_current_password(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        seed_onboarded(session_factory)
        admin_id, _, _ = _seed_users(session_factory)
        client = _make_client(session_factory)
        _login(client, "admin", _ADMIN_PW)

        resp = _post_password(client, "wrong-password", "new-pw")
        assert resp.status_code == 400
        assert "incorrect" in resp.text.lower()

        with session_factory() as db:
            user = db.get(User, admin_id)
            assert user is not None
            assert verify_password(_ADMIN_PW, user.password_hash)

    def test_confirmation_mismatch(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        seed_onboarded(session_factory)
        client = _make_client(session_factory)
        _seed_users(session_factory)
        _login(client, "admin", _ADMIN_PW)

        resp = _post_password(client, _ADMIN_PW, "new-pw", confirm="different")
        assert resp.status_code == 400
        assert "do not match" in resp.text.lower()

    def test_empty_new_password(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        seed_onboarded(session_factory)
        _seed_users(session_factory)
        client = _make_client(session_factory)
        _login(client, "admin", _ADMIN_PW)

        resp = _post_password(client, _ADMIN_PW, "")
        assert resp.status_code == 400

    def test_invalid_csrf(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        seed_onboarded(session_factory)
        _seed_users(session_factory)
        client = _make_client(session_factory)
        _login(client, "admin", _ADMIN_PW)

        resp = _post_password(client, _ADMIN_PW, "new-pw", csrf_token="bogus")
        assert resp.status_code == 403

    def test_csrf_wrong_action_rejected(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        seed_onboarded(session_factory)
        _seed_users(session_factory)
        client = _make_client(session_factory)
        _login(client, "admin", _ADMIN_PW)

        login_token = mint_csrf_token(_CSRF_SECRET, CSRF_LOGIN)
        resp = _post_password(client, _ADMIN_PW, "new-pw", csrf_token=login_token)
        assert resp.status_code == 403

        users_token = mint_csrf_token(_CSRF_SECRET, CSRF_USERS)
        resp = _post_password(client, _ADMIN_PW, "new-pw", csrf_token=users_token)
        assert resp.status_code == 403


class TestRoles:
    def test_reviewer_can_change_password(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        seed_onboarded(session_factory)
        _, reviewer_id, _ = _seed_users(session_factory)
        client = _make_client(session_factory)
        _login(client, "reviewer", _REVIEWER_PW)

        resp = _post_password(client, _REVIEWER_PW, "reviewer-new-pw")
        assert resp.status_code == 303

        with session_factory() as db:
            user = db.get(User, reviewer_id)
            assert user is not None
            assert verify_password("reviewer-new-pw", user.password_hash)

    def test_viewer_can_change_password(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        seed_onboarded(session_factory)
        _, _, viewer_id = _seed_users(session_factory)
        client = _make_client(session_factory)
        _login(client, "viewer", _VIEWER_PW)

        resp = _post_password(client, _VIEWER_PW, "viewer-new-pw")
        assert resp.status_code == 303

        with session_factory() as db:
            user = db.get(User, viewer_id)
            assert user is not None
            assert verify_password("viewer-new-pw", user.password_hash)


class TestSingleOperatorMode:
    def test_get_redirects(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        seed_onboarded(session_factory)
        client = TestClient(
            create_app(
                settings=Settings(
                    voxint_multi_user=False,
                    voxint_user="op",
                    voxint_password="secret",
                    csrf_secret=_CSRF_SECRET,
                ),
                session_factory=session_factory,
            ),
            follow_redirects=False,
        )
        client.auth = ("op", "secret")
        resp = client.get("/account/password", headers={"accept": "text/html"})
        assert resp.status_code == 302

    def test_post_returns_404(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        seed_onboarded(session_factory)
        client = TestClient(
            create_app(
                settings=Settings(
                    voxint_multi_user=False,
                    voxint_user="op",
                    voxint_password="secret",
                    csrf_secret=_CSRF_SECRET,
                ),
                session_factory=session_factory,
            ),
            follow_redirects=False,
        )
        client.auth = ("op", "secret")
        resp = _post_password(client, "secret", "new-pw")
        assert resp.status_code == 404
