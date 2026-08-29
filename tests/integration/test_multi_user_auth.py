"""Integration tests for multi-user session auth lifecycle (#307).

Exercises the dual-mode identity resolver, session cookies, role gating,
attribution persistence, logout, disabled-user revocation, and Basic auth
rejection -- all against real Postgres.
"""

import uuid

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.auth import SESSION_COOKIE
from voxint.api.csrf import CSRF_LOGIN, CSRF_LOGOUT, mint_csrf_token
from voxint.config import Settings
from voxint.db.models import AuthSession, User, UserRole
from voxint.users import create_user

_CSRF_SECRET = "multi-user-test-csrf-secret"
_ADMIN_PW = "admin-pass-123"
_REVIEWER_PW = "reviewer-pass-456"


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
) -> tuple[uuid.UUID, uuid.UUID]:
    """Create an admin and a reviewer, returning (admin_id, reviewer_id)."""
    with session_factory() as db:
        admin = create_user(db, username="admin", password=_ADMIN_PW)
        reviewer = create_user(
            db, username="reviewer", password=_REVIEWER_PW, role=UserRole.REVIEWER
        )
        db.commit()
        return admin.id, reviewer.id


def _login(client: TestClient, username: str, password: str) -> TestClient:
    """POST /login and return the same client (cookie jar is updated)."""
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
    assert resp.status_code == 303, f"login failed: {resp.status_code} {resp.text}"
    assert SESSION_COOKIE in client.cookies
    return client


def _make_client(session_factory: sessionmaker[Session], **kw: object) -> TestClient:
    settings = _make_settings(**kw)
    app = create_app(settings=settings, session_factory=session_factory)
    return TestClient(app, follow_redirects=False)


# ---------------------------------------------------------------- login flow


class TestLoginFlow:
    def test_login_sets_session_cookie(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        seed_onboarded(session_factory)
        _seed_users(session_factory)
        client = _make_client(session_factory)

        csrf = mint_csrf_token(_CSRF_SECRET, CSRF_LOGIN)
        resp = client.post(
            "/login",
            data={
                "username": "admin",
                "password": _ADMIN_PW,
                "csrf_token": csrf,
                "next": "/",
            },
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert resp.headers["location"] == "/"
        assert SESSION_COOKIE in client.cookies

    def test_authenticated_request_succeeds(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        seed_onboarded(session_factory)
        _seed_users(session_factory)
        client = _make_client(session_factory)
        _login(client, "admin", _ADMIN_PW)

        resp = client.get("/", headers={"accept": "text/html"})
        assert resp.status_code == 200


# --------------------------------------------------------- attribution


class TestAttribution:
    def test_login_creates_session_with_correct_user_id(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        """Login persists an auth_sessions row attributed to the logged-in user."""
        seed_onboarded(session_factory)
        admin_id, _ = _seed_users(session_factory)
        client = _make_client(session_factory)
        _login(client, "admin", _ADMIN_PW)

        with session_factory() as db:
            row = db.execute(
                select(AuthSession).where(AuthSession.user_id == admin_id)
            ).scalar_one_or_none()
        assert row is not None
        assert row.user_id == admin_id


# --------------------------------------------------------- role gate


class TestRoleGate:
    def test_reviewer_settings_returns_403(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        seed_onboarded(session_factory)
        _seed_users(session_factory)
        client = _make_client(session_factory)
        _login(client, "reviewer", _REVIEWER_PW)

        resp = client.get("/settings", headers={"accept": "text/html"})
        assert resp.status_code == 403

    def test_admin_settings_returns_200(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        seed_onboarded(session_factory)
        _seed_users(session_factory)
        client = _make_client(session_factory)
        _login(client, "admin", _ADMIN_PW)

        resp = client.get("/settings", headers={"accept": "text/html"})
        assert resp.status_code == 200

    def test_settings_link_hidden_for_reviewer(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        seed_onboarded(session_factory)
        _seed_users(session_factory)
        client = _make_client(session_factory)
        _login(client, "reviewer", _REVIEWER_PW)

        resp = client.get("/", headers={"accept": "text/html"})
        assert resp.status_code == 200
        assert 'href="/settings"' not in resp.text

    def test_settings_link_visible_for_admin(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        seed_onboarded(session_factory)
        _seed_users(session_factory)
        client = _make_client(session_factory)
        _login(client, "admin", _ADMIN_PW)

        resp = client.get("/", headers={"accept": "text/html"})
        assert resp.status_code == 200
        assert 'href="/settings"' in resp.text


# --------------------------------------------------------- logout


class TestLogout:
    def test_logout_clears_session(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        seed_onboarded(session_factory)
        _seed_users(session_factory)
        client = _make_client(session_factory)
        _login(client, "admin", _ADMIN_PW)

        csrf = mint_csrf_token(_CSRF_SECRET, CSRF_LOGOUT)
        resp = client.post(
            "/logout",
            data={"csrf_token": csrf},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert resp.headers["location"] == "/login"

        # Server-side session row should be deleted
        with session_factory() as db:
            count = db.execute(
                select(AuthSession)
            ).scalars().all()
            assert len(count) == 0

        # Subsequent request should be unauthenticated
        resp = client.get("/", headers={"accept": "text/html"})
        assert resp.status_code == 303
        assert "/login" in resp.headers["location"]


# ------------------------------------------------- disabled user revocation


class TestDisabledUser:
    def test_disabled_user_session_rejected(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        seed_onboarded(session_factory)
        _, reviewer_id = _seed_users(session_factory)
        client = _make_client(session_factory)
        _login(client, "reviewer", _REVIEWER_PW)

        # Verify the session works before disabling
        resp = client.get("/", headers={"accept": "text/html"})
        assert resp.status_code == 200

        # Disable the user out-of-band
        with session_factory() as db:
            user = db.get(User, reviewer_id)
            assert user is not None
            from datetime import UTC, datetime

            user.disabled_at = datetime.now(UTC)
            db.commit()

        # Existing session cookie should now be rejected
        resp = client.get("/", headers={"accept": "text/html"})
        assert resp.status_code == 303
        assert "/login" in resp.headers["location"]


# ------------------------------------------------ basic auth rejection


class TestBasicAuthRejection:
    def test_basic_auth_rejected_in_multi_user_mode(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        seed_onboarded(session_factory)
        _seed_users(session_factory)
        client = _make_client(session_factory)

        resp = client.get(
            "/",
            auth=("admin", _ADMIN_PW),
            headers={"accept": "text/html"},
        )
        # Multi-user mode does not honour Basic auth -- should redirect to login
        assert resp.status_code == 303
        assert "/login" in resp.headers["location"]


# --------------------------------------------- single-user contrast


class TestSingleUserContrast:
    def test_single_user_basic_auth_works(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        seed_onboarded(session_factory)
        settings = Settings(
            voxint_multi_user=False,
            voxint_user="operator",
            voxint_password="s3cret",
            csrf_secret=_CSRF_SECRET,
        )
        app = create_app(settings=settings, session_factory=session_factory)
        client = TestClient(app, follow_redirects=False)
        client.auth = ("operator", "s3cret")

        resp = client.get("/", headers={"accept": "text/html"})
        assert resp.status_code == 200

    def test_single_user_settings_link_visible(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        seed_onboarded(session_factory)
        settings = Settings(
            voxint_multi_user=False,
            voxint_user="operator",
            voxint_password="s3cret",
            csrf_secret=_CSRF_SECRET,
        )
        app = create_app(settings=settings, session_factory=session_factory)
        client = TestClient(app, follow_redirects=False)
        client.auth = ("operator", "s3cret")

        resp = client.get("/", headers={"accept": "text/html"})
        assert resp.status_code == 200
        assert 'href="/settings"' in resp.text

    def test_login_returns_404_in_single_user_mode(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        seed_onboarded(session_factory)
        settings = Settings(
            voxint_multi_user=False,
            voxint_user="operator",
            voxint_password="s3cret",
            csrf_secret=_CSRF_SECRET,
        )
        app = create_app(settings=settings, session_factory=session_factory)
        client = TestClient(app, follow_redirects=False)
        client.auth = ("operator", "s3cret")

        resp = client.get("/login")
        assert resp.status_code == 404
