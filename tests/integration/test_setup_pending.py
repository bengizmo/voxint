"""Non-admin first-run waiting page, using real users and Postgres sessions."""

import re

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from tests.integration.test_multi_user_auth import _CSRF_SECRET, _login, _make_client
from voxint.api.app import create_app
from voxint.api.auth import SESSION_COOKIE
from voxint.api.csrf import CSRF_SETUP, mint_csrf_token
from voxint.api.routers.deps import templates
from voxint.app_settings import is_onboarded
from voxint.config import Settings
from voxint.db.models import UserRole
from voxint.users import create_user

_PASSWORD = "pending-test-password"


@pytest.fixture()
def pending_client(
    session_factory: sessionmaker[Session], request: pytest.FixtureRequest
) -> TestClient:
    role: UserRole = getattr(request, "param", UserRole.REVIEWER)
    with session_factory() as db:
        # The first account is always promoted to admin.
        create_user(db, username="admin", password=_PASSWORD)
        if role != UserRole.ADMIN:
            create_user(db, username=role.value, password=_PASSWORD, role=role)
        db.commit()
    return _login(_make_client(session_factory), role.value, _PASSWORD)


@pytest.mark.parametrize("pending_client", [UserRole.REVIEWER, UserRole.VIEWER], indirect=True)
@pytest.mark.parametrize("htmx", [False, True])
@pytest.mark.parametrize("path", ["/", "/runs"])
def test_non_admin_gate(pending_client: TestClient, htmx: bool, path: str) -> None:
    response = pending_client.get(path, headers={"HX-Request": "true"} if htmx else {})
    if htmx:
        assert response.status_code == 204
        assert response.headers["hx-redirect"] == "/setup/pending"
        assert "location" not in response.headers
        assert response.text == ""
    else:
        assert response.status_code == 303
        assert response.headers["location"] == "/setup/pending"


@pytest.mark.parametrize("pending_client", [UserRole.REVIEWER, UserRole.VIEWER], indirect=True)
def test_pending_page_and_logout(pending_client: TestClient) -> None:
    response = pending_client.get("/setup/pending")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "<h1>Setup in progress</h1>" in response.text
    assert (
        "This Voxint instance is still being set up. An administrator needs to "
        "finish setup before you can use it."
    ) in response.text
    assert "You can reload this page once setup is done." in response.text
    assert '<a href="/">Reload</a>' in response.text
    assert "403" not in response.text
    assert 'class="wizard-steps"' not in response.text
    form = re.search(r'<form method="post" action="/logout">(.*?)</form>', response.text, re.S)
    assert form is not None
    token = re.search(r'name="csrf_token" value="([^"]+)"', form.group(1))
    assert token is not None
    assert "Sign out" in form.group(1)
    logout = pending_client.post("/logout", data={"csrf_token": token.group(1)})
    assert logout.status_code == 303
    assert logout.headers["location"] == "/login"
    assert SESSION_COOKIE not in pending_client.cookies
    assert pending_client.get("/setup/pending").status_code == 401


@pytest.mark.parametrize(
    "pending_client", [UserRole.ADMIN, UserRole.REVIEWER, UserRole.VIEWER], indirect=True
)
def test_pending_after_onboarding(
    pending_client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    seed_onboarded(session_factory)
    response = pending_client.get("/setup/pending")
    assert response.status_code == 303
    assert response.headers["location"] == "/"
    response = pending_client.get("/setup/pending", headers={"HX-Request": "true"})
    assert response.status_code == 204
    assert response.headers["hx-redirect"] == "/"
    home = pending_client.get("/")
    assert home.status_code == 200
    assert "Setup in progress" not in home.text


@pytest.mark.parametrize("pending_client", [UserRole.ADMIN], indirect=True)
def test_admin_goes_to_wizard(pending_client: TestClient) -> None:
    for path in ("/", "/setup/pending"):
        response = pending_client.get(path)
        assert response.status_code == 303
        assert response.headers["location"] == "/setup"
        response = pending_client.get(path, headers={"HX-Request": "true"})
        assert response.status_code == 204
        assert response.headers["hx-redirect"] == "/setup"


@pytest.mark.parametrize("onboarded", [False, True])
def test_pending_requires_authentication(
    session_factory: sessionmaker[Session], onboarded: bool
) -> None:
    if onboarded:
        seed_onboarded(session_factory)
    client = _make_client(session_factory)
    response = client.get("/setup/pending")
    protected = client.get("/")
    assert response.status_code == protected.status_code == 401
    assert response.json() == protected.json() == {"detail": "authentication required"}
    assert "www-authenticate" not in response.headers
    assert "location" not in response.headers
    assert "hx-redirect" not in response.headers
    assert "Setup in progress" not in response.text


def test_reviewer_cannot_use_wizard(
    pending_client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    assert pending_client.get("/setup").status_code == 403
    # A valid setup token gets past the handler's CSRF check, so only the
    # router's admin gate can produce this 403.
    finish = pending_client.post(
        "/setup/finish", data={"csrf_token": mint_csrf_token(_CSRF_SECRET, CSRF_SETUP)}
    )
    assert finish.status_code == 403
    assert finish.json() == {"detail": "admin role required"}
    with session_factory() as db:
        assert not is_onboarded(db)


@pytest.mark.parametrize("onboarded", [False, True])
def test_pending_browser_login_challenge(
    session_factory: sessionmaker[Session], onboarded: bool
) -> None:
    if onboarded:
        seed_onboarded(session_factory)
    client = _make_client(session_factory)
    for path in ("/", "/setup/pending"):
        response = client.get(path, headers={"Accept": "text/html"})
        assert response.status_code == 303
        assert response.headers["location"] == f"/login?next={path}"
        response = client.get(path, headers={"HX-Request": "true"})
        assert response.status_code == 204
        assert response.headers["hx-redirect"] == "/login"
        assert response.text == ""


def test_pending_template_hides_logout_in_single_user_mode() -> None:
    html = templates.env.get_template("settings/setup_pending.html").render(
        shell={"multi_user": False, "current_user": {"username": "operator", "role": "admin"}}
    )
    assert "Setup in progress" in html
    assert 'action="/logout"' not in html
    assert "Sign out" not in html


def test_single_user_operator_goes_to_wizard(session_factory: sessionmaker[Session]) -> None:
    settings = Settings(
        voxint_user="operator", voxint_password="single-user-pass", csrf_secret=_CSRF_SECRET
    )
    client = TestClient(
        create_app(settings=settings, session_factory=session_factory), follow_redirects=False
    )
    client.auth = ("operator", "single-user-pass")
    for path in ("/", "/setup/pending"):
        response = client.get(path)
        assert response.status_code == 303
        assert response.headers["location"] == "/setup"
        assert "Setup in progress" not in response.text
