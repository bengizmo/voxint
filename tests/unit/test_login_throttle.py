import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from starlette.requests import Request

from voxint.api.csrf import CSRF_LOGIN, mint_csrf_token
from voxint.api.login_throttle import LoginThrottle
from voxint.api.routers import auth_pages
from voxint.api.routers.deps import _get_session
from voxint.config import Settings


def test_login_account_limit_expiry_and_backoff():
    now = [0.0]
    throttle = LoginThrottle(clock=lambda: now[0])
    delays = []
    for _ in range(5):
        assert throttle.start("alice", "ip") == 0
        delays.append(throttle.finish("alice", "ip", success=False))
    assert delays == [0, 0, 1, 2, 4]
    assert throttle.start("alice", "different-ip") == 300
    now[0] = 299.1
    assert throttle.start("alice", "ip") == 1
    now[0] = 300
    assert throttle.start("alice", "ip") == 0
    assert throttle.finish("alice", "ip", success=False) == 0


def test_login_source_limit_and_success_reset():
    throttle = LoginThrottle(clock=lambda: 0)
    for i in range(20):
        assert throttle.start(str(i), "ip") == 0
        assert throttle.finish(str(i), "ip", success=False) <= 8
    assert throttle.start("new", "ip") == 300
    assert throttle.start("0", "other-ip") == 0
    throttle.finish("0", "other-ip", success=True)
    assert throttle.start("0", "ip") == 300  # success did not reset source
    for _ in range(5):
        assert throttle.start("0", "other-ip") == 0
        throttle.finish("0", "other-ip", success=False)
    assert throttle.start("0", "third-ip") == 300


def test_login_reservations_and_errors_are_not_failures():
    throttle = LoginThrottle(clock=lambda: 0)
    for _ in range(5):
        assert throttle.start("alice", "ip") == 0
    assert throttle.start("alice", "ip") == 1
    for _ in range(5):
        throttle.finish("alice", "ip", success=None)
    assert throttle.start("alice", "ip") == 0
    assert throttle.finish("alice", "ip", success=False) == 0


@pytest.fixture
def login_env(monkeypatch):
    throttle = LoginThrottle(clock=lambda: 0)
    monkeypatch.setattr(auth_pages, "login_throttle", throttle)
    monkeypatch.setattr(auth_pages, "argon2_slots", asyncio.Semaphore(3))
    app = FastAPI()
    app.state.settings = Settings(voxint_multi_user=True)
    app.state.csrf_secret = "test-secret"
    app.include_router(auth_pages.router)
    app.dependency_overrides[_get_session] = lambda: Mock()
    request = Request({
        "type": "http", "app": app, "client": ("ip", 123),
        "scheme": "http", "server": ("test", 80), "path": "/login", "headers": [],
    })
    return app, request, mint_csrf_token("test-secret", CSRF_LOGIN)


async def test_login_http_429_normalization_and_backoff(login_env, monkeypatch):
    app, _, csrf = login_env
    authenticate = Mock(return_value=None)
    monkeypatch.setattr(auth_pages, "authenticate", authenticate)
    delays = []

    async def sleep(delay):
        # Backoff never holds a verification slot.
        assert not auth_pages.argon2_slots.locked()
        delays.append(delay)

    monkeypatch.setattr(auth_pages.asyncio, "sleep", sleep)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for username in [" Alice ", "alice", "ALICE", "alice", "alice", "alice"]:
            response = await client.post("/login", data={
                "username": username, "password": "wrong", "csrf_token": csrf,
            })
        assert response.status_code == 429
        assert response.headers["Retry-After"] == "300"
        assert authenticate.call_count == 5
        assert all(call.kwargs["username"] == "alice" for call in authenticate.call_args_list)
        assert delays == [1, 2, 4]


async def test_login_success_and_csrf(login_env, monkeypatch):
    _, request, csrf = login_env
    throttle = auth_pages.login_throttle
    for _ in range(2):
        throttle.start("alice", "ip")
        throttle.finish("alice", "ip", success=False)
    authenticate = Mock(return_value=SimpleNamespace(id="user-id"))
    monkeypatch.setattr(auth_pages, "authenticate", authenticate)
    monkeypatch.setattr(auth_pages, "cleanup_expired_sessions", Mock())
    monkeypatch.setattr(auth_pages, "create_session", Mock())
    response = await auth_pages.login_submit(request, Mock(), "alice", "pw", "bad", "/")
    assert response.status_code == 403
    authenticate.assert_not_called()
    response = await auth_pages.login_submit(request, Mock(), "alice", "pw", csrf, "/")
    assert response.status_code == 303
    assert "voxint_session=" in response.headers["set-cookie"]
    for _ in range(5):
        assert throttle.start("alice", "other-ip") == 0
        throttle.finish("alice", "other-ip", success=False)


async def test_login_concurrency_and_cancellation(login_env, monkeypatch):
    _, request, csrf = login_env
    release = threading.Event()
    started = [threading.Event() for _ in range(3)]

    def authenticate(session, *, username, password):
        started[int(username)].set()
        assert release.wait(timeout=5)
        return None

    monkeypatch.setattr(auth_pages, "authenticate", authenticate)
    tasks = [asyncio.create_task(auth_pages.login_submit(
        request, Mock(), str(i), "pw", csrf, "/"
    )) for i in range(3)]
    try:
        async with asyncio.timeout(3):
            while not all(event.is_set() for event in started):
                await asyncio.sleep(0.001)
        tasks[0].cancel()
        await asyncio.sleep(0)
        with pytest.raises(HTTPException) as exc:
            await auth_pages.login_submit(request, Mock(), "extra", "pw", csrf, "/")
        assert exc.value.status_code == 429
        assert exc.value.headers == {"Retry-After": "1"}
        assert not tasks[0].done()  # cancelled verifier still holds its slot
    finally:
        release.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
    assert isinstance(results[0], asyncio.CancelledError)
    assert all(result.status_code == 401 for result in results[1:])
    assert not auth_pages.argon2_slots.locked()


async def test_login_auth_error_releases_capacity(login_env, monkeypatch):
    _, request, csrf = login_env
    monkeypatch.setattr(auth_pages, "authenticate", Mock(side_effect=RuntimeError("db error")))
    for _ in range(6):
        with pytest.raises(RuntimeError, match="db error"):
            await auth_pages.login_submit(request, Mock(), "alice", "pw", csrf, "/")
    assert not auth_pages.argon2_slots.locked()
