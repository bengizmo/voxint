"""Service lifecycle controls and settings routes (#556), without Docker or a DB."""

import json
import os
import threading
from unittest.mock import Mock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from voxint.api.csrf import CSRF_SERVICE_CONTROL, mint_csrf_token
from voxint.api.routers import deps
from voxint.api.routers import settings as routes
from voxint.api.service_control import (
    SERVICE_KEYS,
    SERVICES,
    ControlOutcome,
    ControlResult,
    DockerController,
    NoopController,
    ServiceDef,
    ServiceState,
    get_controller,
)
from voxint.config import Settings


def test_service_registry() -> None:
    assert {"transcription", "diarization", "speaker_embedding"} == SERVICE_KEYS
    assert SERVICES.keys() == SERVICE_KEYS
    settings = Settings(_env_file=None)
    for key, service in SERVICES.items():
        assert isinstance(service, ServiceDef)
        assert service.key == key
        assert service.compose_service and service.label and service.launchd_label
        assert hasattr(settings, service.settings_url_attr)


def test_noop_controller() -> None:
    for kind in ("docker", "native"):
        controller = NoopController("disabled", install_kind=kind)
        assert controller.controllable is False
        assert controller.backend_name == "none"
        for key, service in SERVICES.items():
            assert controller.restart(key) == ControlResult(ControlOutcome.UNAVAILABLE, "disabled")
            assert controller.inspect(key) == ServiceState.UNKNOWN
            expected = (
                f"docker compose restart {service.compose_service}"
                if kind == "docker"
                else f"launchctl kickstart -k gui/$(id -u)/{service.launchd_label}"
            )
            assert controller.terminal_hint(key) == expected
        assert controller.terminal_hint("invalid_key") is None


@pytest.fixture
def docker(monkeypatch: pytest.MonkeyPatch):
    """Keep the real client/request code; replace only its socket transport."""
    requests = []

    def install(handler):
        def record(request):
            requests.append(request)
            return handler(request)

        monkeypatch.setattr(httpx, "HTTPTransport", lambda **kwargs: httpx.MockTransport(record))
        return DockerController("/test/docker.sock", "test-project"), requests

    return install


@pytest.mark.parametrize("count", [0, 1, 2])
def test_find_container(docker, count: int) -> None:
    controller, requests = docker(
        lambda request: httpx.Response(200, json=[{"Id": str(i)} for i in range(count)])
    )
    with controller._client() as client:
        assert controller._find_container(client, "whisper") == ("0" if count == 1 else None)
    request = requests[0]
    assert request.url.path == "/v1.43/containers/json"
    assert request.url.params["all"] == "true"
    assert json.loads(request.url.params["filters"]) == {
        "label": [
            "com.docker.compose.project=test-project",
            "com.docker.compose.service=whisper",
        ]
    }


@pytest.mark.parametrize(
    ("scenario", "outcome"),
    [
        ("running", ControlOutcome.RESTARTED),
        ("missing", ControlOutcome.ERROR),
        ("stopped", ControlOutcome.RESTARTED),
        ("timeout", ControlOutcome.TIMEOUT),
        ("rejected", ControlOutcome.ERROR),
    ],
)
def test_docker_restart(docker, scenario: str, outcome: ControlOutcome) -> None:
    def handler(request):
        if request.url.path.endswith("/containers/json"):
            return httpx.Response(200, json=[] if scenario == "missing" else [{"Id": "abc"}])
        if request.url.path.endswith("/abc/json"):
            return httpx.Response(200, json={"State": {"Running": scenario != "stopped"}})
        assert request.method == "POST"
        assert request.url.path == "/v1.43/containers/abc/restart"
        assert request.url.params["t"] == "10"
        if scenario == "timeout":
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(500 if scenario == "rejected" else 204)

    controller, requests = docker(handler)
    assert controller.restart("transcription").outcome == outcome
    assert any(r.method == "POST" for r in requests) is (scenario != "missing")
    # Every exit path, including exceptions, must release the per-service lock.
    assert controller._locks["transcription"].acquire(blocking=False)
    controller._locks["transcription"].release()


def test_busy_and_invalid_restart_do_not_contact_docker(docker) -> None:
    controller, requests = docker(lambda request: pytest.fail("unexpected Docker request"))
    with controller._locks["transcription"]:
        assert controller.restart("transcription").outcome == ControlOutcome.BUSY
    assert controller.restart("invalid_key").outcome == ControlOutcome.ERROR
    assert controller.inspect("invalid_key") == ServiceState.UNKNOWN
    assert requests == []


@pytest.mark.parametrize(
    ("state", "status", "expected"),
    [
        ({"Running": True}, 200, ServiceState.RUNNING),
        ({"Running": True, "Restarting": True}, 200, ServiceState.RESTARTING),
        ({"Running": False}, 200, ServiceState.STOPPED),
        ({}, 404, ServiceState.UNKNOWN),
        ({}, 500, ServiceState.UNKNOWN),
    ],
)
def test_docker_inspect(docker, state, status: int, expected: ServiceState) -> None:
    def handler(request):
        if request.url.path.endswith("/containers/json"):
            return httpx.Response(200, json=[{"Id": "abc"}])
        return httpx.Response(status, json={"State": state})

    controller, _ = docker(handler)
    assert controller.inspect("transcription") == expected


@pytest.mark.parametrize(
    ("mode", "exists", "accessible", "expected"),
    [
        (None, False, False, NoopController),
        ("docker", True, True, DockerController),
        ("docker", False, False, NoopController),
        ("docker", True, False, NoopController),
        ("launchd", False, False, NoopController),
    ],
)
def test_controller_factory(mode, exists: bool, accessible: bool, expected) -> None:
    settings = Settings(
        _env_file=None,
        voxint_service_control=mode,
        voxint_docker_socket="/test/docker.sock",
        voxint_compose_project="test-project",
    )
    with (
        patch("voxint.api.service_control.os.path.exists", return_value=exists) as path_exists,
        patch("voxint.api.service_control.os.access", return_value=accessible) as access,
    ):
        controller = get_controller(settings)
    assert isinstance(controller, expected)
    if mode == "docker":
        path_exists.assert_called_once_with("/test/docker.sock")
        if exists:
            access.assert_called_once_with("/test/docker.sock", os.R_OK | os.W_OK)
        else:
            access.assert_not_called()
    if isinstance(controller, DockerController):
        assert controller.controllable is True
        assert controller.backend_name == "docker"
        assert controller._socket_path == "/test/docker.sock"
        assert controller._compose_project == "test-project"


@pytest.fixture
def route_client():
    app = FastAPI()
    app.include_router(routes.router)
    app.state.settings = Settings(_env_file=None)
    app.state.csrf_secret = "service-control-test-secret"
    app.state.service_controller = Mock(spec=DockerController)
    app.state.loop_threads = []
    app.dependency_overrides[deps.require_onboarded] = lambda: None

    @app.middleware("http")
    async def record_loop_thread(request, call_next):
        app.state.loop_threads.append(threading.get_ident())
        return await call_next(request)

    app.dependency_overrides[deps._resolve_identity] = lambda: deps.AuthContext(
        user_id=None, username="tester", role="admin"
    )
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("outcome", list(ControlOutcome))
def test_restart_route(route_client, outcome: ControlOutcome) -> None:
    app = route_client.app
    restart_threads = []

    def restart(key):
        restart_threads.append(threading.get_ident())
        assert key == "transcription"
        return ControlResult(outcome, "failure <script>")

    app.state.service_controller.restart.side_effect = restart
    token = mint_csrf_token(app.state.csrf_secret, CSRF_SERVICE_CONTROL)
    response = route_client.post(
        "/settings/status/services/transcription/restart", data={"csrf_token": token}
    )
    assert response.status_code == 200
    assert 'id="service-transcription"' in response.text
    assert ("hx-trigger" in response.text) is (
        outcome in {ControlOutcome.RESTARTED, ControlOutcome.BUSY, ControlOutcome.TIMEOUT}
    )
    assert "<script>" not in response.text
    if outcome == ControlOutcome.ERROR:
        assert "&lt;script&gt;" in response.text
    assert restart_threads[0] != app.state.loop_threads[0]


def test_routes_reject_non_admin_and_invalid_csrf(route_client) -> None:
    app = route_client.app
    path = "/settings/status/services/transcription"
    assert route_client.post(path + "/restart", data={"csrf_token": "bad"}).status_code == 403
    app.dependency_overrides[deps._resolve_identity] = lambda: deps.AuthContext(
        user_id=None, username="viewer", role="viewer"
    )
    token = mint_csrf_token(app.state.csrf_secret, CSRF_SERVICE_CONTROL)
    assert route_client.get(path + "/row").status_code == 403
    assert route_client.post(path + "/restart", data={"csrf_token": token}).status_code == 403
    app.state.service_controller.restart.assert_not_called()
    app.state.service_controller.inspect.assert_not_called()


def test_routes_reject_unknown_service(route_client) -> None:
    token = mint_csrf_token(route_client.app.state.csrf_secret, CSRF_SERVICE_CONTROL)
    path = "/settings/status/services/invalid_key"
    assert route_client.get(path + "/row").status_code == 404
    assert route_client.post(path + "/restart", data={"csrf_token": token}).status_code == 404
    route_client.app.state.service_controller.restart.assert_not_called()
    route_client.app.state.service_controller.inspect.assert_not_called()


@pytest.mark.parametrize("healthy", [True, False])
def test_service_row_probes_health_and_controls_polling(route_client, monkeypatch, healthy) -> None:
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200 if healthy else 503, json={"status": "ok", "model_loaded": healthy}
        )

    # Patch the probe's client factory without replacing TestClient's own transport.
    real_client = httpx.Client
    monkeypatch.setattr(
        routes.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    controller = route_client.app.state.service_controller
    controller.controllable = True
    controller.inspect.return_value = ServiceState.RUNNING
    response = route_client.get("/settings/status/services/transcription/row")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert str(requests[0].url) == route_client.app.state.settings.asr_url.rstrip("/") + "/healthz"
    assert ("hx-trigger" in response.text) is (not healthy)
    assert 'name="csrf_token"' in response.text
    assert 'hx-post="/settings/status/services/transcription/restart"' in response.text
    controller.inspect.assert_called_once_with("transcription")


def test_component_control_metadata_is_limited_to_model_services() -> None:
    for controller in (
        NoopController("disabled", install_kind="docker"),
        DockerController("/test/docker.sock", "test-project"),
    ):
        rows = routes._build_components(
            [
                {"name": name, "state": "ready", "detail": "ready", "remediation": ""}
                for name in ("transcription", "diarization", "speaker embedding")
            ],
            controller,
        )
        models = [row for row in rows if row["is_model_service"]]
        assert {row["key"] for row in models} == SERVICE_KEYS
        for row in rows:
            if row["is_model_service"]:
                assert row["controllable"] is controller.controllable
                assert row["terminal_hint"] == controller.terminal_hint(row["key"])
            else:
                assert row["key"] is None
                assert row["controllable"] is False
                assert row["terminal_hint"] is None
