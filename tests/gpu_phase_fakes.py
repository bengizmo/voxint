"""Deterministic broker, clock and service controls shared by phase tests."""

import json
from datetime import UTC, datetime, timedelta

import httpx

from voxint.api.health_probe import ServiceHealth
from voxint.api.service_control import SERVICE_KEYS, ControlOutcome, ControlResult, ServiceState
from voxint.config import Settings


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime.now(UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: int = 30) -> None:
        self.now += timedelta(seconds=seconds)


class FakeGpuBroker:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.mode = "held"
        self.release_mode = "held"
        self.lease_id: str | None = None
        self.expiry: datetime | None = None
        self.explicit_release = False
        # Number of release calls answered 202 before the release completes.
        self.release_pending = 0
        self.calls: list[httpx.Request] = []
        self.transport = httpx.MockTransport(self)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if self.expiry and self.clock() >= self.expiry:
            self.lease_id = None
        mode = self.release_mode if request.url.path == "/release" else self.mode
        if mode == "down":
            raise httpx.ConnectError("offline", request=request)
        if mode == "5xx":
            return httpx.Response(503, json={})
        if mode == "malformed":
            return httpx.Response(200, content=b"not json")
        if request.url.path == "/status":
            return httpx.Response(
                200,
                json={
                    "state": "held" if self.lease_id else "free",
                    "holder": "voxint" if self.lease_id else None,
                    "lease_id": self.lease_id,
                    "expires_at": self.expiry.isoformat() if self.expiry else None,
                },
            )
        body = json.loads(request.content)
        if request.url.path == "/release":
            if self.release_pending > 0 and self.lease_id == body["lease_id"]:
                self.release_pending -= 1
                return httpx.Response(202, json={"state": "pending"})
            code = 200 if self.lease_id == body["lease_id"] else 404
            if code == 200:
                self.lease_id = None
                self.expiry = None
            return httpx.Response(code)
        if mode == "busy":
            return httpx.Response(409, json={"holder": "other", "retry_after_seconds": 90})
        if mode == "pending":
            return httpx.Response(202, json={"state": "pending"})
        if body["lease_id"] and body["lease_id"] != self.lease_id:
            return httpx.Response(404)
        self.lease_id = self.lease_id or "test-lease"
        self.expiry = (
            None if self.explicit_release else self.clock() + timedelta(seconds=body["ttl_seconds"])
        )
        return httpx.Response(
            200,
            json={
                "lease_id": self.lease_id,
                "expires_at": self.expiry.isoformat() if self.expiry else None,
            },
        )


class FakeController:
    controllable = True
    backend_name = "fake"

    def __init__(self) -> None:
        self.states = dict.fromkeys(SERVICE_KEYS, ServiceState.STOPPED)
        self.failures: dict[tuple[str, str], ControlOutcome] = {}
        self.calls: list[tuple[str, str]] = []

    def start(self, service_key: str) -> ControlResult:
        self.calls.append(("start", service_key))
        failure = self.failures.get(("start", service_key))
        if failure:
            return ControlResult(failure, "injected")
        old = self.states[service_key]
        self.states[service_key] = ServiceState.RUNNING
        return ControlResult(
            ControlOutcome.ALREADY_RUNNING
            if old == ServiceState.RUNNING
            else ControlOutcome.STARTED,
            "ok",
        )

    def stop(self, service_key: str) -> ControlResult:
        self.calls.append(("stop", service_key))
        failure = self.failures.get(("stop", service_key))
        if failure:
            return ControlResult(failure, "injected")
        old = self.states[service_key]
        self.states[service_key] = ServiceState.STOPPED
        return ControlResult(
            ControlOutcome.ALREADY_STOPPED
            if old == ServiceState.STOPPED
            else ControlOutcome.STOPPED,
            "ok",
        )

    def restart(self, service_key: str) -> ControlResult:
        self.stop(service_key)
        return self.start(service_key)

    def inspect(self, service_key: str) -> ServiceState:
        self.calls.append(("inspect", service_key))
        return self.states[service_key]

    def terminal_hint(self, service_key: str, action: str = "restart") -> str | None:
        return None


class FakeProbe:
    ready = True

    def __call__(self, settings: Settings) -> list[ServiceHealth]:
        return [
            ServiceHealth(k, "http://localhost", self.ready, "fake", 1)
            for k in sorted(SERVICE_KEYS)
        ]
