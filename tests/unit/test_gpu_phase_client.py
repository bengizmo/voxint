"""Broker contract parsing, failure containment and secret handling."""

import json

import httpx
import pytest

from tests.gpu_phase_fakes import FakeClock, FakeGpuBroker
from tests.unit.test_gpu_phase import phase_settings
from voxint.gpu_phase.client import LeaseClient


@pytest.mark.parametrize(
    "code,body,renew,kind",
    [
        (200, {"lease_id": "id"}, False, "held"),
        (200, {"lease_id": "id", "expires_at": None}, False, "held"),
        (200, {"lease_id": "id", "expires_at": "2026-10-01T00:00:00Z"}, True, "held"),
        (202, {"state": "pending"}, False, "pending"),
        (409, {"holder": "other"}, False, "busy"),
        (409, {"holder": "other", "retry_after_seconds": 40}, False, "busy"),
        (409, {}, True, "lost"),
        (404, {}, True, "lost"),
        (404, {}, False, "failed"),
        (503, {}, False, "failed"),
        (200, [], False, "failed"),
        (200, {"lease_id": 4}, False, "failed"),
        (200, {"lease_id": ""}, False, "failed"),
        (200, {"lease_id": "id", "expires_at": 5}, False, "failed"),
        (200, {"lease_id": "id", "expires_at": "2026-01-01"}, False, "failed"),
        (200, {"lease_id": "id", "expires_at": "bad"}, False, "failed"),
        (202, {"state": "wrong"}, False, "failed"),
        (409, {"holder": "other", "retry_after_seconds": True}, False, "failed"),
        (409, {"holder": "other", "retry_after_seconds": -1}, False, "failed"),
    ],
)
def test_acquire_responses(code, body, renew, kind):
    client = LeaseClient(
        phase_settings(), transport=httpx.MockTransport(lambda r: httpx.Response(code, json=body))
    )
    assert client.acquire("old" if renew else None).kind == kind


@pytest.mark.parametrize("token", ["", "private-token"])
def test_headers_ttl_and_renew(token):
    clock = FakeClock()
    broker = FakeGpuBroker(clock)
    with httpx.Client(transport=broker.transport) as http:
        client = LeaseClient(phase_settings(gpu_lease_token=token, gpu_phase_tick_seconds=60), http)
        held = client.acquire()
        assert held.kind == "held"
        assert client.acquire(held.lease_id).kind == "held"
        body = json.loads(broker.calls[-1].content)
        assert body == {"holder": "voxint", "ttl_seconds": 240, "lease_id": held.lease_id}
        assert broker.calls[-1].headers.get("Authorization") == (
            f"Bearer {token}" if token else None
        )
        assert broker.calls[-1].extensions["timeout"]["read"] == 10
        assert client.status().kind == "held"
        assert client.release(held.lease_id).kind == "released"
        assert client.release(held.lease_id).kind == "released"
        assert client.status().kind == "free"


@pytest.mark.parametrize("mode", ["timeout", "down", "malformed", "5xx"])
def test_failures_never_echo_secrets(mode, caplog):
    secret = "private-broker-token"

    def handler(request):
        if mode == "timeout":
            raise httpx.ReadTimeout(secret)
        if mode == "down":
            raise httpx.ConnectError(secret)
        return httpx.Response(503 if mode == "5xx" else 200, content=secret.encode())

    client = LeaseClient(
        phase_settings(gpu_lease_token=secret), transport=httpx.MockTransport(handler)
    )
    results = [client.acquire(), client.status(), client.release("lease")]
    # Release 200 intentionally has no body contract.
    assert results[0].kind == results[1].kind == "failed"
    assert results[2].kind == ("released" if mode == "malformed" else "failed")
    assert secret not in repr(results) + caplog.text


@pytest.mark.parametrize(
    "body,kind",
    [
        ({"state": "free", "holder": None, "lease_id": None}, "free"),
        ({"state": "free"}, "failed"),
        ({"state": "held", "holder": "voxint", "lease_id": "id"}, "held"),
        ({"state": "held", "holder": 4, "lease_id": "id"}, "failed"),
        ({"state": "free", "holder": "someone"}, "failed"),
        ({"state": "wrong"}, "failed"),
        ([], "failed"),
    ],
)
def test_status_shapes(body, kind):
    client = LeaseClient(
        phase_settings(), transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))
    )
    assert client.status().kind == kind


def test_expiry_frees_lease():
    clock = FakeClock()
    broker = FakeGpuBroker(clock)
    client = LeaseClient(phase_settings(), transport=broker.transport)
    held = client.acquire()
    clock.advance(121)
    assert client.status().kind == "free"
    assert client.acquire(held.lease_id).kind == "lost"
    broker.explicit_release = True
    assert client.acquire().expires_at is None


@pytest.mark.parametrize("code", [404, 409])
def test_renew_lost_without_json(code):
    client = LeaseClient(
        phase_settings(), transport=httpx.MockTransport(lambda r: httpx.Response(code))
    )
    assert client.acquire("existing").kind == "lost"
