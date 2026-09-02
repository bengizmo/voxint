"""GPU-service readiness probe against httpx.MockTransport — no services involved.

Exercises every normalized outcome (ready, degraded, other-HTTP, timeout,
transport error, malformed 2xx, explicit model_loaded=false) plus the mixed case
and the own-client create/close branch, and asserts a probe never raises.
"""

import socket

import httpx
import pytest

from voxint.api.health_probe import ServiceHealth, probe_services
from voxint.config import Settings

# The default service ports (config.py) — the probe hits "{url}/healthz", so the
# handler dispatches on the request's port to give each service its own response.
_ASR_PORT = 8022
_DIARIZER_PORT = 8024
_EMBEDDER_PORT = 8021


def _settings() -> Settings:
    # Loopback + default password is a valid dev config; the probe only reads the
    # three service URLs, which keep their defaults here.
    return Settings(voxint_user="u", voxint_password="p")


def _client(by_port: dict[int, object]) -> httpx.Client:
    """An httpx.Client whose MockTransport answers per destination port.

    A value that is an ``Exception`` is raised (transport failure); an
    ``httpx.Response`` is returned; the ``/healthz`` suffix is asserted so a
    wrong path would fail loudly.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/healthz"
        outcome = by_port[request.url.port or 0]
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, httpx.Response)
        return outcome

    return httpx.Client(transport=httpx.MockTransport(handler))


def _healthy(service: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "status": "ok",
            "service": service,
            "model": f"{service}-model",
            "model_loaded": True,
        },
    )


def _by_name(results: list[ServiceHealth]) -> dict[str, ServiceHealth]:
    return {r.name: r for r in results}


def test_all_services_up() -> None:
    client = _client(
        {
            _ASR_PORT: _healthy("whisper"),
            _DIARIZER_PORT: _healthy("pyannote"),
            _EMBEDDER_PORT: _healthy("titanet"),
        }
    )
    results = probe_services(_settings(), client=client)
    assert [r.name for r in results] == ["transcription", "diarization", "speaker embedding"]
    for r in results:
        assert r.up is True
        assert r.detail == "ready"
        assert r.latency_ms is not None and r.latency_ms >= 0.0


def test_device_surfaced_when_service_reports_it() -> None:
    # The additive `device` field (docs/gpu-contracts.md) flows through to
    # ServiceHealth so `voxint doctor` can show the compute tier.
    ready_rocm = httpx.Response(
        200, json={"status": "ok", "model": "m", "model_loaded": True, "device": "rocm"}
    )
    client = _client(
        {_ASR_PORT: ready_rocm, _DIARIZER_PORT: _healthy("p"), _EMBEDDER_PORT: _healthy("t")}
    )
    results = _by_name(probe_services(_settings(), client=client))
    assert results["transcription"].device == "rocm"
    # A ready service that omits `device` reports None (not an error).
    assert results["diarization"].up is True
    assert results["diarization"].device is None


def test_embedder_health_exposes_embedding_space() -> None:
    ready = httpx.Response(
        200,
        json={
            "status": "ok",
            "model": "titanet",
            "model_loaded": True,
            "embedding_space": "titanet-large-v2",
        },
    )
    client = _client(
        {
            _ASR_PORT: _healthy("whisper"),
            _DIARIZER_PORT: _healthy("pyannote"),
            _EMBEDDER_PORT: ready,
        }
    )
    results = _by_name(probe_services(_settings(), client=client))
    assert results["speaker embedding"].embedding_space == "titanet-large-v2"


def test_non_string_device_is_ignored() -> None:
    weird = httpx.Response(
        200, json={"status": "ok", "model": "m", "model_loaded": True, "device": 123}
    )
    client = _client(
        {_ASR_PORT: weird, _DIARIZER_PORT: _healthy("p"), _EMBEDDER_PORT: _healthy("t")}
    )
    asr = _by_name(probe_services(_settings(), client=client))["transcription"]
    assert asr.up is True
    assert asr.device is None


def test_degraded_503_is_down_but_distinguished() -> None:
    degraded = httpx.Response(503, json={"status": "degraded", "model": None})
    client = _client(
        {
            _ASR_PORT: degraded,
            _DIARIZER_PORT: _healthy("pyannote"),
            _EMBEDDER_PORT: _healthy("titanet"),
        }
    )
    results = _by_name(probe_services(_settings(), client=client))
    asr = results["transcription"]
    assert asr.up is False
    assert asr.detail == "degraded (model not loaded)"
    # A 503 is a completed round-trip, so latency is still reported.
    assert asr.latency_ms is not None
    assert results["diarization"].up is True


def test_other_http_error_is_down() -> None:
    client = _client(
        {
            _ASR_PORT: httpx.Response(500),
            _DIARIZER_PORT: _healthy("p"),
            _EMBEDDER_PORT: _healthy("t"),
        }
    )
    asr = _by_name(probe_services(_settings(), client=client))["transcription"]
    assert asr.up is False
    assert asr.detail == "HTTP 500"


def test_timeout_is_down_with_no_latency() -> None:
    client = _client(
        {
            _ASR_PORT: httpx.TimeoutException("read timed out"),
            _DIARIZER_PORT: _healthy("p"),
            _EMBEDDER_PORT: _healthy("t"),
        }
    )
    asr = _by_name(probe_services(_settings(), client=client))["transcription"]
    assert asr.up is False
    assert asr.detail == "timeout"
    assert asr.latency_ms is None


def test_transport_error_is_unreachable() -> None:
    client = _client(
        {
            _ASR_PORT: httpx.ConnectError("connection refused"),
            _DIARIZER_PORT: _healthy("p"),
            _EMBEDDER_PORT: _healthy("t"),
        }
    )
    asr = _by_name(probe_services(_settings(), client=client))["transcription"]
    assert asr.up is False
    assert asr.detail == "unreachable"
    assert asr.latency_ms is None


def test_malformed_2xx_body_is_down() -> None:
    bad = httpx.Response(200, content=b"<html>not json</html>")
    client = _client(
        {_ASR_PORT: bad, _DIARIZER_PORT: _healthy("p"), _EMBEDDER_PORT: _healthy("t")}
    )
    asr = _by_name(probe_services(_settings(), client=client))["transcription"]
    assert asr.up is False
    assert asr.detail == "invalid response"


def test_2xx_with_model_not_loaded_is_down() -> None:
    unloaded = httpx.Response(200, json={"status": "ok", "model_loaded": False})
    client = _client(
        {_ASR_PORT: unloaded, _DIARIZER_PORT: _healthy("p"), _EMBEDDER_PORT: _healthy("t")}
    )
    asr = _by_name(probe_services(_settings(), client=client))["transcription"]
    assert asr.up is False
    assert asr.detail == "model not loaded"


def test_mixed_outcomes_all_reported() -> None:
    client = _client(
        {
            _ASR_PORT: _healthy("whisper"),
            _DIARIZER_PORT: httpx.Response(503, json={"status": "degraded"}),
            _EMBEDDER_PORT: httpx.TimeoutException("slow"),
        }
    )
    results = _by_name(probe_services(_settings(), client=client))
    assert len(results) == 3
    assert results["transcription"].up is True
    assert results["diarization"].up is False
    assert results["speaker embedding"].detail == "timeout"


def test_creates_and_closes_own_client_when_none_given() -> None:
    # Point every service at a closed loopback port so the real (own) client's
    # connect fails fast and deterministically — covering the client=None branch
    # (create + close) without a live server or a slow timeout.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    url = f"http://127.0.0.1:{port}"
    settings = Settings(
        voxint_user="u",
        voxint_password="p",
        asr_url=url,
        diarizer_url=url,
        embedder_url=url,
        health_probe_timeout_seconds=1.0,
    )
    results = probe_services(settings)  # no client injected → own client path
    assert len(results) == 3
    assert all(r.up is False for r in results)
    assert all(r.detail == "unreachable" for r in results)


def test_malformed_url_is_reported_not_raised() -> None:
    # A bad configured URL raises httpx.InvalidURL, which is NOT an httpx.HTTPError;
    # the probe must catch it so "never raises" holds. URL parsing fails before any
    # network, so this is deterministic without a server.
    bad = "http://localhost:not-a-port"
    settings = Settings(
        voxint_user="u",
        voxint_password="p",
        asr_url=bad,
        diarizer_url=bad,
        embedder_url=bad,
    )
    results = probe_services(settings)
    assert len(results) == 3
    for r in results:
        assert r.up is False
        assert r.detail == "invalid url"
        assert r.latency_ms is None


def test_2xx_without_ready_shape_is_not_ready() -> None:
    # A 200 that lacks status="ok" + model_loaded=true is not "ready" — a foreign
    # server answering 200 on the port must not read as available.
    client = _client(
        {
            _ASR_PORT: httpx.Response(200, json={"status": "starting"}),
            _DIARIZER_PORT: _healthy("p"),
            _EMBEDDER_PORT: _healthy("t"),
        }
    )
    asr = _by_name(probe_services(_settings(), client=client))["transcription"]
    assert asr.up is False
    assert asr.detail == "not ready"


def test_2xx_non_object_body_is_invalid() -> None:
    client = _client(
        {
            _ASR_PORT: httpx.Response(200, json=["ok"]),
            _DIARIZER_PORT: _healthy("p"),
            _EMBEDDER_PORT: _healthy("t"),
        }
    )
    asr = _by_name(probe_services(_settings(), client=client))["transcription"]
    assert asr.up is False
    assert asr.detail == "invalid response"


def test_own_client_closed_even_on_unexpected_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # The client=None path must close its own client even if probing raises an
    # unexpected (non-httpx) error — the finally block owns the close, and the bug
    # is not masked (it still propagates).
    closed = {"n": 0}

    class _SpyClient:
        def __init__(self, **_: object) -> None:
            pass

        def get(self, _url: str) -> httpx.Response:
            raise RuntimeError("boom")

        def close(self) -> None:
            closed["n"] += 1

    monkeypatch.setattr("voxint.api.health_probe.httpx.Client", _SpyClient)
    with pytest.raises(RuntimeError):
        probe_services(_settings())
    assert closed["n"] == 1
