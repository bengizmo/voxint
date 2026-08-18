"""`voxint doctor`'s checks against injected transports — no live deps.

Hard checks (postgres/redis/models) drive the exit code; advisory checks
(HF token, LLM) are reported but never do. Details must never leak a URL,
token, or raw exception string.
"""

import httpx
from sqlalchemy import create_engine

from voxint.config import Settings
from voxint.diagnostics import (
    CheckResult,
    check_database,
    check_hf_token,
    check_llm,
    check_models,
    check_redis,
    check_state,
    exit_code,
    run_diagnostics,
)

_ASR_PORT, _DIARIZER_PORT, _EMBEDDER_PORT = 8022, 8024, 8021


def _settings(**over: object) -> Settings:
    return Settings(voxint_user="u", voxint_password="p", **over)


def _healthz(device: str) -> httpx.Response:
    return httpx.Response(
        200, json={"status": "ok", "model": "m", "model_loaded": True, "device": device}
    )


def _http(handler: object) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


# ---- exit-code policy -------------------------------------------------------


def test_exit_code_zero_when_hard_pass_even_if_advisory_fails() -> None:
    results = [
        CheckResult("postgres", True, True, "connected"),
        CheckResult("hugging face token", False, False, "rejected (401)"),
    ]
    assert exit_code(results) == 0


def test_exit_code_one_when_a_hard_dep_is_down() -> None:
    results = [CheckResult("redis", False, True, "unreachable (ConnectionError)")]
    assert exit_code(results) == 1


# ---- database ---------------------------------------------------------------


def test_check_database_ok_on_working_engine() -> None:
    engine = create_engine("sqlite://")  # in-memory; SELECT 1 succeeds
    result = check_database(engine)
    assert result.ok and result.hard and result.detail == "connected"


def test_check_database_reports_failure_type_only() -> None:
    class _BadEngine:
        def connect(self) -> object:
            raise OSError("host=secret.internal password=hunter2")

    result = check_database(_BadEngine())  # type: ignore[arg-type]
    assert result.ok is False and result.hard is True
    assert "OSError" in result.detail
    assert "hunter2" not in result.detail and "secret.internal" not in result.detail


# ---- redis ------------------------------------------------------------------


def test_check_redis_ok_with_injected_client() -> None:
    class _Ping:
        def ping(self) -> bool:
            return True

    result = check_redis("redis://ignored", client=_Ping())
    assert result.ok and result.hard and result.detail == "reachable"


def test_check_redis_failure_hides_url_and_message() -> None:
    class _Down:
        def ping(self) -> bool:
            raise ConnectionError("redis://user:pw@host:6379 refused")

    result = check_redis("redis://user:pw@host", client=_Down())
    assert result.ok is False and result.hard is True
    assert "ConnectionError" in result.detail
    assert "pw" not in result.detail and "host" not in result.detail


def test_check_redis_malformed_url_is_hard_failure_not_traceback() -> None:
    # A bad DSN raises ValueError inside Redis.from_url (construction), which must
    # normalize to a hard FAIL — never escape doctor as a traceback.
    result = check_redis("not-a-redis-url")
    assert result.ok is False and result.hard is True
    assert "not-a-redis-url" not in result.detail


# ---- models -----------------------------------------------------------------


def test_check_models_surfaces_device_and_marks_hard() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return {
            _ASR_PORT: _healthz("rocm"),
            _DIARIZER_PORT: _healthz("cpu"),
            _EMBEDDER_PORT: _healthz("cpu"),
        }[request.url.port or 0]

    results = check_models(_settings(), client=_http(handler))
    assert all(r.hard for r in results)
    assert results[0].ok and results[0].detail == "ready (rocm)"
    assert results[1].detail == "ready (cpu)"


def test_check_models_down_service_is_hard_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if (request.url.port or 0) == _ASR_PORT:
            raise httpx.ConnectError("refused")
        return _healthz("cpu")

    results = {r.name: r for r in check_models(_settings(), client=_http(handler))}
    assert results["transcription"].ok is False
    assert results["transcription"].hard is True


# ---- HF token (advisory) ----------------------------------------------------


def test_check_hf_token_absent_is_ok_advisory() -> None:
    result = check_hf_token(None, client=_http(lambda r: httpx.Response(200)))
    assert result.ok is True and result.hard is False
    assert "not set" in result.detail


def test_check_hf_token_valid_shows_name() -> None:
    client = _http(lambda r: httpx.Response(200, json={"name": "ben"}))
    result = check_hf_token("hf_x", client=client)
    assert result.ok is True and result.hard is False and result.detail == "valid (ben)"


def test_check_hf_token_rejected_is_advisory_failure() -> None:
    client = _http(lambda r: httpx.Response(401))
    result = check_hf_token("hf_bad", client=client)
    assert result.ok is False and result.hard is False and result.detail == "rejected (401)"


def test_check_hf_token_non_dict_200_body_does_not_crash() -> None:
    # A valid-JSON but non-object 200 (a captive portal / proxy) must not raise
    # AttributeError from .get — it resolves to "valid" without a name.
    client = _http(lambda r: httpx.Response(200, json=["not", "a", "dict"]))
    result = check_hf_token("hf_x", client=client)
    assert result.ok is True and result.hard is False and result.detail == "valid"


# ---- LLM (advisory) ---------------------------------------------------------


def test_check_llm_none_when_disabled() -> None:
    client = _http(lambda r: httpx.Response(200))
    assert (
        check_llm(
            enabled=False, base_url="http://localhost:8000/v1", api_key="", client=client
        )
        is None
    )


def test_check_llm_reachable_on_any_http_answer() -> None:
    client = _http(lambda r: httpx.Response(404))  # host answered ⇒ reachable
    result = check_llm(
        enabled=True, base_url="http://localhost:8000/v1", api_key="sk-x", client=client
    )
    assert result is not None
    assert result.ok is True and result.hard is False
    assert "reachable" in result.detail


def test_check_llm_transport_error_is_advisory_failure() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    result = check_llm(
        enabled=True, base_url="http://localhost:8000/v1", api_key="", client=_http(handler)
    )
    assert result is not None
    assert result.ok is False and result.hard is False


def test_check_llm_invalid_url_does_not_crash() -> None:
    # httpx.InvalidURL is not an httpx.HTTPError; a malformed base_url must
    # resolve to an advisory failure, not abort the whole doctor run.
    def handler(_r: httpx.Request) -> httpx.Response:
        raise AssertionError("request should not be attempted on an invalid url")

    result = check_llm(
        enabled=True, base_url="http://[::1", api_key="", client=_http(handler)
    )
    assert result is not None
    assert result.ok is False and result.hard is False and result.detail == "invalid url"


# ---- run_diagnostics orchestration -----------------------------------------


def test_run_diagnostics_collects_hard_and_advisory() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "huggingface" in request.url.host:
            return httpx.Response(200, json={"name": "ben"})
        return _healthz("cpu")

    class _Ping:
        def ping(self) -> bool:
            return True

    results = run_diagnostics(
        _settings(llm_enabled=False),
        create_engine("sqlite://"),
        hf_token="hf_x",
        http_client=_http(handler),
        redis_client=_Ping(),
    )
    names = [r.name for r in results]
    assert "postgres" in names and "redis" in names and "transcription" in names
    assert "hugging face token" in names
    assert "llm endpoint" not in names  # disabled → not checked
    assert exit_code(results) == 0


def test_run_diagnostics_include_hf_token_false_omits_hf_and_skips_whoami() -> None:
    # The wizard SERVICES step (issue #61) passes include_hf_token=False: the HF row
    # must be gone AND no whoami call may be made (the handler asserts it is never hit).
    def handler(request: httpx.Request) -> httpx.Response:
        assert "huggingface" not in request.url.host, "HF whoami must not be called"
        return _healthz("cpu")

    class _Ping:
        def ping(self) -> bool:
            return True

    results = run_diagnostics(
        _settings(llm_enabled=False),
        create_engine("sqlite://"),
        hf_token="hf_x",
        http_client=_http(handler),
        redis_client=_Ping(),
        include_hf_token=False,
    )
    names = [r.name for r in results]
    assert "hugging face token" not in names
    assert "postgres" in names and "redis" in names and "transcription" in names


# ---- check_state (three honest display states, issue #61) ------------------


def test_check_state_ready_when_ok() -> None:
    assert check_state(CheckResult("postgres", True, True, "connected")) == "ready"
    # An advisory pass is still "ready" — an ok result is ready regardless of hardness.
    assert check_state(CheckResult("llm endpoint", True, False, "reachable")) == "ready"


def test_check_state_failed_when_hard_and_down() -> None:
    assert check_state(CheckResult("redis", False, True, "unreachable")) == "failed"


def test_check_state_unverified_when_advisory_and_down() -> None:
    assert check_state(CheckResult("llm endpoint", False, False, "invalid url")) == "unverified"
