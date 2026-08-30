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
    check_llm_bundled,
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
            enabled=False,
            configured=True,
            base_url="http://localhost:8000/v1",
            api_key="",
            client=client,
        )
        is None
    )


def test_check_llm_ready_on_2xx() -> None:
    client = _http(lambda r: httpx.Response(200, json={"data": []}))
    result = check_llm(
        enabled=True,
        configured=True,
        base_url="http://localhost:8000/v1",
        api_key="sk-x",
        client=client,
    )
    assert result is not None
    assert result.ok is True and result.hard is False
    assert result.detail == "reachable (HTTP 200)"


def test_check_llm_rejected_non_2xx_is_advisory_miss() -> None:
    # A wrong key (401) or wrong path (404) means the host is reachable but rejected the
    # probe — a real enhancement call would be rejected the same way, so this is an
    # advisory miss (→ "unverified" in the wizard), never a green "ready" (issue #61).
    for status in (401, 404, 500):
        client = _http(lambda r, s=status: httpx.Response(s))
        result = check_llm(
            enabled=True,
            configured=True,
            base_url="http://localhost:8000/v1",
            api_key="sk-x",
            client=client,
        )
        assert result is not None
        assert result.ok is False and result.hard is False
        assert result.detail == f"rejected (HTTP {status})"


def test_check_llm_unexpected_exception_does_not_raise() -> None:
    # The "never raises into the caller" contract must hold for ANY error building or
    # sending the request, not only httpx.HTTPError — e.g. a non-ASCII env api_key that
    # httpx can't encode into the Authorization header raises UnicodeEncodeError. Any
    # such error must normalize to a redacted advisory miss (the wizard GET stays 200).
    def handler(_r: httpx.Request) -> httpx.Response:
        raise UnicodeEncodeError("ascii", "clé", 2, 3, "ordinal not in range")

    result = check_llm(
        enabled=True,
        configured=True,
        base_url="http://localhost:8000/v1",
        api_key="sk-x",
        client=_http(handler),
    )
    assert result is not None
    assert result.ok is False and result.hard is False
    assert result.detail == "unreachable (UnicodeEncodeError)"


def test_check_llm_transport_error_is_advisory_failure() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    result = check_llm(
        enabled=True,
        configured=True,
        base_url="http://localhost:8000/v1",
        api_key="",
        client=_http(handler),
    )
    assert result is not None
    assert result.ok is False and result.hard is False


def test_check_llm_invalid_url_does_not_crash() -> None:
    # httpx.InvalidURL is not an httpx.HTTPError; a malformed base_url must
    # resolve to an advisory failure, not abort the whole doctor run.
    def handler(_r: httpx.Request) -> httpx.Response:
        raise AssertionError("request should not be attempted on an invalid url")

    result = check_llm(
        enabled=True, configured=True, base_url="http://[::1", api_key="", client=_http(handler)
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


# ---- #316: lane-aware LLM checks -------------------------------------------


def test_check_llm_not_configured_reports_ok_without_probing() -> None:
    # Enabled but no deliberate BYO endpoint (the untouched install default with
    # no key): probing would 401 against an endpoint the operator never chose,
    # so the check reports an ok-state "not configured" and NEVER touches the
    # network (the handler proves no request is attempted).
    def handler(_r: httpx.Request) -> httpx.Response:
        raise AssertionError("an unconfigured BYO endpoint must not be probed")

    result = check_llm(
        enabled=True,
        configured=False,
        base_url="https://api.openai.com/v1",
        api_key="",
        client=_http(handler),
    )
    assert result is not None
    assert result.ok is True and result.hard is False
    assert result.detail == "not configured"


def test_check_llm_bundled_none_when_inactive() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        raise AssertionError("an inactive bundle must not be probed")

    assert (
        check_llm_bundled(active=False, base_url="http://127.0.0.1:8090/v1", client=_http(handler))
        is None
    )


def test_check_llm_bundled_state_matrix() -> None:
    base = "http://127.0.0.1:8090/v1"
    ok = check_llm_bundled(active=True, base_url=base, client=_http(lambda r: httpx.Response(200)))
    assert ok is not None and ok.ok is True and ok.hard is False
    assert ok.name == "llm bundled" and ok.detail == "reachable (HTTP 200)"

    rejected = check_llm_bundled(
        active=True, base_url=base, client=_http(lambda r: httpx.Response(500))
    )
    assert rejected is not None and rejected.ok is False and rejected.hard is False
    assert rejected.detail == "rejected (HTTP 500)"

    def refuse(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    down = check_llm_bundled(active=True, base_url=base, client=_http(refuse))
    assert down is not None and down.ok is False and down.hard is False
    assert down.detail == "unreachable (ConnectError)"
    assert base not in down.detail  # never echo the URL

    bad = check_llm_bundled(active=True, base_url="http://[::1", client=_http(refuse))
    assert bad is not None and bad.ok is False and bad.detail == "invalid url"


def test_check_llm_bundled_probe_is_keyless_models_get(monkeypatch) -> None:
    seen: list[tuple[str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers.get("authorization")))
        return httpx.Response(200)

    check_llm_bundled(active=True, base_url="http://127.0.0.1:8090/v1", client=_http(handler))
    assert seen == [("http://127.0.0.1:8090/v1/models", None)]


def test_run_diagnostics_bundled_only_install_is_all_green() -> None:
    """The walkthrough scenario (#316): bundle active, BYO at the untouched
    default with no key. The bundled endpoint is probed, the BYO endpoint is
    NOT (the handler rejects any request to it), and nothing is a miss. The
    sqlite engine has no app_settings table, so this also exercises the
    row-read-failure fallback to env-only predicate resolution."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert "openai" not in (request.url.host or ""), (
            "the unconfigured BYO default endpoint must not be probed"
        )
        if (request.url.port or 0) == 8090:
            return httpx.Response(200, json={"data": []})
        return _healthz("cpu")

    class _Ping:
        def ping(self) -> bool:
            return True

    results = run_diagnostics(
        _settings(
            llm_enabled=True,
            llm_bundled_enabled=True,
            llm_bundled_base_url="http://127.0.0.1:8090/v1",
            llm_bundled_model="qwen",
        ),
        create_engine("sqlite://"),
        hf_token=None,
        http_client=_http(handler),
        redis_client=_Ping(),
        include_hf_token=False,
    )
    by_name = {r.name: r for r in results}
    assert by_name["llm bundled"].ok is True
    assert by_name["llm bundled"].detail == "reachable (HTTP 200)"
    assert by_name["llm endpoint"].ok is True
    assert by_name["llm endpoint"].detail == "not configured"
    assert exit_code(results) == 0


def test_run_diagnostics_bundled_inactive_has_no_bundled_row() -> None:
    class _Ping:
        def ping(self) -> bool:
            return True

    results = run_diagnostics(
        _settings(llm_enabled=False),
        create_engine("sqlite://"),
        hf_token=None,
        http_client=_http(lambda r: _healthz("cpu")),
        redis_client=_Ping(),
        include_hf_token=False,
    )
    names = [r.name for r in results]
    assert "llm bundled" not in names
    assert "llm endpoint" not in names
