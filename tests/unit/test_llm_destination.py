"""LLM SSRF policy permits local inference without permitting metadata access."""

import socket
from unittest.mock import Mock

import pytest

from voxint.api.setup_wizard import SetupValidationError, normalize_llm_base_url
from voxint.clients import llm_destination
from voxint.clients.llm import HttpLLMClient, LLMError
from voxint.clients.llm_destination import validate_llm_destination
from voxint.config import DEFAULT_LLM_BASE_URL


@pytest.mark.parametrize(
    "host",
    [
        "169.254.169.254",
        "169.254.0.1",
        "169.254.255.255",
        "[fe80::1]",
        "[febf::1]",
        "100.100.100.200",
        "[fd00:ec2::254]",
        "[::ffff:169.254.169.254]",
        "[64:ff9b::a9fe:a9fe]",
    ],
)
@pytest.mark.parametrize("multi_user", [False, True])
def test_llm_blocks_metadata_and_link_local(host, multi_user):
    with pytest.raises(ValueError, match="blocked"):
        validate_llm_destination(f"http://{host}/v1", multi_user=multi_user)


@pytest.mark.parametrize("host", ["10.0.0.1", "172.16.0.1", "172.31.255.254", "192.168.1.1"])
@pytest.mark.parametrize("multi_user", [False, True])
def test_llm_allows_lan(host, multi_user):
    validate_llm_destination(f"http://{host}:11434/v1", multi_user=multi_user)


@pytest.mark.parametrize("host", ["127.0.0.1", "127.2.3.4", "[::1]", "[::ffff:127.0.0.1]"])
def test_llm_loopback_depends_on_mode(host):
    url = f"http://{host}/v1"
    validate_llm_destination(url, multi_user=False)
    with pytest.raises(ValueError, match="loopback"):
        validate_llm_destination(url, multi_user=True)


def resolve_to(monkeypatch, *addresses):
    resolver = Mock(
        return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 80)) for ip in addresses]
    )
    monkeypatch.setattr(llm_destination.socket, "getaddrinfo", resolver)
    return resolver


def test_llm_checks_every_dns_answer_at_setup_and_runtime(monkeypatch):
    resolver = resolve_to(monkeypatch, "10.0.0.1")
    url = "http://ollama.example/v1"
    assert normalize_llm_base_url(url, multi_user=False) == url
    resolver.return_value.append(
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 80))
    )
    with pytest.raises(SetupValidationError, match="link-local"):
        normalize_llm_base_url(url, multi_user=False)
    constructor = Mock(side_effect=AssertionError("must reject before client construction"))
    monkeypatch.setattr("voxint.clients.llm.httpx.Client", constructor)
    with pytest.raises(LLMError, match="link-local"):
        HttpLLMClient(url, "model", "secret", 5)
    constructor.assert_not_called()


def test_llm_default_endpoint_exempt_from_loopback_restriction(monkeypatch):
    resolve_to(monkeypatch, "127.0.0.1")
    validate_llm_destination(DEFAULT_LLM_BASE_URL, multi_user=True)
    with pytest.raises(SetupValidationError, match="loopback"):
        normalize_llm_base_url("http://localhost/v1", multi_user=True)


@pytest.mark.parametrize("answer", [[], "error"])
def test_llm_dns_failures_reject(monkeypatch, answer):
    resolver = resolve_to(monkeypatch)
    if answer == "error":
        resolver.side_effect = socket.gaierror("DNS unavailable")
    with pytest.raises(ValueError, match="resolv"):
        validate_llm_destination("https://unknown.example/v1", multi_user=False)


def test_llm_runtime_uses_multi_user_setting(monkeypatch):
    monkeypatch.setattr(llm_destination, "get_settings", lambda: Mock(voxint_multi_user=True))
    with pytest.raises(LLMError, match="loopback"):
        HttpLLMClient("http://127.0.0.1/v1", "model", "", 5)
