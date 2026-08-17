"""Unit tests for the filtering egress proxy's pure request-parsing and
vet-decision logic. DI-only: the resolver is injected so no test touches real
DNS, and the per-address policy under test is the SAME ``netcheck.ip_is_public``
the worker gate uses (so these also assert the two can't drift). The full
socket loop (pin → connect → tunnel / refuse) is covered in
``tests/integration/test_egress_proxy_loop.py``.
"""

import socket

import pytest

from voxint.media.egress_proxy import (
    ProxyError,
    _parse_absolute_target,
    _parse_port,
    _parse_request_line,
    _rewrite_head,
    _split_authority,
    _vetted_addresses,
)
from voxint.media.netcheck import Resolver

_Answer = list[tuple[int, int, int, str, tuple[str, int]]]

# Real globally-routable IPs for the "public" cases (the injected connector
# redirects the actual socket to a local stub, so no real egress happens). Note
# the codebase convention: TEST-NET ranges (192.0.2/24, 198.51.100/24,
# 203.0.113/24) are NON-public per ``is_global`` and are used as refusal fixtures.
_PUBLIC_A = "93.184.216.34"
_PUBLIC_B = "1.1.1.1"


def _resolver_returning(*addresses: str) -> Resolver:
    def resolve(host: str, *args: object, **kwargs: object) -> _Answer:
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (addr, 0)) for addr in addresses
        ]

    return resolve


def _resolver_raising(host: str, *args: object, **kwargs: object) -> _Answer:
    raise OSError("name resolution failed")


def _resolver_empty(host: str, *args: object, **kwargs: object) -> _Answer:
    return []


# -- request line -----------------------------------------------------------


def test_parse_request_line_splits_three_parts() -> None:
    assert _parse_request_line("CONNECT host:443 HTTP/1.1") == (
        "CONNECT",
        "host:443",
        "HTTP/1.1",
    )


@pytest.mark.parametrize(
    "line",
    ["", "GET", "GET /only-two", "a b c d", "GET  /double-space HTTP/1.1", "GET /x "],
)
def test_parse_request_line_rejects_malformed(line: str) -> None:
    with pytest.raises(ProxyError) as exc:
        _parse_request_line(line)
    assert exc.value.status.startswith("400")


# -- strict port parsing ----------------------------------------------------


@pytest.mark.parametrize(("port_s", "expected"), [("443", 443), ("1", 1), ("65535", 65535)])
def test_parse_port_accepts_valid(port_s: str, expected: int) -> None:
    assert _parse_port(port_s) == expected


@pytest.mark.parametrize("port_s", ["", "0", "abc", "443a", "65536", "70000", "9" * 4400])
def test_parse_port_rejects_invalid(port_s: str) -> None:
    with pytest.raises(ProxyError) as exc:
        _parse_port(port_s)
    assert exc.value.status.startswith("400")


@pytest.mark.parametrize(
    "authority",
    [
        "host:abc",  # non-numeric port
        "host:443a",  # trailing junk
        "host:0",  # port 0
        "host:99999",  # out of range
        "[::1]:abc",  # bad bracketed port
        "[::1]junk",  # garbage after the bracket
        "[::1",  # unclosed bracket
    ],
)
def test_split_authority_rejects_bad_ports_strictly(authority: str) -> None:
    with pytest.raises(ProxyError) as exc:
        _split_authority(authority, default_port=443)
    assert exc.value.status.startswith("400")


# -- absolute-form target parsing -------------------------------------------


def test_parse_absolute_target_builds_origin_from_parts() -> None:
    host, port, origin = _parse_absolute_target("http://host.example:8080/a/b?q=1#frag")
    assert host == "host.example"
    assert port == 8080
    assert origin == "/a/b?q=1"  # fragment dropped, query kept


def test_parse_absolute_target_defaults_port_and_root() -> None:
    host, port, origin = _parse_absolute_target("http://host.example")
    assert (host, port, origin) == ("host.example", 80, "/")


def test_parse_absolute_target_strips_trailing_dot() -> None:
    host, _, _ = _parse_absolute_target("http://host.example./x")
    assert host == "host.example"


@pytest.mark.parametrize(
    "target",
    [
        "https://host.example/",  # https must use CONNECT, not cleartext absolute-form
        "http://user:pass@host.example/",  # embedded credentials
        "http://host.example:99999/",  # out-of-range port -> clean 400, not ValueError
        "http://[::1/",  # malformed IPv6 literal -> clean 400
        "ftp://host.example/",  # non-http scheme
        "http:///path",  # no host
    ],
)
def test_parse_absolute_target_rejects(target: str) -> None:
    with pytest.raises(ProxyError) as exc:
        _parse_absolute_target(target)
    assert exc.value.status.startswith("400")


# -- CONNECT authority ------------------------------------------------------


@pytest.mark.parametrize(
    ("authority", "expected"),
    [
        ("host.example:8443", ("host.example", 8443)),
        ("host.example", ("host.example", 443)),  # default port
        ("host.example.:443", ("host.example", 443)),  # trailing DNS root dot stripped
        ("[2606:4700:4700::1111]:443", ("2606:4700:4700::1111", 443)),
        ("[2606:4700:4700::1111]", ("2606:4700:4700::1111", 443)),
    ],
)
def test_split_authority(authority: str, expected: tuple[str, int]) -> None:
    assert _split_authority(authority, default_port=443) == expected


@pytest.mark.parametrize("authority", ["", ":443", "[bad"])
def test_split_authority_rejects_malformed(authority: str) -> None:
    with pytest.raises(ProxyError):
        _split_authority(authority, default_port=443)


# -- vet decision (the SSRF gate, reused) -----------------------------------


def test_vetted_addresses_allows_public() -> None:
    assert _vetted_addresses(
        "cdn.example.com", _resolver_returning(_PUBLIC_A)
    ) == [_PUBLIC_A]


def test_vetted_addresses_preserves_order_and_dedupes() -> None:
    assert _vetted_addresses(
        "dual.example.com",
        _resolver_returning(_PUBLIC_A, _PUBLIC_A, _PUBLIC_B),
    ) == [_PUBLIC_A, _PUBLIC_B]


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",  # loopback
        "192.0.2.5",  # TEST-NET-1 (RFC 5737 documentation range, non-public)
        "198.51.100.5",  # TEST-NET-2 (non-public)
        "169.254.169.254",  # link-local / cloud metadata
        "::1",  # IPv6 loopback
        "fec0::1",  # IPv6 site-local (is_global mis-judges)
        "64:ff9b::127.0.0.1",  # NAT64 embedding loopback
    ],
)
def test_vetted_addresses_refuses_non_public(address: str) -> None:
    # ANY non-public resolved address refuses the whole host, fail-closed.
    with pytest.raises(ProxyError) as exc:
        _vetted_addresses("evil.example.com", _resolver_returning(_PUBLIC_A, address))
    assert exc.value.status.startswith("403")


def test_vetted_addresses_fails_closed_on_unresolvable() -> None:
    with pytest.raises(ProxyError) as exc:
        _vetted_addresses("nx.example.com", _resolver_raising)
    assert exc.value.status.startswith("403")


def test_vetted_addresses_fails_closed_on_empty_answer() -> None:
    with pytest.raises(ProxyError) as exc:
        _vetted_addresses("empty.example.com", _resolver_empty)
    assert exc.value.status.startswith("403")


def test_vetted_addresses_refuses_missing_host() -> None:
    with pytest.raises(ProxyError):
        _vetted_addresses("", _resolver_returning("203.0.113.5"))


def test_vetted_addresses_message_names_only_host() -> None:
    # The refusal must not echo a full URL (whose query can carry a token).
    with pytest.raises(ProxyError) as exc:
        _vetted_addresses("secret.example.com", _resolver_returning("198.51.100.1"))
    assert "secret.example.com" in str(exc.value)
    assert "http" not in str(exc.value).lower()


# -- head rewrite (absolute-form → origin-form) -----------------------------


def test_rewrite_head_produces_origin_form_and_synthesizes_host() -> None:
    head = _rewrite_head("GET", "/path?q=1", "host.example", 80, ["Accept: */*"])
    text = head.decode("iso-8859-1")
    assert text.startswith("GET /path?q=1 HTTP/1.1\r\n")
    assert "Host: host.example\r\n" in text
    assert "Connection: close\r\n" in text  # forced — no keep-alive through the tunnel
    assert "Accept: */*\r\n" in text
    assert text.endswith("\r\n\r\n")


def test_rewrite_head_replaces_client_host_with_vetted_authority() -> None:
    # A client Host that disagrees with the vetted host (domain fronting) must be
    # discarded and replaced with exactly one Host from the vetted authority.
    head = _rewrite_head(
        "GET", "/", "host.example", 80, ["Host: internal.evil", "Host: second.evil"]
    )
    text = head.decode("iso-8859-1")
    assert text.count("Host:") == 1
    assert "Host: host.example\r\n" in text
    assert "evil" not in text


def test_rewrite_head_strips_hop_by_hop_and_connection_nominated() -> None:
    head = _rewrite_head(
        "GET",
        "/",
        "host.example",
        80,
        [
            "Proxy-Connection: keep-alive",
            "Proxy-Authorization: Basic x",
            "Keep-Alive: timeout=5",
            "Transfer-Encoding: chunked",
            "Upgrade: h2c",
            "Connection: keep-alive, X-Custom-Hop",
            "X-Custom-Hop: secret",  # named by Connection → hop-by-hop, must be stripped
            "Accept: */*",  # ordinary header survives
        ],
    )
    text = head.decode("iso-8859-1")
    for banned in (
        "Proxy-Connection", "Proxy-Authorization", "Keep-Alive",
        "Transfer-Encoding", "Upgrade", "X-Custom-Hop",
    ):
        assert banned not in text, banned
    # the client's own Connection header is dropped; ours is the only one
    assert text.count("Connection:") == 1
    assert "Connection: close\r\n" in text
    assert "Accept: */*\r\n" in text


def test_rewrite_head_brackets_ipv6_host() -> None:
    head = _rewrite_head("GET", "/", "2606:4700:4700::1111", 8080, ["Accept: */*"])
    text = head.decode("iso-8859-1")
    assert "Host: [2606:4700:4700::1111]:8080\r\n" in text


def test_rewrite_head_synthesizes_host_default_port_bare() -> None:
    head = _rewrite_head("GET", "/", "host.example", 80, ["Accept: */*"])
    text = head.decode("iso-8859-1")
    assert "Host: host.example\r\n" in text
