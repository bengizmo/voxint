"""Worker-side SSRF gate: the shared IP policy and the DNS re-resolution check.

Pure/DI-only — the resolver is injected so no test touches real DNS. Non-public
fixtures use the IETF documentation ranges (TEST-NET-1/2/3, 2001:db8::/32) and the
well-known special literals (loopback / link-local / multicast / unspecified / IPv6
ULA), never an RFC1918 / internal-network literal.
"""

import ipaddress
import socket

import pytest

from voxint.media.netcheck import (
    HostNotPublicError,
    Resolver,
    UrlPolicyError,
    assert_host_resolves_public,
    ip_is_public,
    parse_http_url,
    resolve_public_addresses,
)


@pytest.mark.parametrize(
    "address",
    [
        "8.8.8.8",  # a globally-routable IPv4 literal
        "1.1.1.1",
        "93.184.216.34",
        "2606:4700:4700::1111",  # a globally-routable IPv6 literal
        "64:ff9b::8.8.8.8",  # NAT64 wrapping a PUBLIC IPv4 stays public (only tightens)
    ],
)
def test_ip_is_public_true_for_global_unicast(address: str) -> None:
    assert ip_is_public(ipaddress.ip_address(address)) is True


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",  # loopback
        "0.0.0.0",  # unspecified
        "169.254.0.1",  # link-local
        "192.0.2.1",  # TEST-NET-1 (reserved, stands in for RFC1918)
        "198.51.100.7",  # TEST-NET-2
        "203.0.113.9",  # TEST-NET-3
        "169.254.169.254",  # cloud metadata endpoint (classic SSRF target)
        "224.0.0.1",  # IPv4 local multicast (is_global True → is_multicast rejects)
        "239.255.255.250",  # IPv4 SSDP multicast
        "::1",  # IPv6 loopback
        "fe80::1",  # IPv6 link-local
        "fc00::1",  # IPv6 unique-local (private)
        "ff02::1",  # IPv6 multicast
        "2001:db8::1",  # IPv6 documentation range (reserved)
        # is_global mis-judges these as global — the regression this fix closes:
        "fec0::1",  # deprecated IPv6 site-local
        "::127.0.0.1",  # deprecated IPv4-compatible embedding loopback
        "::192.0.2.1",  # IPv4-compatible embedding a non-global IPv4 (TEST-NET-1)
        "::ffff:127.0.0.1",  # IPv4-mapped loopback
        "2002:7f00:1::",  # 6to4 embedding loopback
        "64:ff9b::127.0.0.1",  # RFC 6052 NAT64 embedding loopback
        "64:ff9b::169.254.169.254",  # NAT64 embedding the cloud metadata endpoint
    ],
)
def test_ip_is_public_false_for_non_public(address: str) -> None:
    assert ip_is_public(ipaddress.ip_address(address)) is False


def _resolver_returning(*addresses: str) -> Resolver:
    def resolve(host: str, *args: object, **kwargs: object) -> list[
        tuple[int, int, int, str, tuple[str, int]]
    ]:
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (addr, 0))
            for addr in addresses
        ]

    return resolve


def test_public_host_passes() -> None:
    # A host that resolves to a global address returns cleanly (no raise).
    assert_host_resolves_public(
        "cdn.example.com", resolver=_resolver_returning("93.184.216.34")
    )


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "192.0.2.1",
        "169.254.169.254",
        "::1",
        "fe80::1",
        "ff02::1",
        # A name resolving to an embedded/site-local IPv6 is caught too (the same
        # ip_is_public), not just literal IPv4/IPv6.
        "64:ff9b::127.0.0.1",
        "fec0::1",
        "::127.0.0.1",
    ],
)
def test_non_public_resolution_is_rejected(address: str) -> None:
    with pytest.raises(HostNotPublicError, match="non-public"):
        assert_host_resolves_public(
            "rebinding.example.com", resolver=_resolver_returning(address)
        )


def test_empty_resolution_fails_closed() -> None:
    # A resolver that returns no addresses (without raising) must fail closed, not
    # sail through the skipped loop as a success.
    def _empty(host: str, *args: object, **kwargs: object) -> list[object]:
        return []

    with pytest.raises(HostNotPublicError, match="no addresses"):
        assert_host_resolves_public("empty.example.com", resolver=_empty)


def test_rejects_on_first_non_public_among_many() -> None:
    # A host answering with a public AND a private address is refused.
    with pytest.raises(HostNotPublicError, match="non-public"):
        assert_host_resolves_public(
            "split.example.com",
            resolver=_resolver_returning("93.184.216.34", "198.51.100.7"),
        )


def test_unresolvable_host_fails_closed() -> None:
    def _boom(host: str, *args: object, **kwargs: object) -> list[object]:
        raise socket.gaierror("Name or service not known")

    with pytest.raises(HostNotPublicError, match="could not be resolved"):
        assert_host_resolves_public("nx.example.com", resolver=_boom)


def test_unparseable_answer_fails_closed() -> None:
    with pytest.raises(HostNotPublicError, match="unparseable"):
        assert_host_resolves_public(
            "weird.example.com", resolver=_resolver_returning("not-an-ip")
        )


def test_error_message_names_only_the_host() -> None:
    # The message carries the host (not a secret) but no address detail beyond it.
    with pytest.raises(HostNotPublicError) as exc:
        assert_host_resolves_public(
            "host.example.com", resolver=_resolver_returning("192.0.2.5")
        )
    assert "host.example.com" in str(exc.value)


# --- resolve_public_addresses (the pinning-capable core the ingest gate wraps) ---


def test_resolve_returns_vetted_addresses_in_answer_order() -> None:
    vetted = resolve_public_addresses(
        "cdn.example.com",
        resolver=_resolver_returning("93.184.216.34", "2606:4700:4700::1111"),
    )
    assert vetted == (
        ipaddress.ip_address("93.184.216.34"),
        ipaddress.ip_address("2606:4700:4700::1111"),
    )


def test_resolve_deduplicates_repeated_answers() -> None:
    vetted = resolve_public_addresses(
        "cdn.example.com",
        resolver=_resolver_returning("93.184.216.34", "93.184.216.34", "1.1.1.1"),
    )
    assert vetted == (
        ipaddress.ip_address("93.184.216.34"),
        ipaddress.ip_address("1.1.1.1"),
    )


def test_resolve_discards_entire_answer_set_on_one_private_member() -> None:
    # Never filtered down to the public members — the whole set is refused.
    with pytest.raises(HostNotPublicError, match="non-public"):
        resolve_public_addresses(
            "split.example.com",
            resolver=_resolver_returning("93.184.216.34", "169.254.169.254"),
        )


def test_resolve_messages_are_context_neutral_ingest_wrapper_prefixes() -> None:
    # The shared core's message carries no "ingest" prefix (web research reuses
    # it); the ingest-facing wrapper preserves the historical string byte-for-byte.
    resolver = _resolver_returning("192.0.2.5")
    with pytest.raises(HostNotPublicError) as core:
        resolve_public_addresses("host.example.com", resolver=resolver)
    with pytest.raises(HostNotPublicError) as wrapped:
        assert_host_resolves_public("host.example.com", resolver=resolver)
    assert not str(core.value).startswith("ingest ")
    assert str(wrapped.value) == f"ingest {core.value}"
    assert str(wrapped.value) == (
        "ingest host 'host.example.com' resolves to a non-public address"
    )


# --- parse_http_url (the shared string-level gate the ingest validator wraps) ---


@pytest.mark.parametrize(
    "url",
    [
        "",
        "example.com/watch",  # no scheme
        "ftp://example.com/f.mp3",  # non-http(s) scheme
        "http://user:pass@example.com/v",  # embedded credentials
        "http://exa mple.com/v",  # internal whitespace
        "http://example.com/\x00",  # control char
        "http://[::1/v",  # malformed IPv6 literal
        "http://localhost/v",  # localhost by name
        "http://localhost./v",  # trailing root dot must not side-step the denylist
        "http://127.0.0.1/v",  # loopback literal
        "http://169.254.169.254/latest/meta-data",  # cloud metadata SSRF
        "http://[64:ff9b::127.0.0.1]/v",  # NAT64 embedding loopback
        "http://127.0.0.1\\/v",  # backslash parser split
        "http://[v1.foo]/v",  # bracketed IPvFuture
        "https://" + "a" * 2100 + ".com/v",  # over the length ceiling
    ],
)
def test_parse_http_url_rejects_unsafe(url: str) -> None:
    with pytest.raises(UrlPolicyError):
        parse_http_url(url)


def test_parse_http_url_messages_have_no_ingest_prefix() -> None:
    with pytest.raises(UrlPolicyError) as exc:
        parse_http_url("")
    assert str(exc.value) == "URL is empty"


def test_parse_http_url_typed_result_for_dns_name() -> None:
    parsed = parse_http_url("  https://Sub.Example.co.uk:8443/a/b?c=d#frag\n")
    assert parsed.url == "https://Sub.Example.co.uk:8443/a/b?c=d#frag"  # trimmed only
    assert parsed.scheme == "https"
    assert parsed.host == "sub.example.co.uk"  # urlsplit lowercases the host
    assert parsed.port == 8443
    assert parsed.ip is None  # a DNS name — resolution deferred to fetch time


def test_parse_http_url_typed_result_for_ip_literal() -> None:
    parsed = parse_http_url("https://8.8.8.8/v")
    assert parsed.host == "8.8.8.8"
    assert parsed.port is None
    assert parsed.ip == ipaddress.ip_address("8.8.8.8")


def test_parse_http_url_strips_trailing_root_dot_from_host() -> None:
    assert parse_http_url("https://example.com./v").host == "example.com"


def test_parse_http_url_custom_max_bytes() -> None:
    with pytest.raises(UrlPolicyError, match="exceeds 64 bytes"):
        parse_http_url("https://example.com/" + "a" * 64, max_bytes=64)
