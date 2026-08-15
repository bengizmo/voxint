"""Worker-side SSRF gate: re-resolve an ingest host and reject non-public addresses.

This is the SECOND, authoritative SSRF guard for URL ingestion. The FIRST is the
string-level :func:`voxint.ingest.service.validate_ingest_url`, which gates row
creation but deliberately does **not** resolve DNS — a hostname that looks public
at submit time can rebind to a private address before the worker downloads it. So
the worker re-resolves the host at download time (this module) and refuses to hand
yt-dlp a host that resolves to *any* non-public address.

:func:`ip_is_public` is the single per-address policy, shared with
``validate_ingest_url``'s IP-literal check, so the literal gate and the resolved
gate can never diverge on what "public" means. It does **not** trust the stdlib
``is_global`` alone: that flag mis-classifies deprecated IPv6 **site-local**
(``fec0::/10``) and several **IPv4-in-IPv6 embeddings** (deprecated
``::a.b.c.d``, RFC 6052 NAT64 ``64:ff9b::/96``) as global, so a literal such as
``[64:ff9b::127.0.0.1]`` would otherwise pass — and ``getaddrinfo`` echoes an IP
*literal* straight back, so re-resolution alone does not expose the embedded
IPv4. :func:`ip_is_public` therefore unwraps the embedded IPv4 and judges the
real target. The module is stdlib-only (no DB, no project deps) so both the read
path and the worker can import it cheaply.

This is **not** a sandbox. yt-dlp re-resolves the host *independently* when it
connects, so a name that rebinds between our check and yt-dlp's fetch, or an HTTP
redirect / extractor-constructed URL to a private address, is beyond a userland
check's reach — closing those needs network policy (an egress firewall or a route
with no path to RFC1918 / link-local). See ``docs/architecture.md``.
"""

import ipaddress
import re
import socket
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

_IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

# The string-level URL gate is an http(s)-only capability; anything else (file:,
# data:, ftp:, a bare scheme-relative //host) is rejected at the string level.
_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})
# A pasted URL that runs to kilobytes is almost certainly hostile or malformed;
# 2048 is the de-facto interoperable URL length ceiling.
MAX_URL_BYTES = 2048
# A well-formed URL carries no raw whitespace (spaces/tabs/newlines must be
# percent-encoded); an unencoded whitespace char is a splitting/smuggling smell.
_URL_WHITESPACE = re.compile(r"\s")
# Control characters (incl. NUL) are never valid anywhere in a URL.
_URL_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")

# IPv6 prefixes carrying an IPv4 payload that ``is_global`` mis-judges as global.
# RFC 6052 NAT64 well-known prefix (64:ff9b::/96) is legitimately globally
# routable per spec, but nothing stops an attacker embedding a non-public IPv4 in
# it; the deprecated IPv4-compatible ``::/96`` (``::a.b.c.d``) has no stdlib
# special-case at all. Both must be judged by the embedded IPv4, not the wrapper.
_NAT64_WKP = ipaddress.IPv6Network("64:ff9b::/96")
_IPV4_COMPAT = ipaddress.IPv6Network("::/96")

# A resolver maps a host to getaddrinfo's 5-tuples ``(family, type, proto,
# canonname, sockaddr)``; only ``sockaddr[0]`` (the address string) is read.
# Injected in tests so no real DNS lookup happens in CI.
Resolver = Callable[..., Iterable[tuple[Any, Any, Any, Any, "tuple[Any, ...]"]]]


class HostNotPublicError(Exception):
    """A host is refused because it resolves to (or could not be verified as) a
    globally-routable address.

    Deterministic: the ACQUIRE stage translates it to an ``AcquisitionError`` so
    the run parks FAILED @ acquire for a manual Requeue. The message names only
    the **host** (never the full URL — the host is not a secret, but a URL's query
    can carry a signed token).
    """


class UrlPolicyError(Exception):
    """A URL is refused by the string-level HTTP(S) URL policy.

    Messages describe the violated rule and never echo the URL (or any part of
    it beyond, at most, nothing — a query string can carry a signed token).
    Callers that need a context prefix re-raise with one (``validate_ingest_url``
    prepends ``"ingest "``), keeping this module's messages context-neutral.
    """


@dataclass(frozen=True)
class HttpUrl:
    """The typed result of :func:`parse_http_url`.

    ``url`` is the whitespace-trimmed original (safe to store/fetch verbatim);
    ``host`` is the lowercased authority host with any trailing DNS root dot
    stripped; ``ip`` is the parsed address when the host is an IP literal, else
    ``None`` (a DNS name — whether it resolves publicly is deliberately not
    judged at the string level; see :func:`resolve_public_addresses`).
    """

    url: str
    scheme: str
    host: str
    port: int | None
    ip: _IPAddress | None


def parse_http_url(url: str, *, max_bytes: int = MAX_URL_BYTES) -> HttpUrl:
    """Validate a URL at the string level and return its typed parts.

    This is the SINGLE string-level gate for every outbound-fetch capability
    (URL ingestion and web research). It enforces the shape a fetcher is
    permitted to touch: an absolute http/https URL with a plain hostname, no
    embedded credentials, no whitespace/control characters, under the length
    ceiling, and — when the host is an IP *literal* — a globally routable
    address per :func:`ip_is_public` (loopback/private/link-local/reserved/
    multicast literals are refused, including the IPv4-in-IPv6 embeddings and
    site-local that ``is_global`` alone mis-classifies).

    It deliberately does **not** resolve DNS: a name that looks public now can
    rebind before a worker fetches it, so the authoritative "resolves to a
    public address" check belongs at fetch time (:func:`resolve_public_addresses`).
    Only ``localhost`` is refused by name. Error messages never echo the URL,
    so a signed/secret query string cannot leak into an error body or logs.
    """
    candidate = (url or "").strip()
    if not candidate:
        raise UrlPolicyError("URL is empty")
    try:
        url_bytes = len(candidate.encode("utf-8"))
    except UnicodeEncodeError as exc:  # e.g. an unpaired surrogate — uphold the typed contract
        raise UrlPolicyError("URL is not valid UTF-8") from exc
    if url_bytes > max_bytes:
        raise UrlPolicyError(f"URL exceeds {max_bytes} bytes")
    if _URL_WHITESPACE.search(candidate) or _URL_CONTROL_CHARS.search(candidate):
        raise UrlPolicyError("URL contains whitespace or control characters")
    if "\\" in candidate:
        # urlsplit keeps backslashes in the authority, but browsers/yt-dlp may
        # treat "\" as "/" — a parser split we refuse rather than try to model.
        raise UrlPolicyError("URL must not contain a backslash")
    try:
        parts = urlsplit(candidate)
        # .port is lazily parsed; touch it so a bad port (":abc"/out-of-range)
        # surfaces here rather than as an obscure failure deeper in a fetcher.
        port = parts.port
    except ValueError as exc:  # malformed IPv6 literal, un-castable/out-of-range port
        raise UrlPolicyError("URL is malformed") from exc
    if parts.scheme not in _ALLOWED_URL_SCHEMES:
        raise UrlPolicyError("URL must be an absolute http/https URL")
    if parts.username is not None or parts.password is not None:
        raise UrlPolicyError("URL must not embed credentials")
    host = parts.hostname
    if not host:
        raise UrlPolicyError("URL has no host")
    # A trailing DNS root dot ("localhost.", "127.0.0.1.") resolves identically to
    # the un-dotted form, so strip it before the policy checks — otherwise a lone
    # dot side-steps both the localhost denylist and the IP-literal parse.
    host = host.rstrip(".")
    if not host:
        raise UrlPolicyError("URL has no host")
    bracketed = "[" in parts.netloc  # the authority was an IPv6/IPvFuture literal
    if host == "localhost" or host.endswith(".localhost"):
        # urlsplit lowercases .hostname, so a plain case check is exhaustive.
        raise UrlPolicyError("URL host is not permitted")
    ip: _IPAddress | None
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        if bracketed:
            # A bracketed authority that is not a valid IPv6 literal (e.g. an
            # IPvFuture "[v1.foo]") is ambiguous across parsers — refuse it rather
            # than let it fall through to the DNS-name branch.
            raise UrlPolicyError("URL has an invalid IPv6 host") from None
        ip = None  # a DNS name — public-address check deferred to fetch time
    if ip is not None and not ip_is_public(ip):
        # Loopback/private/link-local/reserved/unspecified/multicast literals, plus
        # site-local and the IPv4-in-IPv6 embeddings (::a.b.c.d, NAT64) that
        # is_global alone mis-judges — ip_is_public is the single per-address policy.
        raise UrlPolicyError("URL host is not permitted")
    return HttpUrl(url=candidate, scheme=parts.scheme, host=host, port=port, ip=ip)


def _embedded_ipv4(ip: ipaddress.IPv6Address) -> "ipaddress.IPv4Address | None":
    """Return the IPv4 address an IPv6 address embeds, or ``None``.

    Covers the stdlib-recognised forms (IPv4-mapped ``::ffff:0:0/96``, 6to4
    ``2002::/16``, Teredo ``2001::/32`` — its *client* endpoint) plus the two the
    stdlib does not special-case: RFC 6052 NAT64 ``64:ff9b::/96`` and the
    deprecated IPv4-compatible ``::/96`` (``::a.b.c.d``, excluding ``::`` and
    ``::1`` which are unspecified/loopback and already non-global).
    """
    if ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    if ip.sixtofour is not None:
        return ip.sixtofour
    if ip.teredo is not None:
        return ip.teredo[1]  # the tunnelled client IPv4 (the real endpoint)
    if ip in _NAT64_WKP:
        return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    if ip in _IPV4_COMPAT and int(ip) not in (0, 1):
        return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return None


def ip_is_public(ip: _IPAddress) -> bool:
    """The single "is this address safe to fetch" policy.

    Public = globally routable AND not multicast. ``is_global`` is already False
    for loopback / private / link-local / reserved / unspecified, and True for
    multicast (224/4, ff00::/8), so multicast is excluded explicitly. Shared by the
    string-level literal check (``validate_ingest_url``) and the resolved-host
    check here, so the two gates apply an identical policy.

    ``is_global`` alone is not enough for IPv6: it mis-classifies deprecated
    **site-local** (``fec0::/10``) as global, and it judges an IPv4-in-IPv6
    **embedding** by its wrapper prefix rather than the IPv4 it carries — so
    ``[64:ff9b::127.0.0.1]`` (NAT64→loopback) and ``[::169.254.169.254]`` would
    pass. Site-local is rejected outright; an embedded IPv4 is unwrapped and judged
    recursively, so the gate never trusts a public-looking wrapper around a private
    target. This only *tightens*: an embedding of a public IPv4 still falls through
    to the wrapper's own ``is_global`` verdict.
    """
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.is_site_local:  # fec0::/10, deprecated but never public
            return False
        embedded = _embedded_ipv4(ip)
        if embedded is not None and not ip_is_public(embedded):
            return False
    return ip.is_global and not ip.is_multicast


def resolve_public_addresses(
    host: str, *, resolver: Resolver = socket.getaddrinfo
) -> tuple[_IPAddress, ...]:
    """Resolve ``host`` (A + AAAA) and return its vetted addresses, raising
    :class:`HostNotPublicError` if ANY resolved address is non-public.

    Rejects on the FIRST non-public address rather than requiring every answer to
    be private, so a host with even one private answer (a DNS-rebinding or
    split-horizon trick) is refused — the ENTIRE answer set is discarded, never
    filtered down to its public members. The per-address verdict is
    :func:`ip_is_public`, which already unwraps IPv4-in-IPv6 embeddings and
    rejects site-local — so a name resolving to ``64:ff9b::127.0.0.1`` or
    ``fec0::1`` is caught here too, not just literal IPv4/IPv6.

    Returns the vetted addresses (deduplicated, answer order preserved) so a
    caller that connects can PIN one — connecting to a returned address rather
    than re-resolving closes the check-then-connect rebinding window for its own
    fetch path.

    Fail-closed: an unresolvable host, a resolver that returns **no** addresses, or
    an answer that does not parse as an IP is each terminal (never proof of a
    public address), so all raise ``HostNotPublicError`` — a caller refuses to
    fetch from an unverifiable host. ``resolver`` defaults to
    ``socket.getaddrinfo`` and is injected in tests so CI never touches real DNS.
    Messages name only ``host``, context-neutrally; callers needing a context
    prefix re-raise with one (:func:`assert_host_resolves_public` prepends
    ``"ingest "``).
    """
    try:
        infos = list(resolver(host, None))  # both A and AAAA; the port is irrelevant
    except OSError:
        # Unresolvable / transient DNS error. `from None`: a getaddrinfo error
        # names only the host (never the URL), but suppress the cause anyway to
        # match the module's "the message carries only the host" contract.
        raise HostNotPublicError(f"host {host!r} could not be resolved") from None
    if not infos:
        # A resolver that returns an empty list without raising (a custom/future
        # resolver, a platform quirk) is not proof of a public address — fail closed
        # rather than sail through the (skipped) loop as a success.
        raise HostNotPublicError(f"host {host!r} resolved to no addresses")
    vetted: list[_IPAddress] = []
    for info in infos:
        address = info[4][0]  # sockaddr[0] is the address string for A and AAAA
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            raise HostNotPublicError(
                f"host {host!r} resolved to an unparseable address"
            ) from None
        if not ip_is_public(ip):
            raise HostNotPublicError(f"host {host!r} resolves to a non-public address")
        if ip not in vetted:
            vetted.append(ip)
    return tuple(vetted)


def assert_host_resolves_public(
    host: str, *, resolver: Resolver = socket.getaddrinfo
) -> None:
    """Ingest-facing wrapper over :func:`resolve_public_addresses` — same
    fail-closed policy, addresses discarded, messages prefixed ``"ingest "``
    (preserving the historical ingest error strings byte-for-byte)."""
    try:
        resolve_public_addresses(host, resolver=resolver)
    except HostNotPublicError as exc:
        raise HostNotPublicError(f"ingest {exc}") from None
