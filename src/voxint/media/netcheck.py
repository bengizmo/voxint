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
import socket
from collections.abc import Callable, Iterable
from typing import Any

_IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

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


def assert_host_resolves_public(
    host: str, *, resolver: Resolver = socket.getaddrinfo
) -> None:
    """Resolve ``host`` (A + AAAA) and raise :class:`HostNotPublicError` if ANY
    resolved address is non-public.

    Rejects on the FIRST non-public address rather than requiring every answer to
    be private, so a host with even one private answer (a DNS-rebinding or
    split-horizon trick) is refused. The per-address verdict is :func:`ip_is_public`,
    which already unwraps IPv4-in-IPv6 embeddings and rejects site-local — so a
    name resolving to ``64:ff9b::127.0.0.1`` or ``fec0::1`` is caught here too, not
    just literal IPv4/IPv6.

    Fail-closed: an unresolvable host, a resolver that returns **no** addresses, or
    an answer that does not parse as an IP is each terminal (never proof of a
    public address), so all raise ``HostNotPublicError`` — the worker refuses to
    hand an unverifiable host to yt-dlp. ``resolver`` defaults to
    ``socket.getaddrinfo`` and is injected in tests so CI never touches real DNS.
    Messages name only ``host``.
    """
    try:
        infos = list(resolver(host, None))  # both A and AAAA; the port is irrelevant
    except OSError:
        # Unresolvable / transient DNS error. `from None`: a getaddrinfo error
        # names only the host (never the URL), but suppress the cause anyway to
        # match the module's "the message carries only the host" contract.
        raise HostNotPublicError(
            f"ingest host {host!r} could not be resolved"
        ) from None
    if not infos:
        # A resolver that returns an empty list without raising (a custom/future
        # resolver, a platform quirk) is not proof of a public address — fail closed
        # rather than sail through the (skipped) loop as a success.
        raise HostNotPublicError(f"ingest host {host!r} resolved to no addresses")
    for info in infos:
        address = info[4][0]  # sockaddr[0] is the address string for A and AAAA
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            raise HostNotPublicError(
                f"ingest host {host!r} resolved to an unparseable address"
            ) from None
        if not ip_is_public(ip):
            raise HostNotPublicError(
                f"ingest host {host!r} resolves to a non-public address"
            )
