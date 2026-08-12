"""Worker-side SSRF gate: re-resolve an ingest host and reject non-public addresses.

This is the SECOND, authoritative SSRF guard for URL ingestion. The FIRST is the
string-level :func:`voxint.ingest.service.validate_ingest_url`, which gates row
creation but deliberately does **not** resolve DNS — a hostname that looks public
at submit time can rebind to a private address before the worker downloads it. So
the worker re-resolves the host at download time (this module) and refuses to hand
yt-dlp a host that resolves to *any* non-public address.

:func:`ip_is_public` is the single per-address policy, shared with
``validate_ingest_url``'s IP-literal check, so the literal gate and the resolved
gate can never diverge on what "public" means. The module is stdlib-only (no DB,
no project deps) so both the read path and the worker can import it cheaply.

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


def ip_is_public(ip: _IPAddress) -> bool:
    """The single "is this address safe to fetch" policy.

    Public = globally routable AND not multicast. ``is_global`` is already False
    for loopback / private / link-local / reserved / unspecified, and it is True
    for multicast (224/4, ff00::/8), so multicast is excluded explicitly. Shared
    by the string-level literal check (``validate_ingest_url``) and the
    resolved-host check here, so the two gates apply an identical policy.
    """
    return ip.is_global and not ip.is_multicast


def assert_host_resolves_public(
    host: str, *, resolver: Resolver = socket.getaddrinfo
) -> None:
    """Resolve ``host`` (A + AAAA) and raise :class:`HostNotPublicError` if ANY
    resolved address is non-public.

    Rejects on the FIRST non-public address rather than requiring every answer to
    be private, so a host with even one private answer (a DNS-rebinding or
    split-horizon trick) is refused. This also catches the IPv4-in-IPv6 embeddings
    (deprecated ``::a.b.c.d``, RFC 6052 NAT64) that ``is_global`` mis-classifies
    as literals: once resolved they surface as their real addresses.

    Fail-closed: an unresolvable host or an answer that does not parse as an IP is
    itself terminal (never proof of a public address), so both raise
    ``HostNotPublicError`` — the worker refuses to hand an unverifiable host to
    yt-dlp. ``resolver`` defaults to ``socket.getaddrinfo`` and is injected in
    tests so CI never touches real DNS. Messages name only ``host``.
    """
    try:
        infos = resolver(host, None)  # both A and AAAA; the port is irrelevant
    except OSError:
        # Unresolvable / transient DNS error. `from None`: a getaddrinfo error
        # names only the host (never the URL), but suppress the cause anyway to
        # match the module's "the message carries only the host" contract.
        raise HostNotPublicError(
            f"ingest host {host!r} could not be resolved"
        ) from None
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
