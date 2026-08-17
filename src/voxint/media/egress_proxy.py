"""A tiny, fail-closed filtering forward proxy for restricted yt-dlp egress.

This is the runnable half of the "network policy, not a userland check" answer to
the URL-ingestion SSRF residual (see ``docs/architecture.md`` and
:mod:`voxint.media.netcheck`). The two SSRF gates in ``netcheck`` re-resolve and
vet a host, but they hand yt-dlp the original URL and yt-dlp re-resolves the host
*independently* when it connects, follows HTTP redirects, and constructs extractor
URLs — so a DNS rebind between our check and yt-dlp's fetch, or a redirect to a
private address, is beyond a userland check's reach. yt-dlp-as-subprocess offers
no per-connection pin hook; the only in-product egress choke point is its
always-passed ``--proxy``.

So the ``compose.ytdlp-egress.yaml`` overlay routes that ``--proxy`` at this
process, which re-implements the resolved-host gate *at the connection boundary*:
every CONNECT / absolute-form request is resolved once, every resolved address is
vetted through the SAME :func:`voxint.media.netcheck.ip_is_public` policy the
worker gate uses (so the two can never drift), and the tunnel is opened to the
**vetted IP itself** — pinning it, which closes the rebind window that yt-dlp's own
re-resolution reopens, and catches redirect/extractor destinations too (each new
CONNECT/absolute-form request is vetted afresh). Anything that fails to resolve,
resolves to a non-public address, or fails to connect is **refused** — the proxy
never falls back to a direct or unvetted connection.

Precise guarantee (state it honestly, per the project's UX-copy doctrine): this
constrains the HTTP(S) traffic that honours ``--proxy``, including yt-dlp's
redirects and extractor-constructed URLs. It is **not** a kernel-level route
removal and **not** a sandbox: a sub-process yt-dlp spawns (e.g. ffmpeg) that
ignores the proxy, or the worker container's own routable network, is beyond it —
closing that needs a host-level egress firewall. See ``docs/operations.md``.

The module is stdlib-only (mirroring ``netcheck``) and injects its ``resolver``
and ``connector`` so the full request loop is exercised in tests without real DNS
or real egress.
"""

import argparse
import contextlib
import io
import ipaddress
import socket
import socketserver
import sys
import threading
import time
from collections.abc import Callable
from urllib.parse import urlsplit

from voxint.media.netcheck import HostNotPublicError, Resolver, resolve_public_addresses

# A connector opens a TCP connection to an already-vetted address and returns the
# connected socket (or raises OSError). Injected in tests so the vet→pin→connect
# path can be exercised against a local stub without real egress.
Connector = Callable[[str, int, float], socket.socket]

# yt-dlp's default proxy port is arbitrary; 3128 is the conventional forward-proxy
# port (Squid's default) and matches the compose overlay.
DEFAULT_LISTEN = "0.0.0.0:3128"
# Connect + idle timeouts keep a stuck upstream from pinning a worker thread
# forever. Generous enough for slow media origins; the download's own wall-clock
# cap lives in the yt-dlp invocation, not here.
_CONNECT_TIMEOUT_SECONDS = 30.0
_IDLE_TIMEOUT_SECONDS = 300.0
# One request line + headers must arrive within this bound and under this size —
# a client that dribbles a multi-kilobyte header block is hostile or broken.
_MAX_HEADER_BYTES = 65536
_HEADER_READ_TIMEOUT_SECONDS = 30.0


class ProxyError(Exception):
    """A request is refused. ``status`` is the HTTP status to return to the
    client; the message names only the host (never a full URL, whose query can
    carry a signed token), matching the ``netcheck`` message contract."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


def _default_connector(address: str, port: int, timeout: float) -> socket.socket:
    """Open a TCP connection to a vetted numeric ``address`` (v4 or v6).

    ``address`` is always an IP literal produced by :func:`resolve_public_addresses`,
    so ``create_connection`` performs no real name lookup — it connects to exactly
    the address we vetted, which is the pin.
    """
    return socket.create_connection((address, port), timeout=timeout)


def _vetted_addresses(host: str, resolver: Resolver) -> list[str]:
    """Resolve + vet ``host`` and return its public addresses as connect strings.

    Raises :class:`ProxyError` (403) if the host is missing or resolves to any
    non-public address, or cannot be verified as public — the same fail-closed
    policy as the worker gate, reusing the identical per-address rule so the two
    can never diverge.
    """
    if not host:
        raise ProxyError("403 Forbidden", "request has no host")
    try:
        # The identical per-address policy the worker gate uses; the resolver is
        # the same test seam single-sourced in netcheck.
        vetted = resolve_public_addresses(host, resolver=resolver)
    except HostNotPublicError as exc:
        # Message already names only the host; surface it as a refusal.
        raise ProxyError("403 Forbidden", str(exc)) from None
    return [_connect_str(ip) for ip in vetted]


def _connect_str(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str:
    """The plain (bracket-free) string ``create_connection`` wants for an IP."""
    return ip.compressed


def _send_error(conn: socket.socket, status: str, detail: str) -> None:
    """Write a minimal, connection-closing HTTP error and nothing else.

    The body repeats only the host-only ``detail`` (never a full URL). ``502`` is
    used for an upstream connect failure, ``403`` for a policy refusal.
    """
    body = detail.encode("utf-8", "replace")
    head = (
        f"HTTP/1.1 {status}\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode("ascii")
    with contextlib.suppress(OSError):
        conn.sendall(head + body)  # client already gone — nothing to report


def _read_head(reader: io.RawIOBase, conn: socket.socket) -> tuple[str, list[str]]:
    """Read the request line and header lines (up to the blank line).

    Returns ``(request_line, header_lines)`` with CRLFs stripped. Enforces the
    size/time bounds; a client that never sends a blank line, or overruns the
    byte cap, is refused rather than allowed to hang or exhaust memory.

    ``reader`` is the handler's UNBUFFERED ``rfile`` (``rbufsize = 0``), so it
    never reads past the blank line into the body or TLS ClientHello — leaving
    those bytes on the raw socket for :func:`_tunnel` to forward.

    The timeout is an ABSOLUTE deadline over the whole head, not per-read: a client
    dribbling one byte per socket-timeout would otherwise pin a handler thread for
    (byte-cap * timeout) seconds. Once the deadline passes the read is refused.
    """
    deadline = time.monotonic() + _HEADER_READ_TIMEOUT_SECONDS
    lines: list[str] = []
    total = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProxyError("408 Request Timeout", "request head timed out")
        conn.settimeout(remaining)
        raw = reader.readline(_MAX_HEADER_BYTES + 1)
        if not raw:
            raise ProxyError("400 Bad Request", "connection closed before request head")
        total += len(raw)
        if total > _MAX_HEADER_BYTES:
            raise ProxyError("431 Request Header Fields Too Large", "request head too large")
        line = raw.decode("iso-8859-1").rstrip("\r\n")
        if not lines:
            if not line:
                raise ProxyError("400 Bad Request", "empty request line")
            lines.append(line)  # the request line
            continue
        if line == "":
            break
        lines.append(line)
    return lines[0], lines[1:]


def _parse_request_line(request_line: str) -> tuple[str, str, str]:
    """Split ``METHOD TARGET VERSION``; refuse anything malformed.

    HTTP permits exactly one SP between the three tokens, and none may be empty —
    reject multi-space / tab / empty-token lines rather than guess.
    """
    parts = request_line.split(" ")
    if len(parts) != 3 or not all(parts):
        raise ProxyError("400 Bad Request", "malformed request line")
    return parts[0], parts[1], parts[2]


def _parse_port(port_s: str) -> int:
    """Parse an explicit port string strictly: digits only, in ``1..65535``.

    Rejects empty, non-numeric, out-of-range, and absurdly long inputs (a
    >4300-digit string would otherwise raise ``ValueError`` from ``int()`` under
    the CPython int-string limit, or ``OverflowError`` later at connect time)."""
    if not port_s.isdigit() or len(port_s) > 5:
        raise ProxyError("400 Bad Request", "malformed port")
    port = int(port_s)
    if not 1 <= port <= 65535:
        raise ProxyError("400 Bad Request", "port out of range")
    return port


def _pump(src: socket.socket, dst: socket.socket) -> None:
    """Copy ``src`` → ``dst`` until EOF or an idle timeout, then half-close
    ``dst``'s write side so the peer sees the end of stream."""
    try:
        while True:
            chunk = src.recv(65536)
            if not chunk:
                break
            dst.sendall(chunk)
    except OSError:
        pass  # timeout / reset — tear this direction down
    finally:
        with contextlib.suppress(OSError):
            dst.shutdown(socket.SHUT_WR)


def _tunnel(a: socket.socket, b: socket.socket) -> None:
    """Blind bidirectional byte pump between two connected sockets until both
    sides reach EOF (or idle out). Used verbatim for CONNECT and, after the
    rewritten head is sent upstream, for the absolute-form HTTP body/response."""
    a.settimeout(_IDLE_TIMEOUT_SECONDS)
    b.settimeout(_IDLE_TIMEOUT_SECONDS)
    other = threading.Thread(target=_pump, args=(a, b), daemon=True)
    other.start()
    _pump(b, a)  # this direction runs inline
    other.join()


class _Handler(socketserver.StreamRequestHandler):
    """One client connection: read the head, vet the destination, pin+connect,
    then tunnel. Every refusal fails closed with an HTTP error and no upstream
    connection."""

    # Unbuffered rfile: readline() must not read past the request head into body
    # or TLS bytes, which would strand them off the raw socket the tunnel copies.
    rbufsize = 0

    # Injected by the server subclass (set as class attributes in run_proxy).
    resolver: Resolver
    connector: Connector

    def handle(self) -> None:
        try:
            request_line, headers = _read_head(self.rfile, self.connection)  # type: ignore[arg-type]
            method, target, _ = _parse_request_line(request_line)
        except ProxyError as exc:
            _send_error(self.connection, exc.status, str(exc))
            return
        except OSError:
            return  # client vanished mid-head

        if method.upper() == "CONNECT":
            self._do_connect(target)
        else:
            self._do_absolute(method, target, headers)

    # -- CONNECT (HTTPS): parse host:port, vet, pin, 200, blind tunnel ----------
    def _do_connect(self, target: str) -> None:
        try:
            host, port = _split_authority(target, default_port=443)
            addresses = _vetted_addresses(host, self.resolver)
            upstream = self._connect_any(addresses, port)
        except ProxyError as exc:
            _send_error(self.connection, exc.status, str(exc))
            return
        try:
            self.connection.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            _tunnel(self.connection, upstream)
        finally:
            _close(upstream)

    # -- absolute-form HTTP: vet, pin, forward rewritten head, tunnel ----------
    def _do_absolute(self, method: str, target: str, headers: list[str]) -> None:
        try:
            host, port, origin = _parse_absolute_target(target)
            addresses = _vetted_addresses(host, self.resolver)
            upstream = self._connect_any(addresses, port)
        except ProxyError as exc:
            _send_error(self.connection, exc.status, str(exc))
            return
        try:
            head = _rewrite_head(method, origin, host, port, headers)
            upstream.sendall(head)
            _tunnel(self.connection, upstream)
        finally:
            _close(upstream)

    def _connect_any(self, addresses: list[str], port: int) -> socket.socket:
        """Try the vetted addresses in answer order (v4/v6 failover), NEVER
        re-resolving. A total connect failure is a 502 refusal, not a fallback."""
        last = "no vetted address"
        for address in addresses:
            try:
                return self.connector(address, port, _CONNECT_TIMEOUT_SECONDS)
            except OSError as exc:
                last = f"{type(exc).__name__}"
                continue
        raise ProxyError("502 Bad Gateway", f"upstream connect failed: {last}")


def _split_authority(authority: str, *, default_port: int) -> tuple[str, int]:
    """Split a ``CONNECT`` ``host:port`` (or bracketed ``[v6]:port``) authority.

    Strict: a malformed or out-of-range port, or any trailing garbage after a
    bracketed IPv6 literal, is a 400 — never silently swallowed and defaulted. A
    filtering proxy must not be lenient about input it will act on.
    """
    authority = authority.strip()
    if authority.startswith("["):
        end = authority.find("]")
        if end == -1:
            raise ProxyError("400 Bad Request", "malformed IPv6 authority")
        host = authority[1:end]
        rest = authority[end + 1 :]
        if rest == "":
            port = default_port
        elif rest.startswith(":"):
            port = _parse_port(rest[1:])
        else:
            raise ProxyError("400 Bad Request", "malformed IPv6 authority")
    elif ":" in authority:
        host, _, port_s = authority.rpartition(":")
        port = _parse_port(port_s)
    else:
        host, port = authority, default_port
    host = host.rstrip(".")
    if not host:
        raise ProxyError("400 Bad Request", "authority has no host")
    return host, port


def _parse_absolute_target(target: str) -> tuple[str, int, str]:
    """Validate an absolute-form ``http://…`` proxy target and return
    ``(host, port, origin)``.

    Fail-closed and http-only: a malformed URL (bad IPv6 literal, out-of-range
    port) becomes a clean 400 rather than an unhandled ``ValueError`` out of the
    handler; an ``https://`` absolute-form target is refused (yt-dlp uses CONNECT
    for TLS — forwarding it as cleartext to :443 would mis-speak the protocol);
    embedded credentials are refused (matching ``netcheck.parse_http_url``). The
    origin-form target is rebuilt from the parsed parts (never a string slice, and
    the fragment — client-side only — is dropped).
    """
    try:
        parts = urlsplit(target)
        port = parts.port  # lazily parsed; touch it so a bad port raises here
        host = parts.hostname
    except ValueError:
        raise ProxyError("400 Bad Request", "malformed proxy target") from None
    if parts.scheme != "http":
        raise ProxyError(
            "400 Bad Request", "absolute-form proxy target must be http (use CONNECT for https)"
        )
    if parts.username is not None or parts.password is not None:
        raise ProxyError("400 Bad Request", "proxy target must not embed credentials")
    host = (host or "").rstrip(".")
    if not host:
        raise ProxyError("400 Bad Request", "proxy target has no host")
    origin = parts.path or "/"
    if parts.query:
        origin += f"?{parts.query}"
    return host, port or 80, origin


# Headers a forward proxy must not pass upstream: RFC 7230 hop-by-hop plus the
# proxy-auth pair. ``Host`` is stripped too — we synthesize exactly one from the
# VETTED authority so a client cannot front a different vhost than the one vetted.
_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-connection", "proxy-authorization",
    "proxy-authenticate", "te", "trailer", "transfer-encoding", "upgrade", "host",
})


def _rewrite_head(
    method: str, origin: str, host: str, port: int, headers: list[str]
) -> bytes:
    """Rebuild the request head in origin-form for the upstream connection.

    Drops every hop-by-hop header (and every field a ``Connection`` header names),
    synthesizes exactly one ``Host`` from the vetted authority (bracketing an IPv6
    literal, appending a non-default port), and forces ``Connection: close`` — so
    the blind tunnel cannot carry a second, unvetted pipelined request to the
    pinned upstream. All other client headers pass through unchanged.
    """
    # Any token named by a Connection header is also hop-by-hop for this message.
    nominated: set[str] = set()
    for line in headers:
        name, _, value = line.partition(":")
        if name.strip().lower() == "connection":
            nominated.update(t.strip().lower() for t in value.split(",") if t.strip())
    kept = [
        line
        for line in headers
        if (name := line.split(":", 1)[0].strip().lower()) not in _HOP_BY_HOP
        and name not in nominated
    ]
    authority = f"[{host}]" if ":" in host else host
    if port != 80:
        authority = f"{authority}:{port}"
    out = [
        f"{method} {origin} HTTP/1.1",
        f"Host: {authority}",
        "Connection: close",
        *kept,
        "",
        "",
    ]
    return "\r\n".join(out).encode("iso-8859-1")


def _close(sock: socket.socket) -> None:
    with contextlib.suppress(OSError):
        sock.close()


class _ProxyServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def run_proxy(
    host: str,
    port: int,
    *,
    resolver: Resolver = socket.getaddrinfo,
    connector: Connector = _default_connector,
    ready: threading.Event | None = None,
) -> _ProxyServer:
    """Build and start the threaded proxy, returning the running server.

    ``resolver`` and ``connector`` are injected in tests. ``ready`` (if given) is
    set once the server is listening. The caller owns the returned server's
    lifecycle (``serve_forever`` / ``shutdown``); :func:`main` runs it forever.
    """
    handler = type(
        "_BoundHandler",
        (_Handler,),
        # staticmethod on BOTH: a bare function stored as a class attribute becomes
        # a descriptor (bound method), which would inject the handler instance as
        # the first argument and corrupt the resolver/connector call.
        {"resolver": staticmethod(resolver), "connector": staticmethod(connector)},
    )
    server = _ProxyServer((host, port), handler)
    if ready is not None:
        ready.set()
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m voxint.media.egress_proxy",
        description="Fail-closed filtering forward proxy for restricted yt-dlp egress.",
    )
    parser.add_argument(
        "--listen",
        default=DEFAULT_LISTEN,
        help=f"HOST:PORT to bind (default {DEFAULT_LISTEN}).",
    )
    args = parser.parse_args(argv)
    try:
        host, port = _split_authority(args.listen, default_port=3128)
    except ProxyError as exc:
        parser.error(f"invalid --listen {args.listen!r}: {exc}")  # exit 2 with usage
    server = run_proxy(host, port)
    print(f"voxint egress proxy listening on {host}:{port}", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
