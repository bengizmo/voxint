"""Full-loop tests for the egress proxy: a real client speaks to a real proxy
server, with an injected resolver (synthetic DNS) and an injected connector that
records the address it was asked to reach and redirects the actual socket to a
local stub. This exercises ``handle`` → CONNECT/absolute-form → vet → pin →
connect → tunnel / refuse without any real DNS or egress.

The load-bearing assertions: on ALLOW the connector is called with the exact
vetted public IP (the pin); on REFUSE the connector is NEVER called (fail-closed,
no unvetted connection).
"""

import contextlib
import socket
import threading
from collections.abc import Callable, Iterator

import pytest

from voxint.media import egress_proxy
from voxint.media.egress_proxy import _ProxyServer, run_proxy

_Answer = list[tuple[int, int, int, str, tuple[str, int]]]
_PUBLIC_IP = "93.184.216.34"  # real global IP; the connector never actually dials it


def _resolver(mapping: dict[str, list[str]]) -> Callable[..., _Answer]:
    def resolve(host: str, *args: object, **kwargs: object) -> _Answer:
        addrs = mapping.get(host)
        if addrs is None:
            raise OSError(f"no synthetic answer for {host!r}")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, 0)) for a in addrs]

    return resolve


class _Stub:
    """A one-shot local origin: either replies with a canned HTTP response then
    closes (HTTP path), or echoes bytes until EOF (CONNECT tunnel path).

    In canned mode it records everything received in ``self.received`` (up to
    ``reply_after``, if given) so a test can assert the request body reached the
    upstream — the unbuffered-``rfile`` tunnel invariant."""

    def __init__(
        self, *, mode: str, payload: bytes = b"", reply_after: bytes | None = None
    ) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(4)
        self.addr: tuple[str, int] = self._sock.getsockname()
        self._mode = mode
        self._payload = payload
        self._reply_after = reply_after
        self.received = b""
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            if self._mode == "canned":
                conn.settimeout(1.0)
                buf = b""
                try:
                    while True:
                        chunk = conn.recv(65536)
                        if not chunk:
                            break
                        buf += chunk
                        if self._reply_after is None or self._reply_after in buf:
                            break
                except OSError:
                    pass
                self.received = buf
                with contextlib.suppress(OSError):
                    conn.sendall(self._payload)
            else:  # echo
                while True:
                    try:
                        chunk = conn.recv(65536)
                    except OSError:
                        return
                    if not chunk:
                        return
                    conn.sendall(chunk)

    def close(self) -> None:
        self._sock.close()


def _connector_to(stub_addr: tuple[str, int], calls: list[tuple[str, int]]):
    def connect(address: str, port: int, timeout: float) -> socket.socket:
        calls.append((address, port))
        return socket.create_connection(stub_addr, timeout=timeout)

    return connect


class _ProxyFixture:
    def __init__(self, server: _ProxyServer, calls: list[tuple[str, int]]) -> None:
        self.server = server
        self.calls = calls
        self.addr: tuple[str, int] = server.server_address  # type: ignore[assignment]

    def client(self) -> socket.socket:
        c = socket.create_connection(self.addr, timeout=5)
        c.settimeout(5)
        return c


@pytest.fixture
def make_proxy() -> Iterator[Callable[[dict[str, list[str]], tuple[str, int]], _ProxyFixture]]:
    servers: list[_ProxyServer] = []

    def build(mapping: dict[str, list[str]], stub_addr: tuple[str, int]) -> _ProxyFixture:
        calls: list[tuple[str, int]] = []
        server = run_proxy(
            "127.0.0.1",
            0,
            resolver=_resolver(mapping),
            connector=_connector_to(stub_addr, calls),
        )
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return _ProxyFixture(server, calls)

    yield build
    for s in servers:
        s.shutdown()
        s.server_close()


def _recv_all(sock: socket.socket) -> bytes:
    chunks = []
    while True:
        try:
            chunk = sock.recv(65536)
        except OSError:
            break
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


# -- absolute-form HTTP -----------------------------------------------------


def test_http_allow_connects_to_vetted_ip(make_proxy) -> None:
    reply = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nhi"
    stub = _Stub(mode="canned", payload=reply)
    try:
        proxy = make_proxy({"public.test": [_PUBLIC_IP]}, stub.addr)
        c = proxy.client()
        c.sendall(b"GET http://public.test/path HTTP/1.1\r\nHost: public.test\r\n\r\n")
        data = _recv_all(c)
        c.close()
    finally:
        stub.close()
    assert data.endswith(b"hi")
    # The pin: connector was asked for the exact vetted IP on port 80.
    assert proxy.calls == [(_PUBLIC_IP, 80)]


def test_http_refuse_private_never_connects(make_proxy) -> None:
    stub = _Stub(mode="canned", payload=b"HTTP/1.1 200 OK\r\n\r\n")
    try:
        proxy = make_proxy({"evil.test": ["192.0.2.5"]}, stub.addr)  # TEST-NET-1, non-public
        c = proxy.client()
        c.sendall(b"GET http://evil.test/ HTTP/1.1\r\nHost: evil.test\r\n\r\n")
        data = _recv_all(c)
        c.close()
    finally:
        stub.close()
    assert data.startswith(b"HTTP/1.1 403")
    assert proxy.calls == []  # fail-closed: no upstream connection attempted


def test_http_refuse_unresolvable_never_connects(make_proxy) -> None:
    stub = _Stub(mode="canned", payload=b"HTTP/1.1 200 OK\r\n\r\n")
    try:
        proxy = make_proxy({}, stub.addr)  # nothing resolves
        c = proxy.client()
        c.sendall(b"GET http://nx.test/ HTTP/1.1\r\nHost: nx.test\r\n\r\n")
        data = _recv_all(c)
        c.close()
    finally:
        stub.close()
    assert data.startswith(b"HTTP/1.1 403")
    assert proxy.calls == []


# -- HTTPS CONNECT ----------------------------------------------------------


def test_connect_allow_tunnels_to_vetted_ip(make_proxy) -> None:
    stub = _Stub(mode="echo")
    try:
        proxy = make_proxy({"public.test": [_PUBLIC_IP]}, stub.addr)
        c = proxy.client()
        c.sendall(b"CONNECT public.test:443 HTTP/1.1\r\nHost: public.test:443\r\n\r\n")
        established = c.recv(65536)
        assert established.startswith(b"HTTP/1.1 200")
        # Now the tunnel is raw: bytes we send are echoed by the stub.
        c.sendall(b"ping-through-tunnel")
        echoed = c.recv(65536)
        c.close()
    finally:
        stub.close()
    assert echoed == b"ping-through-tunnel"
    assert proxy.calls == [(_PUBLIC_IP, 443)]


def test_connect_refuse_private_never_connects(make_proxy) -> None:
    stub = _Stub(mode="echo")
    try:
        proxy = make_proxy({"evil.test": ["198.51.100.7"]}, stub.addr)  # TEST-NET-2, non-public
        c = proxy.client()
        c.sendall(b"CONNECT evil.test:443 HTTP/1.1\r\nHost: evil.test:443\r\n\r\n")
        data = c.recv(65536)
        c.close()
    finally:
        stub.close()
    assert data.startswith(b"HTTP/1.1 403")
    assert proxy.calls == []


def test_connect_metadata_endpoint_refused(make_proxy) -> None:
    stub = _Stub(mode="echo")
    try:
        proxy = make_proxy({"metadata.test": ["169.254.169.254"]}, stub.addr)
        c = proxy.client()
        c.sendall(b"CONNECT metadata.test:80 HTTP/1.1\r\n\r\n")
        data = c.recv(65536)
        c.close()
    finally:
        stub.close()
    assert data.startswith(b"HTTP/1.1 403")
    assert proxy.calls == []


def test_connect_upstream_failure_is_502_not_fallback() -> None:
    # Resolver says public, but the connector always fails: the proxy must return
    # 502 and never silently fall back to a direct connection.
    def failing_connector(address: str, port: int, timeout: float) -> socket.socket:
        raise OSError("connection refused")

    server = run_proxy(
        "127.0.0.1",
        0,
        resolver=_resolver({"public.test": [_PUBLIC_IP]}),
        connector=failing_connector,
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        c = socket.create_connection(server.server_address, timeout=5)  # type: ignore[arg-type]
        c.settimeout(5)
        c.sendall(b"CONNECT public.test:443 HTTP/1.1\r\n\r\n")
        data = c.recv(65536)
        c.close()
    finally:
        server.shutdown()
        server.server_close()
    assert data.startswith(b"HTTP/1.1 502")


def test_connect_mixed_public_private_refused(make_proxy) -> None:
    # A host resolving to one public AND one private address is refused wholesale,
    # never filtered to its public member — and no connection is attempted.
    stub = _Stub(mode="echo")
    try:
        proxy = make_proxy({"mixed.test": [_PUBLIC_IP, "192.0.2.9"]}, stub.addr)
        c = proxy.client()
        c.sendall(b"CONNECT mixed.test:443 HTTP/1.1\r\n\r\n")
        data = c.recv(65536)
        c.close()
    finally:
        stub.close()
    assert data.startswith(b"HTTP/1.1 403")
    assert proxy.calls == []


def test_connect_ipv6_allow_pins_compressed(make_proxy) -> None:
    # A host that resolves to IPv6 pins the compressed literal at connect time.
    stub = _Stub(mode="echo")
    try:
        proxy = make_proxy({"v6.test": ["2606:4700:4700::1111"]}, stub.addr)
        c = proxy.client()
        c.sendall(b"CONNECT v6.test:443 HTTP/1.1\r\n\r\n")
        established = c.recv(65536)
        c.close()
    finally:
        stub.close()
    assert established.startswith(b"HTTP/1.1 200")
    assert proxy.calls == [("2606:4700:4700::1111", 443)]  # compressed literal pin


def test_http_malformed_port_is_400_no_connect(make_proxy) -> None:
    stub = _Stub(mode="canned", payload=b"HTTP/1.1 200 OK\r\n\r\n")
    try:
        proxy = make_proxy({"public.test": [_PUBLIC_IP]}, stub.addr)
        c = proxy.client()
        c.sendall(b"GET http://public.test:99999/ HTTP/1.1\r\nHost: public.test\r\n\r\n")
        data = _recv_all(c)
        c.close()
    finally:
        stub.close()
    assert data.startswith(b"HTTP/1.1 400")
    assert proxy.calls == []  # malformed target never reaches the connector


def test_http_https_absolute_form_refused(make_proxy) -> None:
    stub = _Stub(mode="canned", payload=b"HTTP/1.1 200 OK\r\n\r\n")
    try:
        proxy = make_proxy({"public.test": [_PUBLIC_IP]}, stub.addr)
        c = proxy.client()
        c.sendall(b"GET https://public.test/ HTTP/1.1\r\nHost: public.test\r\n\r\n")
        data = _recv_all(c)
        c.close()
    finally:
        stub.close()
    assert data.startswith(b"HTTP/1.1 400")  # https must use CONNECT
    assert proxy.calls == []


def test_http_request_body_reaches_upstream(make_proxy) -> None:
    # The unbuffered-rfile invariant: bytes after the head (here a request body)
    # are NOT stranded in a reader buffer — the tunnel forwards them upstream.
    reply = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"
    stub = _Stub(mode="canned", payload=reply, reply_after=b"hello")
    try:
        proxy = make_proxy({"public.test": [_PUBLIC_IP]}, stub.addr)
        c = proxy.client()
        c.sendall(
            b"POST http://public.test/ HTTP/1.1\r\n"
            b"Host: public.test\r\nContent-Length: 5\r\n\r\nhello"
        )
        data = _recv_all(c)
        c.close()
    finally:
        stub.close()
    assert data.endswith(b"ok")
    assert stub.received.endswith(b"hello")  # body made it through the tunnel
    # and the rewritten head forced Connection: close + kept the vetted Host
    assert b"Connection: close" in stub.received
    assert b"Host: public.test" in stub.received


def test_header_read_deadline_refuses_dribbler(make_proxy, monkeypatch) -> None:
    # A client that opens a request but never finishes the head must be refused on
    # the absolute deadline, not held until the byte cap.
    monkeypatch.setattr(egress_proxy, "_HEADER_READ_TIMEOUT_SECONDS", 0.3)
    stub = _Stub(mode="canned", payload=b"HTTP/1.1 200 OK\r\n\r\n")
    try:
        proxy = make_proxy({"public.test": [_PUBLIC_IP]}, stub.addr)
        c = proxy.client()
        c.sendall(b"GET http://public.test/ HTTP/1.1\r\n")  # no terminating blank line
        c.settimeout(3)
        data = _recv_all(c)
        c.close()
    finally:
        stub.close()
    assert data.startswith(b"HTTP/1.1 408") or data == b""  # timed out, refused
    assert proxy.calls == []
