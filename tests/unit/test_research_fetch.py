"""read_url hardening (issue #39): pinning, per-hop revalidation, caps, hygiene.

All network behavior is httpx.MockTransport; all DNS is injected fake
resolvers — no test touches sockets or real DNS. The MockTransport handler
sees the REAL outgoing request, so the pinning contract (rewritten URL host,
Host header, SNI extension) is asserted on what would hit the wire.
"""

import socket
from collections.abc import Callable, Iterator

import httpx
import pytest

from voxint.config import Settings
from voxint.media.netcheck import Resolver
from voxint.research import fetch as fetch_mod
from voxint.research.budget import Attribution, ResearchBudget
from voxint.research.fetch import read_url

ATTR = Attribution(feature="test", reason="unit-test")

PUBLIC_A = "93.184.216.34"
PUBLIC_B = "8.8.8.8"
METADATA = "169.254.169.254"
NAT64_LOOPBACK = "64:ff9b::127.0.0.1"


def make_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "voxint_web_research": True,
        "web_search_base_url": "http://searx.lan:8888",
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)  # type: ignore[arg-type]


def budget(reads: int = 5) -> ResearchBudget:
    return ResearchBudget(max_searches=5, max_reads=reads)


def resolver_map(table: dict[str, list[str]]) -> Resolver:
    def resolve(host: str, *args: object, **kwargs: object) -> list[
        tuple[int, int, int, str, tuple[str, int]]
    ]:
        if host not in table:
            raise socket.gaierror("no such host")
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (addr, 0))
            for addr in table[host]
        ]

    return resolve


def stream_response(
    status: int, *, headers: dict[str, str] | None = None, body: bytes = b""
) -> httpx.Response:
    """A MockTransport response whose body is STREAMED (bytes content would be
    pre-consumed and break the fetcher's iter_raw — matching real transports
    requires an iterator body)."""
    return httpx.Response(status, headers=headers or {}, content=iter([body]))


def factory_for(
    handler: Callable[[httpx.Request], httpx.Response],
) -> "fetch_mod.ClientFactory":
    def factory(timeout: httpx.Timeout) -> httpx.Client:
        return httpx.Client(
            transport=httpx.MockTransport(handler),
            follow_redirects=False,
            trust_env=False,
            timeout=timeout,
        )

    return factory


def exploding_factory(timeout: httpx.Timeout) -> httpx.Client:
    raise AssertionError("network client must not be constructed in this scenario")


def test_happy_path_pins_ip_and_preserves_host_identity() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return stream_response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            body=b"<title>T</title><p>hello world</p>",
        )

    out = read_url(
        "https://example.com:8443/page?q=1#frag",
        settings=make_settings(),
        budget=budget(),
        attribution=ATTR,
        client_factory=factory_for(handler),
        resolver=resolver_map({"example.com": [PUBLIC_A]}),
    )
    assert out.ok is True, out.error_detail
    assert "hello world" in out.text
    assert out.title == "T"
    assert out.host == "example.com"
    assert out.hops == 0
    # The pinning contract, asserted on the wire-bound request:
    request = seen[0]
    assert request.url.host == PUBLIC_A  # connects to the vetted address
    assert request.url.port == 8443
    assert request.headers["host"] == "example.com:8443"  # logical identity
    assert request.extensions["sni_hostname"] == "example.com"  # TLS identity
    assert request.headers["accept-encoding"] == "identity"
    assert "#" not in str(request.url)  # fragment never sent
    # final_url is the logical URL (fragment-free), not the pinned-IP URL.
    assert out.final_url == "https://example.com:8443/page?q=1"


def test_disabled_returns_structured_outcome_with_zero_network() -> None:
    out = read_url(
        "https://example.com/",
        settings=Settings(_env_file=None),  # voxint_web_research=False
        budget=budget(),
        attribution=ATTR,
        client_factory=exploding_factory,
        resolver=resolver_map({}),  # would raise if consulted
    )
    assert out.error == "disabled"


def test_budget_exhausted_is_structured_and_networkless() -> None:
    b = budget(reads=1)
    assert b.try_consume_read() is True  # spend it
    out = read_url(
        "https://example.com/",
        settings=make_settings(),
        budget=b,
        attribution=ATTR,
        client_factory=exploding_factory,
        resolver=resolver_map({}),
    )
    assert out.error == "budget_exhausted"


def test_invalid_input_urls_are_refused_before_any_network() -> None:
    for url in ["ftp://example.com/x", "http://127.0.0.1/x", "http://user:p@e.com/"]:
        out = read_url(
            url,
            settings=make_settings(),
            budget=budget(),
            attribution=ATTR,
            client_factory=exploding_factory,
            resolver=resolver_map({}),
        )
        assert out.error == "invalid_input"


def test_host_resolving_private_is_policy_refused() -> None:
    for answer in [METADATA, NAT64_LOOPBACK]:
        out = read_url(
            "https://internal.example.com/",
            settings=make_settings(),
            budget=budget(),
            attribution=ATTR,
            client_factory=exploding_factory,  # refusal happens before any client
            resolver=resolver_map({"internal.example.com": [answer]}),
        )
        assert out.error == "policy_refused"
        assert "internal.example.com" in out.error_detail


def test_redirect_hop_rebinding_to_metadata_is_refused() -> None:
    # Hop 1 (a.example.com) resolves public and redirects to b.example.com,
    # which resolves to the cloud metadata address — THE rebinding scenario.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302, headers={"location": "https://b.example.com/steal"}
        )

    out = read_url(
        "https://a.example.com/start",
        settings=make_settings(),
        budget=budget(),
        attribution=ATTR,
        client_factory=factory_for(handler),
        resolver=resolver_map(
            {"a.example.com": [PUBLIC_A], "b.example.com": [METADATA]}
        ),
    )
    assert out.error == "policy_refused"
    assert "b.example.com" in out.error_detail


def test_relative_redirect_resolves_against_logical_url_and_repins() -> None:
    hosts_seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts_seen.append(request.url.host)
        if request.url.path == "/start":
            return httpx.Response(301, headers={"location": "/moved"})
        assert request.url.path == "/moved"
        return stream_response(
            200, headers={"content-type": "text/plain"}, body=b"arrived"
        )

    out = read_url(
        "https://example.com/start",
        settings=make_settings(),
        budget=budget(),
        attribution=ATTR,
        client_factory=factory_for(handler),
        resolver=resolver_map({"example.com": [PUBLIC_A]}),
    )
    assert out.ok is True
    assert out.text == "arrived"
    assert out.hops == 1
    assert out.final_url == "https://example.com/moved"
    # Both requests were pinned (the relative Location resolved against the
    # LOGICAL url, then re-vetted and re-pinned — never against the IP url).
    assert hosts_seen == [PUBLIC_A, PUBLIC_A]


@pytest.mark.parametrize(
    "location",
    [
        "ftp://example.com/f",  # non-http scheme
        "http://user:pass@example.com/",  # credential-bearing
        "http://127.0.0.1/",  # private literal
        "http://localhost/",  # localhost
    ],
)
def test_hostile_redirect_targets_are_redirect_invalid(location: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": location})

    out = read_url(
        "https://example.com/",
        settings=make_settings(),
        budget=budget(),
        attribution=ATTR,
        client_factory=factory_for(handler),
        resolver=resolver_map({"example.com": [PUBLIC_A]}),
    )
    assert out.error == "redirect_invalid"


def test_missing_and_duplicate_location_are_redirect_invalid() -> None:
    def missing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302)

    def duplicate(request: httpx.Request) -> httpx.Response:
        headers = httpx.Headers(
            [("location", "https://a.example.com/"), ("location", "https://b.example.com/")]
        )
        return httpx.Response(302, headers=headers)

    for handler in (missing, duplicate):
        out = read_url(
            "https://example.com/",
            settings=make_settings(),
            budget=budget(),
            attribution=ATTR,
            client_factory=factory_for(handler),
            resolver=resolver_map({"example.com": [PUBLIC_A]}),
        )
        assert out.error == "redirect_invalid"


def test_redirect_limit_is_enforced() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        n = int(request.url.path.strip("/") or 0)
        return httpx.Response(302, headers={"location": f"https://example.com/{n + 1}"})

    out = read_url(
        "https://example.com/0",
        settings=make_settings(web_read_max_redirects=3),
        budget=budget(),
        attribution=ATTR,
        client_factory=factory_for(handler),
        resolver=resolver_map({"example.com": [PUBLIC_A]}),
    )
    assert out.error == "redirect_limit"


def test_non_identity_content_encoding_is_refused() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html", "content-encoding": "gzip"},
            content=b"\x1f\x8b\x08\x00compressed",
        )

    out = read_url(
        "https://example.com/",
        settings=make_settings(),
        budget=budget(),
        attribution=ATTR,
        client_factory=factory_for(handler),
        resolver=resolver_map({"example.com": [PUBLIC_A]}),
    )
    assert out.error == "encoding_refused"


@pytest.mark.parametrize("content_type", [None, "application/json", "image/png"])
def test_mime_allowlist_refuses_non_text(content_type: str | None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        headers = {} if content_type is None else {"content-type": content_type}
        return httpx.Response(200, headers=headers, content=b"{}")

    out = read_url(
        "https://example.com/",
        settings=make_settings(),
        budget=budget(),
        attribution=ATTR,
        client_factory=factory_for(handler),
        resolver=resolver_map({"example.com": [PUBLIC_A]}),
    )
    assert out.error == "mime_refused"


def test_declared_oversize_refused_early_and_lying_length_caught_streaming() -> None:
    settings = make_settings(web_read_max_bytes=1024)

    def declared(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/plain", "content-length": "999999"},
            content=b"",
        )

    out = read_url(
        "https://example.com/",
        settings=settings,
        budget=budget(),
        attribution=ATTR,
        client_factory=factory_for(declared),
        resolver=resolver_map({"example.com": [PUBLIC_A]}),
    )
    assert out.error == "too_large"

    def lying(request: httpx.Request) -> httpx.Response:
        # Streamed body far exceeds both the declared length and the cap: the
        # streaming counter, not the header, must be authoritative.
        def gen() -> Iterator[bytes]:
            for _ in range(64):
                yield b"x" * 1024

        return httpx.Response(
            200,
            headers={"content-type": "text/plain", "content-length": "10"},
            content=gen(),
        )

    out2 = read_url(
        "https://example.com/",
        settings=settings,
        budget=budget(),
        attribution=ATTR,
        client_factory=factory_for(lying),
        resolver=resolver_map({"example.com": [PUBLIC_A]}),
    )
    assert out2.error == "too_large"


def test_deadline_expiry_yields_timeout_outcome() -> None:
    now = [0.0]

    def handler(request: httpx.Request) -> httpx.Response:
        now[0] += 120.0  # the fetch "takes" longer than web_read_total_seconds
        return httpx.Response(302, headers={"location": "https://example.com/next"})

    out = read_url(
        "https://example.com/",
        settings=make_settings(),
        budget=budget(),
        attribution=ATTR,
        client_factory=factory_for(handler),
        resolver=resolver_map({"example.com": [PUBLIC_A]}),
        clock=lambda: now[0],
    )
    assert out.error == "timeout"


def test_http_error_status_is_terminal_with_status_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b"nope")

    out = read_url(
        "https://example.com/missing",
        settings=make_settings(),
        budget=budget(),
        attribution=ATTR,
        client_factory=factory_for(handler),
        resolver=resolver_map({"example.com": [PUBLIC_A]}),
    )
    # ONE stable error value; the numeric code rides a typed field (the #40
    # loop branches on exact strings, never on prefix-matching "http_404").
    assert out.error == "http_status"
    assert out.status_code == 404


def test_transport_error_retries_only_within_vetted_set() -> None:
    attempted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempted.append(request.url.host)
        if request.url.host == PUBLIC_A:
            raise httpx.ConnectError("refused")
        return stream_response(
            200, headers={"content-type": "text/plain"}, body=b"second address"
        )

    out = read_url(
        "https://example.com/",
        settings=make_settings(),
        budget=budget(),
        attribution=ATTR,
        client_factory=factory_for(handler),
        resolver=resolver_map({"example.com": [PUBLIC_A, PUBLIC_B]}),
    )
    assert out.ok is True
    assert out.text == "second address"
    assert attempted == [PUBLIC_A, PUBLIC_B]  # vetted order, no re-resolution


def test_all_vetted_addresses_failing_is_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    out = read_url(
        "https://example.com/",
        settings=make_settings(),
        budget=budget(),
        attribution=ATTR,
        client_factory=factory_for(handler),
        resolver=resolver_map({"example.com": [PUBLIC_A]}),
    )
    assert out.error == "transport_error"


def test_concurrency_limit_is_structured() -> None:
    acquired = 0
    while fetch_mod._read_slots.acquire(blocking=False):
        acquired += 1
    try:
        out = read_url(
            "https://example.com/",
            settings=make_settings(),
            budget=budget(),
            attribution=ATTR,
            client_factory=exploding_factory,
            resolver=resolver_map({"example.com": [PUBLIC_A]}),
        )
        assert out.error == "concurrency_limit"
    finally:
        for _ in range(acquired):
            fetch_mod._read_slots.release()


def test_html_extraction_end_to_end_strips_script_and_truncates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = b"<script>evil()</script><p>" + b"word " * 50_000 + b"</p>"
        return stream_response(200, headers={"content-type": "text/html"}, body=body)

    out = read_url(
        "https://example.com/",
        settings=make_settings(web_read_max_text_chars=200),
        budget=budget(),
        attribution=ATTR,
        client_factory=factory_for(handler),
        resolver=resolver_map({"example.com": [PUBLIC_A]}),
    )
    assert out.ok is True
    assert "evil" not in out.text
    assert len(out.text) <= 200
    assert out.truncated is True


SECRET_QUERY = "token=SUPERSECRETSIG"


@pytest.mark.parametrize(
    ("url", "table", "handler_kind"),
    [
        (f"http://127.0.0.1/x?{SECRET_QUERY}", {}, "none"),  # invalid_input
        (f"https://priv.example.com/x?{SECRET_QUERY}", {"priv.example.com": [METADATA]}, "none"),
        (f"https://example.com/x?{SECRET_QUERY}", {"example.com": [PUBLIC_A]}, "404"),
        (f"https://example.com/x?{SECRET_QUERY}", {"example.com": [PUBLIC_A]}, "connect_error"),
        (f"https://example.com/x?{SECRET_QUERY}", {"example.com": [PUBLIC_A]}, "redirect_private"),
    ],
)
def test_hygiene_no_refusal_ever_carries_url_or_query(
    url: str, table: dict[str, list[str]], handler_kind: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if handler_kind == "404":
            return httpx.Response(404)
        if handler_kind == "connect_error":
            raise httpx.ConnectError(f"boom for {request.url}")  # echoes the URL
        return httpx.Response(
            302, headers={"location": f"http://127.0.0.1/?{SECRET_QUERY}"}
        )

    out = read_url(
        url,
        settings=make_settings(),
        budget=budget(),
        attribution=ATTR,
        client_factory=factory_for(handler),
        resolver=resolver_map(table),
    )
    assert out.ok is False
    for text in (out.error_detail, out.final_url, str(out.error)):
        assert "SUPERSECRETSIG" not in text
        assert "?" not in text or "token" not in text


def test_attribution_log_lines_are_host_only(caplog: pytest.LogCaptureFixture) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return stream_response(
            200, headers={"content-type": "text/plain"}, body=b"ok"
        )

    with caplog.at_level("INFO", logger="voxint.research.fetch"):
        read_url(
            f"https://example.com/secret/path?{SECRET_QUERY}",
            settings=make_settings(),
            budget=budget(),
            attribution=ATTR,
            client_factory=factory_for(handler),
            resolver=resolver_map({"example.com": [PUBLIC_A]}),
        )
    fetch_lines = [r.getMessage() for r in caplog.records if "web_fetch" in r.getMessage()]
    assert fetch_lines, "every outbound request must produce an attribution line"
    for line in fetch_lines:
        assert "feature=test" in line
        assert "host=example.com" in line
        assert "SUPERSECRETSIG" not in line
        assert "/secret/path" not in line


def test_idn_host_is_resolved_and_pinned_via_punycode() -> None:
    seen: list[httpx.Request] = []
    resolved: list[str] = []

    def resolver(host: str, *args: object, **kwargs: object) -> list[
        tuple[int, int, int, str, tuple[str, int]]
    ]:
        resolved.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_A, 0))]

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return stream_response(
            200, headers={"content-type": "text/plain"}, body=b"ok"
        )

    out = read_url(
        "https://bücher.example/x",
        settings=make_settings(),
        budget=budget(),
        attribution=ATTR,
        client_factory=factory_for(handler),
        resolver=resolver,
    )
    assert out.ok is True
    # One canonical ASCII string everywhere: resolution, Host header, SNI.
    assert resolved == ["xn--bcher-kva.example"]
    assert seen[0].headers["host"] == "xn--bcher-kva.example"
    assert seen[0].extensions["sni_hostname"] == "xn--bcher-kva.example"


def test_read_consumes_exactly_one_budget_unit_even_across_hops() -> None:
    b = budget(reads=2)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/end"})
        return stream_response(
            200, headers={"content-type": "text/plain"}, body=b"done"
        )

    out = read_url(
        "https://example.com/start",
        settings=make_settings(),
        budget=b,
        attribution=ATTR,
        client_factory=factory_for(handler),
        resolver=resolver_map({"example.com": [PUBLIC_A]}),
    )
    assert out.ok is True
    assert b.reads_left == 1  # two hops, ONE read consumed


def test_invalid_input_and_concurrency_do_not_burn_read_budget() -> None:
    # Charging order (review finding, 3/3 reviewers): validate, take the
    # slot, THEN consume — refusals with no network work keep the quota.
    b = budget(reads=3)
    out = read_url(
        "http://127.0.0.1/x",
        settings=make_settings(),
        budget=b,
        attribution=ATTR,
        client_factory=exploding_factory,
        resolver=resolver_map({}),
    )
    assert out.error == "invalid_input"
    assert b.reads_left == 3  # unchanged

    acquired = 0
    while fetch_mod._read_slots.acquire(blocking=False):
        acquired += 1
    try:
        out2 = read_url(
            "https://example.com/",
            settings=make_settings(),
            budget=b,
            attribution=ATTR,
            client_factory=exploding_factory,
            resolver=resolver_map({"example.com": [PUBLIC_A]}),
        )
        assert out2.error == "concurrency_limit"
        assert b.reads_left == 3  # still unchanged
    finally:
        for _ in range(acquired):
            fetch_mod._read_slots.release()


PUBLIC_V6 = "2606:4700:4700::1111"


def test_ipv6_vetted_address_is_pinned_with_brackets() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return stream_response(
            200, headers={"content-type": "text/plain"}, body=b"v6 ok"
        )

    def resolver(host: str, *args: object, **kwargs: object) -> list[
        tuple[int, int, int, str, tuple[object, ...]]
    ]:
        return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (PUBLIC_V6, 0, 0, 0))]

    out = read_url(
        "https://example.com:8443/page",
        settings=make_settings(),
        budget=budget(),
        attribution=ATTR,
        client_factory=factory_for(handler),
        resolver=resolver,
    )
    assert out.ok is True, out.error_detail
    request = seen[0]
    # httpx brackets the IPv6 host in the URL; .host exposes it unbracketed.
    assert request.url.host == PUBLIC_V6
    assert f"[{PUBLIC_V6}]:8443" in str(request.url)
    assert request.headers["host"] == "example.com:8443"  # logical identity
    assert request.extensions["sni_hostname"] == "example.com"


def test_ipv6_literal_logical_url_pins_and_sets_bracketed_host_header() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return stream_response(
            200, headers={"content-type": "text/plain"}, body=b"literal v6"
        )

    out = read_url(
        f"http://[{PUBLIC_V6}]:8080/x",
        settings=make_settings(),
        budget=budget(),
        attribution=ATTR,
        client_factory=factory_for(handler),
        resolver=resolver_map({}),  # IP literal: no DNS resolution happens
    )
    assert out.ok is True, out.error_detail
    request = seen[0]
    assert request.url.host == PUBLIC_V6
    assert request.headers["host"] == f"[{PUBLIC_V6}]:8080"


def test_unparseable_location_is_a_structured_outcome_not_an_exception() -> None:
    # "http://[::1" raises ValueError in urljoin — and httpx's own protocol
    # validation may intercept it even earlier as RemoteProtocolError. Either
    # way the contract holds: remote behavior yields a STRUCTURED outcome
    # (redirect_invalid from our guard, transport_error from httpx's), never
    # an exception escaping read_url (review finding).
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://[::1"})

    out = read_url(
        "https://example.com/",
        settings=make_settings(),
        budget=budget(),
        attribution=ATTR,
        client_factory=factory_for(handler),
        resolver=resolver_map({"example.com": [PUBLIC_A]}),
    )
    assert out.ok is False
    assert out.error in ("redirect_invalid", "transport_error")
    assert "[::1" not in out.error_detail  # hostile Location never surfaced


def test_timeout_on_first_vetted_address_fails_over_to_second() -> None:
    # A blackholed first answer (dual-stack AAAA) must not fail the hop while
    # total time remains (review finding).
    attempted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempted.append(request.url.host)
        if request.url.host == PUBLIC_A:
            raise httpx.ConnectTimeout("blackholed")
        return stream_response(
            200, headers={"content-type": "text/plain"}, body=b"failover ok"
        )

    out = read_url(
        "https://example.com/",
        settings=make_settings(),
        budget=budget(),
        attribution=ATTR,
        client_factory=factory_for(handler),
        resolver=resolver_map({"example.com": [PUBLIC_A, PUBLIC_B]}),
    )
    assert out.ok is True
    assert out.text == "failover ok"
    assert attempted == [PUBLIC_A, PUBLIC_B]


def test_all_addresses_timing_out_reports_timeout_not_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("blackholed")

    out = read_url(
        "https://example.com/",
        settings=make_settings(),
        budget=budget(),
        attribution=ATTR,
        client_factory=factory_for(handler),
        resolver=resolver_map({"example.com": [PUBLIC_A, PUBLIC_B]}),
    )
    assert out.error == "timeout"


def test_mid_body_too_large_reports_bytes_fetched() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        def gen() -> Iterator[bytes]:
            for _ in range(64):
                yield b"x" * 1024

        return httpx.Response(
            200, headers={"content-type": "text/plain"}, content=gen()
        )

    out = read_url(
        "https://example.com/",
        settings=make_settings(web_read_max_bytes=4096),
        budget=budget(),
        attribution=ATTR,
        client_factory=factory_for(handler),
        resolver=resolver_map({"example.com": [PUBLIC_A]}),
    )
    assert out.error == "too_large"
    assert out.bytes_fetched > 4096  # audit trail shows what was pulled
