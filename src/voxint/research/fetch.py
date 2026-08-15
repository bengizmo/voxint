"""``read_url`` — the hardened single-page fetcher (issue #39).

The ingest SSRF doctrine (docs/architecture.md) documents a residual for
yt-dlp: redirects and DNS rebinding are beyond a userland check's reach because
yt-dlp resolves and connects on its own. This fetcher owns its connections, so
it CLOSES that residual for its path:

- every hop (the submitted URL and every redirect target) passes the shared
  string gate (:func:`voxint.research.policy.gate_research_url`) and fail-closed
  DNS vetting (:func:`voxint.media.netcheck.resolve_public_addresses`);
- the connection is PINNED to a vetted address: the request URL's host is
  rewritten to the vetted IP while the Host header and TLS SNI carry the
  canonical hostname, so the address that was checked IS the address that is
  connected to (no check-then-connect window) and certificate verification
  still runs against the hostname;
- a FRESH client is built per network attempt — a shared keepalive pool could
  reuse a connection made for one hostname to serve another that pins to the
  same IP, silently crossing TLS identities;
- responses must be identity-encoded (``Accept-Encoding: identity`` is sent and
  any ``Content-Encoding`` is refused), which removes the decompression-bomb
  class instead of bounding it; the streamed byte count is authoritative and
  ``Content-Length`` only an early refusal hint.

Failures are structured outcomes (a closed error vocabulary), not exceptions —
the issue #40 tool loop consumes them, and ``budget_exhausted`` in particular
must be a value the LLM can conclude from. Every outbound request logs one
attribution line; no message, outcome, or log line ever carries the URL, its
query, or a redirect Location — the host at most.

The total wall clock bounds every HTTP operation via the remaining-time
calculation; blocking DNS resolution is the one step a deadline cannot
hard-interrupt (documented in docs/architecture.md).
"""

import logging
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urljoin

import httpx

from voxint.config import Settings
from voxint.media.netcheck import (
    HostNotPublicError,
    Resolver,
    UrlPolicyError,
    resolve_public_addresses,
)
from voxint.media.redaction import cap_length, redact
from voxint.research.budget import Attribution, ResearchBudget
from voxint.research.extract import extract_html_text, extract_plain_text
from voxint.research.policy import ResearchUrl, gate_research_url

logger = logging.getLogger(__name__)

# Closed error vocabulary — the #40 loop branches on these strings.
ERROR_DISABLED = "disabled"
ERROR_INVALID_INPUT = "invalid_input"
ERROR_BUDGET_EXHAUSTED = "budget_exhausted"
ERROR_POLICY_REFUSED = "policy_refused"
ERROR_REDIRECT_LIMIT = "redirect_limit"
ERROR_REDIRECT_INVALID = "redirect_invalid"
ERROR_MIME_REFUSED = "mime_refused"
ERROR_ENCODING_REFUSED = "encoding_refused"
ERROR_TOO_LARGE = "too_large"
ERROR_TIMEOUT = "timeout"
ERROR_TRANSPORT = "transport_error"
ERROR_CONCURRENCY = "concurrency_limit"

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_ALLOWED_MIME = frozenset({"text/html", "application/xhtml+xml", "text/plain"})
_STREAM_CHUNK_BYTES = 64 * 1024
_USER_AGENT = "voxint-research/1"

# The issue's concurrency cap: at most this many read_url fetches in flight
# process-wide. Non-blocking — a third concurrent caller gets a structured
# refusal instead of a queue (nothing legitimate queues here today).
_MAX_CONCURRENT_READS = 2
_read_slots = threading.BoundedSemaphore(_MAX_CONCURRENT_READS)

# A fresh client per network attempt (pool isolation — see module docstring).
# Injectable so tests supply httpx.MockTransport without touching sockets.
ClientFactory = Callable[[httpx.Timeout], httpx.Client]


def _default_client_factory(timeout: httpx.Timeout) -> httpx.Client:
    # trust_env=False: an ambient HTTP(S)_PROXY must never silently reroute
    # egress (the same doctrine as ytdlp's always-passed --proxy). HTTP/1.1
    # only (http2 is not enabled), redirects driven manually by the hop loop.
    return httpx.Client(follow_redirects=False, trust_env=False, timeout=timeout)


@dataclass(frozen=True)
class FetchOutcome:
    """Structured result of one ``read_url`` invocation."""

    ok: bool
    error: str | None
    error_detail: str  # redacted; names the host at most, never the URL
    text: str = ""
    title: str = ""
    truncated: bool = False
    final_url: str = ""  # the logical URL actually read (evidence provenance)
    host: str = ""
    bytes_fetched: int = 0
    hops: int = 0
    duration_seconds: float = 0.0
    sources_chain: tuple[str, ...] = field(default_factory=tuple)  # hop hosts


def _refusal(
    error: str,
    detail: str,
    *,
    host: str = "",
    hops: int = 0,
    started: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> FetchOutcome:
    duration = 0.0 if started is None else max(0.0, clock() - started)
    return FetchOutcome(
        ok=False,
        error=error,
        error_detail=cap_length(redact(detail)),
        host=host,
        hops=hops,
        duration_seconds=duration,
    )


def _parse_content_type(raw: str | None) -> tuple[str | None, str | None]:
    """Split a Content-Type header into (media-type, charset) — both lowered."""
    if raw is None or not raw.strip():
        return None, None
    parts = raw.split(";")
    media_type = parts[0].strip().lower() or None
    charset = None
    for param in parts[1:]:
        name, _, value = param.partition("=")
        if name.strip().lower() == "charset":
            charset = value.strip().strip("\"'") or None
            break
    return media_type, charset


def read_url(
    url: str,
    *,
    settings: Settings,
    budget: ResearchBudget,
    attribution: Attribution,
    client_factory: ClientFactory | None = None,
    resolver: Resolver = socket.getaddrinfo,
    clock: Callable[[], float] = time.monotonic,
) -> FetchOutcome:
    """Fetch one page as extracted text under the full egress policy.

    Consumes one unit of the budget's read quota per invocation (not per hop).
    All failure modes return structured outcomes; nothing here raises for a
    remote server's behavior.
    """
    if not settings.voxint_web_research:
        return _refusal(ERROR_DISABLED, "web research is disabled")
    if not budget.try_consume_read():
        return _refusal(ERROR_BUDGET_EXHAUSTED, "read budget exhausted")

    started = clock()
    deadline = started + settings.web_read_total_seconds

    def remaining() -> float:
        left = deadline - clock()
        budget_left = budget.remaining_seconds()
        if budget_left is not None:
            left = min(left, budget_left)
        return left

    try:
        gated = gate_research_url(url)
    except UrlPolicyError as exc:
        return _refusal(ERROR_INVALID_INPUT, str(exc), started=started, clock=clock)

    if not _read_slots.acquire(blocking=False):
        return _refusal(
            ERROR_CONCURRENCY, "too many concurrent reads", started=started, clock=clock
        )
    try:
        return _follow_and_read(
            gated,
            settings=settings,
            attribution=attribution,
            client_factory=client_factory or _default_client_factory,
            resolver=resolver,
            clock=clock,
            started=started,
            remaining=remaining,
        )
    finally:
        _read_slots.release()


def _follow_and_read(
    gated: ResearchUrl,
    *,
    settings: Settings,
    attribution: Attribution,
    client_factory: ClientFactory,
    resolver: Resolver,
    clock: Callable[[], float],
    started: float,
    remaining: Callable[[], float],
) -> FetchOutcome:
    current = gated
    chain: list[str] = []
    for hop in range(settings.web_read_max_redirects + 1):
        chain.append(current.ascii_host)
        if remaining() <= 0:
            return _refusal(
                ERROR_TIMEOUT,
                "total time budget exhausted",
                host=current.ascii_host,
                hops=hop,
                started=started,
                clock=clock,
            )
        try:
            vetted = resolve_public_addresses(current.ascii_host, resolver=resolver)
        except HostNotPublicError as exc:
            _log_fetch(attribution, current.ascii_host, hop, "policy_refused", 0, started, clock)
            return _refusal(
                ERROR_POLICY_REFUSED,
                str(exc),
                host=current.ascii_host,
                hops=hop,
                started=started,
                clock=clock,
            )

        step = _request_once(
            current,
            vetted=[str(ip) for ip in vetted],
            settings=settings,
            attribution=attribution,
            client_factory=client_factory,
            clock=clock,
            started=started,
            hop=hop,
            remaining=remaining,
        )
        if isinstance(step, FetchOutcome):
            return FetchOutcome(
                ok=step.ok,
                error=step.error,
                error_detail=step.error_detail,
                text=step.text,
                title=step.title,
                truncated=step.truncated,
                final_url=current.url if step.ok else "",
                host=step.host,
                bytes_fetched=step.bytes_fetched,
                hops=hop,
                duration_seconds=max(0.0, clock() - started),
                sources_chain=tuple(chain),
            )
        # A redirect: gate the new logical target and loop.
        try:
            current = gate_research_url(step)
        except UrlPolicyError as exc:
            return _refusal(
                ERROR_REDIRECT_INVALID,
                f"redirect target refused: {exc}",
                host=current.ascii_host,
                hops=hop + 1,
                started=started,
                clock=clock,
            )
    return _refusal(
        ERROR_REDIRECT_LIMIT,
        f"more than {settings.web_read_max_redirects} redirects",
        host=current.ascii_host,
        hops=settings.web_read_max_redirects + 1,
        started=started,
        clock=clock,
    )


def _request_once(
    target: ResearchUrl,
    *,
    vetted: list[str],
    settings: Settings,
    attribution: Attribution,
    client_factory: ClientFactory,
    clock: Callable[[], float],
    started: float,
    hop: int,
    remaining: Callable[[], float],
) -> "FetchOutcome | str":
    """One hop: pinned request against the vetted addresses.

    Returns a redirect target (str) or a terminal FetchOutcome. Tries the
    vetted addresses in order on TRANSPORT errors only (never re-resolving);
    any HTTP response — including an error status — is terminal for the hop.
    """
    last_transport_error = "no vetted address attempted"
    for address in vetted:
        left = remaining()
        if left <= 0:
            return _refusal(
                ERROR_TIMEOUT,
                "total time budget exhausted",
                host=target.ascii_host,
                hops=hop,
                started=started,
                clock=clock,
            )
        per_attempt = min(settings.web_read_timeout_seconds, left)
        timeout = httpx.Timeout(
            min(10.0, per_attempt), read=per_attempt, write=per_attempt, pool=per_attempt
        )
        headers = {
            "Host": target.authority,
            "User-Agent": _USER_AGENT,
            "Accept": "text/html, application/xhtml+xml, text/plain",
            "Accept-Encoding": "identity",
        }
        extensions: dict[str, str] = {}
        if target.scheme == "https":
            extensions["sni_hostname"] = target.ascii_host
        pinned = httpx.URL(target.url).copy_with(host=address)
        verdict = "transport_error"
        bytes_seen = 0
        try:
            # Fresh client per attempt: pool isolation is part of the pinning
            # contract (see module docstring).
            with client_factory(timeout) as client:
                request = client.build_request(
                    "GET", pinned, headers=headers, extensions=extensions
                )
                response = client.send(request, stream=True)
                try:
                    result = _consume_response(
                        response,
                        target=target,
                        settings=settings,
                        clock=clock,
                        started=started,
                        hop=hop,
                        remaining=remaining,
                    )
                finally:
                    response.close()
                if isinstance(result, FetchOutcome):
                    verdict = result.error or "ok"
                    bytes_seen = result.bytes_fetched
                else:
                    verdict = "redirect"
                return result
        except httpx.TimeoutException:
            verdict = "timeout"
            return _refusal(
                ERROR_TIMEOUT,
                f"request to host {target.ascii_host!r} timed out",
                host=target.ascii_host,
                hops=hop,
                started=started,
                clock=clock,
            )
        except httpx.HTTPError as exc:
            # Connect/protocol failure against THIS vetted address — try the
            # next vetted one (never a re-resolution). Exception text can echo
            # the (pinned) request URL, so redact before keeping it.
            last_transport_error = cap_length(redact(f"{type(exc).__name__}: {exc}"))
            continue
        finally:
            # Exactly one attribution line per outbound request attempt.
            _log_fetch(
                attribution, target.ascii_host, hop, verdict, bytes_seen, started, clock
            )
    return _refusal(
        ERROR_TRANSPORT,
        f"host {target.ascii_host!r}: {last_transport_error}",
        host=target.ascii_host,
        hops=hop,
        started=started,
        clock=clock,
    )


def _consume_response(
    response: httpx.Response,
    *,
    target: ResearchUrl,
    settings: Settings,
    clock: Callable[[], float],
    started: float,
    hop: int,
    remaining: Callable[[], float],
) -> "FetchOutcome | str":
    status = response.status_code
    if status in _REDIRECT_STATUSES:
        locations = response.headers.get_list("location")
        if len(locations) != 1 or not locations[0].strip():
            return _refusal(
                ERROR_REDIRECT_INVALID,
                "missing, empty, or duplicate Location header",
                host=target.ascii_host,
                hops=hop,
                started=started,
                clock=clock,
            )
        # Resolve against the LOGICAL url — never the pinned-IP request URL.
        return urljoin(target.url, locations[0].strip())
    if not (200 <= status < 300):
        return _refusal(
            f"http_{status}",
            f"host {target.ascii_host!r} answered HTTP {status}",
            host=target.ascii_host,
            hops=hop,
            started=started,
            clock=clock,
        )

    encoding = response.headers.get("content-encoding", "").strip().lower()
    if encoding not in ("", "identity"):
        # Identity was requested; anything else re-opens the decompression
        # surface — refuse rather than bound it.
        return _refusal(
            ERROR_ENCODING_REFUSED,
            f"content-encoding {encoding!r} refused (identity required)",
            host=target.ascii_host,
            hops=hop,
            started=started,
            clock=clock,
        )
    media_type, charset = _parse_content_type(response.headers.get("content-type"))
    if media_type not in _ALLOWED_MIME:
        return _refusal(
            ERROR_MIME_REFUSED,
            f"content-type {media_type!r} is not readable text",
            host=target.ascii_host,
            hops=hop,
            started=started,
            clock=clock,
        )
    declared = response.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > settings.web_read_max_bytes:
        return _refusal(
            ERROR_TOO_LARGE,
            f"declared size exceeds {settings.web_read_max_bytes} bytes",
            host=target.ascii_host,
            hops=hop,
            started=started,
            clock=clock,
        )

    body = bytearray()
    for chunk in response.iter_raw(_STREAM_CHUNK_BYTES):
        body.extend(chunk)
        if len(body) > settings.web_read_max_bytes:
            return _refusal(
                ERROR_TOO_LARGE,
                f"body exceeds {settings.web_read_max_bytes} bytes",
                host=target.ascii_host,
                hops=hop,
                started=started,
                clock=clock,
            )
        if remaining() <= 0:
            return _refusal(
                ERROR_TIMEOUT,
                "total time budget exhausted mid-body",
                host=target.ascii_host,
                hops=hop,
                started=started,
                clock=clock,
            )

    if media_type == "text/plain":
        extracted = extract_plain_text(
            bytes(body), charset=charset, max_chars=settings.web_read_max_text_chars
        )
    else:
        extracted = extract_html_text(
            bytes(body), charset=charset, max_chars=settings.web_read_max_text_chars
        )
    return FetchOutcome(
        ok=True,
        error=None,
        error_detail="",
        text=extracted.text,
        title=extracted.title,
        truncated=extracted.truncated,
        final_url=target.url,
        host=target.ascii_host,
        bytes_fetched=len(body),
        hops=hop,
        duration_seconds=max(0.0, clock() - started),
    )


def _log_fetch(
    attribution: Attribution,
    host: str,
    hop: int,
    verdict: str,
    bytes_fetched: int,
    started: float,
    clock: Callable[[], float],
) -> None:
    # One attribution line per outbound request attempt. Host only — never the
    # URL, query, or Location. Fields are bounded identifiers + numbers, so the
    # line is injection-safe by construction.
    logger.info(
        "web_fetch feature=%s reason=%s host=%s hop=%d verdict=%s bytes=%d duration=%.2fs",
        attribution.feature,
        attribution.reason,
        host,
        hop,
        verdict,
        bytes_fetched,
        max(0.0, clock() - started),
    )
