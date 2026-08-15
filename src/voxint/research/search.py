"""``web_search`` — pluggable provider interface + the SearxNG provider (#39).

The provider base URL is operator-configured egress (a LAN SearxNG is the
expected deployment) — the same trust class as ``llm_base_url``, deliberately
NOT subject to the public-address policy. Everything a provider RETURNS is
untrusted: result URLs pass the shared string gate before they are surfaced
(non-conforming results are dropped and counted, not silently vanished), and
titles/snippets are sanitized and length-capped. ``read_url`` then re-applies
the full resolved-address policy before any result is ever fetched.

Provider responses are parsed strictly and boundedly (the llm-client idiom):
non-dict JSON is refused, field lengths are capped, malformed entries are
skipped. The provider HTTP client never follows redirects and never trusts
ambient proxy env. The provider credential (``web_search_api_key``) is sent as
a header when configured and treated as a secret everywhere else — it joins
the redaction pass on every surfaced error.
"""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from voxint.config import Settings
from voxint.media.redaction import cap_length, redact
from voxint.research.budget import Attribution, ResearchBudget
from voxint.research.extract import sanitize_text
from voxint.research.policy import UrlPolicyError, gate_research_url

logger = logging.getLogger(__name__)

ERROR_DISABLED = "disabled"
ERROR_INVALID_INPUT = "invalid_input"
ERROR_BUDGET_EXHAUSTED = "budget_exhausted"
ERROR_PROVIDER = "provider_error"

_MAX_QUERY_BYTES = 512
_MAX_TITLE_CHARS = 150
_MAX_SNIPPET_CHARS = 300
# A sane provider answer is a few KiB of JSON; refuse a response body larger
# than this outright rather than parse an unbounded document.
_MAX_PROVIDER_RESPONSE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class SearchOutcome:
    ok: bool
    error: str | None
    error_detail: str  # redacted; never the query, a URL, or the credential
    results: tuple[SearchResult, ...] = field(default_factory=tuple)
    dropped_results: int = 0  # provider entries refused by the URL gate/shape
    duration_seconds: float = 0.0


class SearchProvider(Protocol):
    """The pluggable provider contract: one bounded query → normalized results.

    Implementations raise :class:`ProviderError` for any transport, status, or
    response-shape failure; ``web_search`` turns that into a structured
    outcome. ``dropped_last`` reports how many entries of the LAST search were
    refused by the URL gate or shape checks (0 when none) — part of the
    protocol so "dropped and counted, not silently vanished" holds for every
    provider, not just the built-in. Implementations must self-bound their
    request time to the ``timeout_seconds`` they were constructed with.
    """

    name: str
    dropped_last: int

    def search(self, query: str, *, max_results: int) -> list[SearchResult]: ...


class ProviderError(Exception):
    """A provider call failed (transport, status, or malformed response).

    Messages must already be safe to surface: no query text, no URLs, no
    credential material — name the provider and the failure class only.
    """


def _normalize_entries(
    entries: list[Any], *, max_results: int
) -> tuple[list[SearchResult], int]:
    """Bounded, strict normalization shared by providers.

    Returns (kept, dropped): entries that are not dicts, lack a usable
    title/URL, or whose URL fails the shared string gate are DROPPED and
    counted — a provider must never surface a target the egress policy would
    refuse to read.
    """
    kept: list[SearchResult] = []
    dropped = 0
    seen_urls: set[str] = set()
    for entry in entries:
        if len(kept) >= max_results:
            break
        if not isinstance(entry, dict):
            dropped += 1
            continue
        raw_url = entry.get("url")
        raw_title = entry.get("title")
        if not isinstance(raw_url, str) or not isinstance(raw_title, str):
            dropped += 1
            continue
        try:
            # The FULL research gate (not just the base string gate): a result
            # read_url would always refuse (%-in-host, non-IDNA) must never be
            # surfaced as usable, and the fragment-free canonical URL also
            # dedupes #frag variants of one page.
            gated_url = gate_research_url(raw_url).url
        except (UrlPolicyError, ValueError):
            dropped += 1
            continue
        if gated_url in seen_urls:
            continue
        raw_snippet = entry.get("content")
        snippet = raw_snippet if isinstance(raw_snippet, str) else ""
        seen_urls.add(gated_url)
        kept.append(
            SearchResult(
                title=sanitize_text(raw_title)[:_MAX_TITLE_CHARS],
                url=gated_url,
                snippet=sanitize_text(snippet)[:_MAX_SNIPPET_CHARS],
            )
        )
    return kept, dropped


class SearxngProvider:
    """SearxNG's JSON API: ``GET {base}/search?q=...&format=json``."""

    name = "searxng"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        timeout_seconds: float,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._client = client
        self.dropped_last = 0

    def _bounded_get(self, url: str, params: dict[str, str]) -> bytes:
        """GET with the body STREAMED against the size bound — a misbehaving
        provider must not force an unbounded allocation before the check."""
        headers = {"Accept": "application/json"}
        if self._api_key:
            # SearxNG deployments behind an auth proxy commonly key on this.
            headers["X-API-Key"] = self._api_key
        if self._client is not None:
            client = self._client
            owns_client = False
        else:
            client = httpx.Client(
                follow_redirects=False,
                trust_env=False,
                timeout=httpx.Timeout(min(10.0, self._timeout), read=self._timeout,
                                      write=self._timeout, pool=self._timeout),
            )
            owns_client = True
        try:
            request = client.build_request("GET", url, params=params, headers=headers)
            response = client.send(request, stream=True)
            try:
                if response.status_code != 200:
                    raise ProviderError(f"searxng answered HTTP {response.status_code}")
                declared = response.headers.get("content-length")
                if (
                    declared is not None
                    and declared.isdigit()
                    and int(declared) > _MAX_PROVIDER_RESPONSE_BYTES
                ):
                    raise ProviderError("searxng response exceeds the size bound")
                body = bytearray()
                for chunk in response.iter_raw(65536):
                    body.extend(chunk)
                    if len(body) > _MAX_PROVIDER_RESPONSE_BYTES:
                        raise ProviderError("searxng response exceeds the size bound")
                return bytes(body)
            finally:
                response.close()
        finally:
            if owns_client:
                client.close()

    def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        try:
            raw = self._bounded_get(
                f"{self._base_url}/search", {"q": query, "format": "json"}
            )
        except httpx.TimeoutException:
            raise ProviderError("searxng request timed out") from None
        except httpx.InvalidURL:
            # Not an HTTPError subclass; a malformed operator base URL must
            # still surface as a structured provider failure, never a raw
            # exception carrying the query-bearing request URL.
            raise ProviderError("searxng base URL is malformed") from None
        except httpx.HTTPError as exc:
            # httpx error text can echo the request URL (which carries the
            # query) — surface the exception CLASS only.
            raise ProviderError(
                f"searxng transport failure ({type(exc).__name__})"
            ) from None
        try:
            data = json.loads(raw)
        except ValueError:
            raise ProviderError("searxng response is not valid JSON") from None
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise ProviderError("searxng response has an unexpected shape")
        kept, dropped = _normalize_entries(data["results"], max_results=max_results)
        self.dropped_last = dropped
        return kept


def provider_from_settings(
    settings: Settings,
    *,
    client: httpx.Client | None = None,
    timeout_seconds: float | None = None,
) -> SearchProvider:
    """Build the configured provider (the Literal in Settings bounds the set).

    ``timeout_seconds`` overrides the settings timeout — ``web_search`` uses it
    to clamp the provider call to the budget's remaining wall clock.
    """
    return SearxngProvider(
        base_url=settings.web_search_base_url,
        api_key=settings.web_search_api_key,
        timeout_seconds=(
            settings.web_search_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        ),
        client=client,
    )


def web_search(
    query: str,
    *,
    settings: Settings,
    budget: ResearchBudget,
    attribution: Attribution,
    provider: SearchProvider | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> SearchOutcome:
    """One bounded provider query → normalized, egress-gated results.

    Consumes one search-budget unit per invocation. All failures are
    structured outcomes; the query never appears in any error or log line.
    """
    if not settings.voxint_web_research:
        return SearchOutcome(
            ok=False, error=ERROR_DISABLED, error_detail="web research is disabled"
        )
    stripped = query.strip()
    if not stripped or len(stripped.encode("utf-8")) > _MAX_QUERY_BYTES:
        return SearchOutcome(
            ok=False,
            error=ERROR_INVALID_INPUT,
            error_detail=f"query must be 1..{_MAX_QUERY_BYTES} bytes",
        )
    if not budget.try_consume_search():
        return SearchOutcome(
            ok=False, error=ERROR_BUDGET_EXHAUSTED, error_detail="search budget exhausted"
        )
    started = clock()
    if provider is not None:
        chosen = provider  # injected providers self-bound their timeout
    else:
        # Clamp the provider timeout to the budget's remaining wall clock so a
        # call started near expiry cannot run the full configured timeout
        # (try_consume_search above already refused an EXPIRED budget).
        timeout = settings.web_search_timeout_seconds
        budget_left = budget.remaining_seconds()
        if budget_left is not None:
            timeout = max(0.1, min(timeout, budget_left))
        chosen = provider_from_settings(settings, timeout_seconds=timeout)
    verdict = "provider_error"
    result_count = 0
    try:
        results = chosen.search(stripped, max_results=settings.web_search_max_results)
        result_count = len(results)
        verdict = "ok"
        dropped = getattr(chosen, "dropped_last", 0)
        return SearchOutcome(
            ok=True,
            error=None,
            error_detail="",
            results=tuple(results),
            dropped_results=int(dropped),
            duration_seconds=max(0.0, clock() - started),
        )
    except ProviderError as exc:
        detail = cap_length(redact(str(exc), extra_secrets=(settings.web_search_api_key,)))
        return SearchOutcome(
            ok=False,
            error=ERROR_PROVIDER,
            error_detail=detail,
            duration_seconds=max(0.0, clock() - started),
        )
    finally:
        # One attribution line per provider call. The QUERY is deliberately
        # absent — operator search terms stay out of the shared log stream.
        logger.info(
            "web_search feature=%s reason=%s provider=%s verdict=%s results=%d duration=%.2fs",
            attribution.feature,
            attribution.reason,
            chosen.name,
            verdict,
            result_count,
            max(0.0, clock() - started),
        )
