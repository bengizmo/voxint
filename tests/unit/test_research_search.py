"""web_search provider interface + SearxNG normalization hygiene (issue #39)."""

import json

import httpx
import pytest

from voxint.config import Settings
from voxint.research.budget import Attribution, ResearchBudget
from voxint.research.search import (
    ProviderError,
    SearchResult,
    SearxngProvider,
    web_search,
)

ATTR = Attribution(feature="test", reason="unit-test")
SECRET_KEY = "PROVIDERSECRETKEY"


def make_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "voxint_web_research": True,
        "web_search_base_url": "http://searx.lan:8888",
        "web_search_api_key": SECRET_KEY,
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)  # type: ignore[arg-type]


def budget(searches: int = 5) -> ResearchBudget:
    return ResearchBudget(max_searches=searches, max_reads=5)


def provider_with(
    handler_payload: object, *, status: int = 200, settings: Settings | None = None
) -> SearxngProvider:
    settings = settings or make_settings()
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        # Streamed body (iterator content): the provider reads via iter_raw,
        # and a bytes-content MockTransport response would be pre-consumed.
        if isinstance(handler_payload, bytes):
            body = handler_payload
        elif isinstance(handler_payload, str):
            body = handler_payload.encode()
        else:
            body = json.dumps(handler_payload).encode()
        return httpx.Response(status, content=iter([body]))

    provider = SearxngProvider(
        base_url=settings.web_search_base_url,
        api_key=settings.web_search_api_key,
        timeout_seconds=settings.web_search_timeout_seconds,
        client=httpx.Client(
            transport=httpx.MockTransport(handler),
            follow_redirects=False,
            trust_env=False,
        ),
    )
    provider.requests_seen = seen  # type: ignore[attr-defined]
    return provider


def test_normalizes_dedupes_and_caps_results() -> None:
    payload = {
        "results": [
            {"title": "A" * 500, "url": "https://a.example.com/1", "content": "s" * 999},
            {"title": "dup", "url": "https://a.example.com/1", "content": "again"},
            {"title": "B", "url": "https://b.example.com/2", "content": "ok"},
        ]
    }
    provider = provider_with(payload)
    out = web_search(
        "podcast host name", settings=make_settings(), budget=budget(),
        attribution=ATTR, provider=provider,
    )
    assert out.ok is True
    assert [r.url for r in out.results] == [
        "https://a.example.com/1", "https://b.example.com/2",
    ]
    assert len(out.results[0].title) == 150  # capped
    assert len(out.results[0].snippet) == 300  # capped


def test_non_conforming_result_urls_are_dropped_and_counted() -> None:
    payload = {
        "results": [
            {"title": "private", "url": "http://127.0.0.1/admin", "content": ""},
            {"title": "scheme", "url": "ftp://example.com/f", "content": ""},
            {"title": "creds", "url": "http://u:p@example.com/", "content": ""},
            {"title": "not-a-dict-follows", "url": "https://ok.example.com/", "content": ""},
            "just a string",
            {"title": 42, "url": "https://x.example.com/", "content": ""},
        ]
    }
    provider = provider_with(payload)
    out = web_search(
        "query", settings=make_settings(), budget=budget(),
        attribution=ATTR, provider=provider,
    )
    assert out.ok is True
    assert [r.url for r in out.results] == ["https://ok.example.com/"]
    assert out.dropped_results == 5


def test_hostile_titles_and_snippets_are_sanitized() -> None:
    payload = {
        "results": [
            {
                "title": "clean​title\U000e0041",  # zero-width + tag block
                "url": "https://a.example.com/",
                "content": "snip‮pet",  # bidi override
            }
        ]
    }
    provider = provider_with(payload)
    out = web_search(
        "q", settings=make_settings(), budget=budget(), attribution=ATTR,
        provider=provider,
    )
    result = out.results[0]
    assert result.title == "cleantitle"
    assert result.snippet == "snippet"


def test_api_key_sent_as_header_never_in_url_or_errors() -> None:
    provider = provider_with({"results": []})
    web_search(
        "q", settings=make_settings(), budget=budget(), attribution=ATTR,
        provider=provider,
    )
    request = provider.requests_seen[0]  # type: ignore[attr-defined]
    assert request.headers["x-api-key"] == SECRET_KEY
    assert SECRET_KEY not in str(request.url)


@pytest.mark.parametrize(
    "payload_kwargs",
    [
        {"handler_payload": "not json at all"},
        {"handler_payload": ["top-level-list"]},
        {"handler_payload": {"no_results_key": True}},
        {"handler_payload": {"results": []}, "status": 503},
    ],
)
def test_malformed_or_failing_provider_is_structured_error(
    payload_kwargs: dict[str, object],
) -> None:
    provider = provider_with(**payload_kwargs)  # type: ignore[arg-type]
    out = web_search(
        "sensitive query text", settings=make_settings(), budget=budget(),
        attribution=ATTR, provider=provider,
    )
    if payload_kwargs.get("status") == 503 or "results" not in str(payload_kwargs):
        assert out.ok is False or out.results == ()
    if not out.ok:
        assert out.error == "provider_error"
        assert "sensitive query text" not in out.error_detail
        assert SECRET_KEY not in out.error_detail


def test_oversized_provider_response_is_refused() -> None:
    huge = json.dumps({"results": [], "pad": "x" * (1024 * 1024 + 10)})
    provider = provider_with(huge)
    out = web_search(
        "q", settings=make_settings(), budget=budget(), attribution=ATTR,
        provider=provider,
    )
    assert out.error == "provider_error"
    assert "size bound" in out.error_detail


def test_disabled_and_budget_and_query_bounds() -> None:
    out = web_search(
        "q", settings=Settings(_env_file=None), budget=budget(), attribution=ATTR,
        provider=provider_with({"results": []}),
    )
    assert out.error == "disabled"

    b = budget(searches=0)
    out2 = web_search(
        "q", settings=make_settings(), budget=b, attribution=ATTR,
        provider=provider_with({"results": []}),
    )
    assert out2.error == "budget_exhausted"

    out3 = web_search(
        "", settings=make_settings(), budget=budget(), attribution=ATTR,
        provider=provider_with({"results": []}),
    )
    assert out3.error == "invalid_input"
    out4 = web_search(
        "x" * 600, settings=make_settings(), budget=budget(), attribution=ATTR,
        provider=provider_with({"results": []}),
    )
    assert out4.error == "invalid_input"


def test_max_results_cap_is_enforced() -> None:
    payload = {
        "results": [
            {"title": f"t{i}", "url": f"https://r{i}.example.com/", "content": ""}
            for i in range(50)
        ]
    }
    provider = provider_with(payload)
    out = web_search(
        "q", settings=make_settings(web_search_max_results=3), budget=budget(),
        attribution=ATTR, provider=provider,
    )
    assert len(out.results) == 3


def test_query_never_in_attribution_log(caplog: pytest.LogCaptureFixture) -> None:
    provider = provider_with({"results": []})
    with caplog.at_level("INFO", logger="voxint.research.search"):
        web_search(
            "operator secret research terms", settings=make_settings(),
            budget=budget(), attribution=ATTR, provider=provider,
        )
    lines = [r.getMessage() for r in caplog.records if "web_search" in r.getMessage()]
    assert lines, "provider calls must log an attribution line"
    for line in lines:
        assert "operator secret research terms" not in line
        assert "provider=searxng" in line


def test_provider_error_raised_directly_names_no_query() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot reach {request.url}")

    provider = SearxngProvider(
        base_url="http://searx.lan:8888",
        timeout_seconds=5.0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ProviderError) as exc:
        provider.search("the query", max_results=5)
    assert "the query" not in str(exc.value)


def test_search_result_is_frozen_contract() -> None:
    r = SearchResult(title="t", url="https://e.example.com/", snippet="s")
    with pytest.raises(AttributeError):
        r.title = "changed"  # type: ignore[misc]


def test_percent_host_and_fragment_results_use_the_full_research_gate() -> None:
    # Results the reader would always refuse (%-in-host) must be dropped, and
    # #frag variants of one page must dedupe to the fragment-free canonical
    # URL (review finding: search used a weaker gate than read_url).
    payload = {
        "results": [
            {"title": "pct", "url": "http://ex%61mple.com/x", "content": ""},
            {"title": "frag-a", "url": "https://ok.example.com/p#a", "content": ""},
            {"title": "frag-b", "url": "https://ok.example.com/p#b", "content": ""},
        ]
    }
    provider = provider_with(payload)
    out = web_search(
        "q", settings=make_settings(), budget=budget(), attribution=ATTR,
        provider=provider,
    )
    assert [r.url for r in out.results] == ["https://ok.example.com/p"]
    assert out.dropped_results == 1  # the percent-host entry
