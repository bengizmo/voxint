"""HttpLLMClient against httpx.MockTransport — contract validation, no network."""

import json
from typing import Any

import httpx
import pytest

from voxint.clients.base import EnhancementRequestSegment
from voxint.clients.llm import MAX_CHAT_REPLY_CHARS, ChatMessage, HttpLLMClient, LLMError

SEGMENTS = (
    EnhancementRequestSegment(segment_index=0, text="hello there", diarization_label="SPEAKER_00"),
    EnhancementRequestSegment(segment_index=1, text="im jane", diarization_label="SPEAKER_01"),
)


def make_client(handler: Any, api_key: str = "sk-test") -> HttpLLMClient:
    http = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    return HttpLLMClient("http://test", "test-model", api_key, 5.0, client=http)


def completion(content: Any) -> httpx.Response:
    body = content if isinstance(content, str) else json.dumps(content)
    return httpx.Response(200, json={"choices": [{"message": {"content": body}}]})


def reply(
    segments: list[dict[str, Any]], hints: list[dict[str, Any]] | None = None
) -> httpx.Response:
    payload: dict[str, Any] = {"segments": segments}
    if hints is not None:
        payload["name_hints"] = hints
    return completion(payload)


# ------------------------------------------------------------------ happy path


def test_enhances_and_returns_hints() -> None:
    result = make_client(
        lambda r: reply(
            [{"index": 0, "text": "Hello there."}, {"index": 1, "text": "I'm Jane."}],
            [{"label": "SPEAKER_01", "name": " Jane ", "kind": "self"}],
        )
    ).enhance_segments(SEGMENTS, "")
    assert result.enhanced == {0: "Hello there.", 1: "I'm Jane."}
    assert len(result.name_hints) == 1
    hint = result.name_hints[0]
    assert (hint.diarization_label, hint.name, hint.kind) == ("SPEAKER_01", "Jane", "self")


@pytest.mark.parametrize("template", ["```json\n{}\n```", "```\n{}\n```", "```json{}```"])
def test_markdown_fence_tolerated(template: str) -> None:
    inner = '{"segments": [{"index": 0, "text": "Hi."}, {"index": 1, "text": "Ok."}]}'
    fenced = template.format(inner)
    result = make_client(lambda r: completion(fenced)).enhance_segments(SEGMENTS, "")
    assert result.enhanced == {0: "Hi.", 1: "Ok."}


def test_nul_in_reply_rejects_batch() -> None:
    # Valid JSON, but PostgreSQL rejects NUL in text columns — persisting it
    # would turn best-effort enhancement into a stage failure.
    with pytest.raises(LLMError, match="NUL"):
        make_client(
            lambda r: reply([{"index": 0, "text": "bad\x00text"}, {"index": 1, "text": "b"}])
        ).enhance_segments(SEGMENTS, "")
    good = [{"index": 0, "text": "a"}, {"index": 1, "text": "b"}]
    with pytest.raises(LLMError, match="NUL"):
        make_client(
            lambda r: reply(good, [{"label": "SPEAKER_00", "name": "Jo\x00e", "kind": "self"}])
        ).enhance_segments(SEGMENTS, "")


def test_duplicate_request_indexes_rejected_before_sending() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected for an invalid batch")

    with pytest.raises(LLMError, match="duplicate segment indexes"):
        make_client(handler).enhance_segments((SEGMENTS[0], SEGMENTS[0]), "")


def test_reply_far_larger_than_source_rejects_batch() -> None:
    bloated = [{"index": 0, "text": "x" * 10_000}, {"index": 1, "text": "b"}]
    with pytest.raises(LLMError, match="not an enhancement"):
        make_client(lambda r: reply(bloated)).enhance_segments(SEGMENTS, "")


def test_empty_batch_short_circuits() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected for an empty batch")

    assert make_client(handler).enhance_segments((), "ctx").enhanced == {}


def test_request_carries_auth_model_context_and_segments() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return reply([{"index": 0, "text": "a"}, {"index": 1, "text": "b"}])

    make_client(handler).enhance_segments(SEGMENTS, "astronomy podcast")
    assert seen["auth"] == "Bearer sk-test"
    body = seen["body"]
    assert body["model"] == "test-model"
    assert body["temperature"] == 0
    assert "astronomy podcast" in body["messages"][0]["content"]
    sent = json.loads(body["messages"][1]["content"])
    assert sent == [
        {"index": 0, "label": "SPEAKER_00", "text": "hello there"},
        {"index": 1, "label": "SPEAKER_01", "text": "im jane"},
    ]


def test_no_auth_header_without_api_key() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return reply([{"index": 0, "text": "a"}, {"index": 1, "text": "b"}])

    make_client(handler, api_key="").enhance_segments(SEGMENTS, "")
    assert seen["auth"] is None


# ------------------------------------------------------- alignment enforcement


@pytest.mark.parametrize(
    "segments_reply",
    [
        [{"index": 0, "text": "only one"}],  # missing index
        [{"index": 0, "text": "a"}, {"index": 0, "text": "b"}],  # duplicate
        [{"index": 0, "text": "a"}, {"index": 5, "text": "b"}],  # unknown
        [{"index": "0", "text": "a"}, {"index": 1, "text": "b"}],  # str index
        [{"index": True, "text": "a"}, {"index": 1, "text": "b"}],  # bool index
        [{"index": 0, "text": 7}, {"index": 1, "text": "b"}],  # non-str text
        ["not-an-object", {"index": 1, "text": "b"}],
    ],
)
def test_misaligned_reply_rejects_whole_batch(segments_reply: list[Any]) -> None:
    with pytest.raises(LLMError):
        make_client(lambda r: reply(segments_reply)).enhance_segments(SEGMENTS, "")


@pytest.mark.parametrize(
    "hints_reply",
    [
        [{"label": "SPEAKER_99", "name": "Jane", "kind": "self"}],  # unknown label
        [{"label": "SPEAKER_00", "name": "Jane", "kind": "guess"}],  # bad kind
        [{"label": "SPEAKER_00", "name": "   ", "kind": "self"}],  # blank name
        [{"label": 3, "name": "Jane", "kind": "self"}],  # non-str label
        ["not-an-object"],
    ],
)
def test_invalid_hints_reject_batch(hints_reply: list[Any]) -> None:
    good = [{"index": 0, "text": "a"}, {"index": 1, "text": "b"}]
    with pytest.raises(LLMError):
        make_client(lambda r: reply(good, hints_reply)).enhance_segments(SEGMENTS, "")


# -------------------------------------------------------------- failure shapes


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(429, text="rate limited"),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]}),
        completion([1, 2, 3]),  # JSON but not an object
        completion({"no_segments": []}),
        completion({"segments": "not-a-list"}),
        completion(
            {
                "segments": [{"index": 0, "text": "a"}, {"index": 1, "text": "b"}],
                "name_hints": "x",
            }
        ),
    ],
)
def test_bad_responses_raise_llm_error(response: httpx.Response) -> None:
    with pytest.raises(LLMError):
        make_client(lambda r: response).enhance_segments(SEGMENTS, "")


def test_transport_failure_raises_llm_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError):
        make_client(handler).enhance_segments(SEGMENTS, "")


# ------------------------------------------------------------------ chat_json


def test_chat_json_returns_object_and_sends_messages() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return completion({"actions": [{"tool": "web_search", "query": "x"}]})

    result = make_client(handler).chat_json(
        [ChatMessage(role="system", content="rules"), ChatMessage(role="user", content="go")]
    )
    assert result == {"actions": [{"tool": "web_search", "query": "x"}]}
    payload = json.loads(seen[0].content)
    assert payload["temperature"] == 0
    assert payload["messages"] == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "go"},
    ]


def test_chat_json_tolerates_fence_and_requires_object() -> None:
    fenced = make_client(lambda r: completion('```json\n{"ok": true}\n```'))
    assert fenced.chat_json([ChatMessage(role="user", content="go")]) == {"ok": True}
    array = make_client(lambda r: completion("[1, 2]"))
    with pytest.raises(LLMError, match="expected a JSON object"):
        array.chat_json([ChatMessage(role="user", content="go")])


def test_chat_json_rejects_oversize_nul_and_empty_input() -> None:
    huge = make_client(lambda r: completion("x" * (MAX_CHAT_REPLY_CHARS + 1)))
    with pytest.raises(LLMError, match="bound"):
        huge.chat_json([ChatMessage(role="user", content="go")])
    nul = make_client(lambda r: completion('{"a": "b\x00c"}'))
    with pytest.raises(LLMError, match="NUL"):
        nul.chat_json([ChatMessage(role="user", content="go")])
    with pytest.raises(LLMError, match="at least one message"):
        make_client(lambda r: completion("{}")).chat_json([])


def test_chat_json_surfaces_http_and_transport_failures() -> None:
    http_error = make_client(lambda r: httpx.Response(500, text="boom"))
    with pytest.raises(LLMError, match="HTTP 500"):
        http_error.chat_json([ChatMessage(role="user", content="go")])

    def explode(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope", request=request)

    transport = make_client(explode)
    with pytest.raises(LLMError, match="transport failure"):
        transport.chat_json([ChatMessage(role="user", content="go")])


def test_chat_message_rejects_unknown_role() -> None:
    with pytest.raises(ValueError, match="unknown chat role"):
        ChatMessage(role="tool", content="x")
