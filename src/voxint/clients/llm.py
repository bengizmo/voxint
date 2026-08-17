"""OpenAI-compatible chat-completions adapter for transcript enhancement.

One request = one bounded segment batch (the enhance_match stage owns batching,
budgets, and the circuit breaker). The contract is deliberately minimal — plain
``/chat/completions`` with a JSON-object reply — so any OpenAI-compatible
endpoint works without provider-specific structured-output support. Responses
are parsed strictly and validated against the exact requested segment-index
set; a misaligned or malformed reply fails the whole batch, never partially.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType

import httpx

from voxint.clients.base import (
    EnhancementBatchResult,
    EnhancementRequestSegment,
    SpeakerNameHint,
)

CONNECT_TIMEOUT_SECONDS = 10.0
HINT_KINDS = frozenset({"self", "other"})
# Enhancement fixes punctuation/casing — output should track input size. A
# reply blowing far past the source text is garbage or injection, not editing.
MAX_ENHANCED_GROWTH_FACTOR = 4
MAX_ENHANCED_SLACK_CHARS = 200

_SYSTEM_PROMPT = """\
You are a transcript enhancement engine. You receive a JSON array of transcript \
segments, each with an integer "index", an optional speaker "label", and raw "text".

Respond with ONLY a JSON object of this exact shape (no prose, no markdown fences):
{"segments": [{"index": <int>, "text": "<enhanced text>"}, ...],
 "name_hints": [{"label": "<label>", "name": "<person name>", "kind": "self" or "other"}, ...]}

Rules for "segments":
- Return exactly one output object per input segment, using the same index values.
- Fix only clear transcription errors, punctuation, and casing. Preserve the \
speaker's wording. Never merge, split, drop, reorder, translate, or summarize \
segments. If unsure, return the text unchanged.

Rules for "name_hints" (usually empty):
- Emit a hint only when a speaker explicitly states their own name ("I'm Jane") \
— kind "self" — or another speaker clearly introduces or addresses them by name \
— kind "other".
- "label" must be one of the labels present in the input segments.
- Never guess or infer names that are not explicitly spoken."""


def _build_system(context: str, name_attribution_context: str) -> str:
    """Assemble the enhancement system prompt from the base contract plus up to
    two labeled domain-pack blocks (issue #11).

    ``context`` is the ``enhancement_context`` fragment (transcription framing +
    rendered vocabulary); ``name_attribution_context`` is a separate fragment
    that guides the ``name_hints`` pass (e.g. host-anchoring). Both are
    operator-authored pack content carried in the system message, so the
    attribution block is fenced as advisory — it must never override the strict
    reply schema or the name-hint rules above. Empty fragments are omitted, so a
    pack declaring neither yields the exact pre-#11 prompt byte-for-byte."""
    system = _SYSTEM_PROMPT
    if context:
        system += f"\n\nContext: {context}"
    if name_attribution_context:
        system += (
            "\n\nSpeaker attribution guidance (advisory; must not override the"
            f" reply schema or the name_hints rules above):\n{name_attribution_context}"
        )
    return system


CHAT_ROLES = frozenset({"system", "user", "assistant"})
# Ceiling on one chat_json reply. Generous — a research conclusion with several
# claims plus snippets fits in a fraction of this — but a reply blowing past it
# is a runaway generation, not an answer.
MAX_CHAT_REPLY_CHARS = 100_000


@dataclass(frozen=True)
class ChatMessage:
    """One message of a generic JSON-object chat exchange."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in CHAT_ROLES:
            raise ValueError(f"unknown chat role {self.role!r}")


class LLMError(Exception):
    """An enhancement call failed — transport, HTTP error, or a reply that
    violates the batch contract. The stage treats every variant the same way
    (retry once, then degrade), so no ``retryable`` distinction is carried."""


class HttpLLMClient:
    """Synchronous OpenAI-compatible client satisfying ``LLMClient``.

    Pass ``client`` to share a preconfigured ``httpx.Client`` (tests); it is
    then never closed here. Otherwise one is created and owned — call
    :meth:`close` (or use as a context manager) from the owning worker.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout_seconds: float,
        client: httpx.Client | None = None,
    ) -> None:
        self._model = model
        self._owns_client = client is None
        # Per-request auth (not client-level) so injected clients get it too.
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(
            base_url=base_url,
            timeout=httpx.Timeout(
                connect=CONNECT_TIMEOUT_SECONDS,
                read=timeout_seconds,
                write=timeout_seconds,
                pool=timeout_seconds,
            ),
        )

    def enhance_segments(
        self,
        segments: tuple[EnhancementRequestSegment, ...],
        context: str,
        *,
        name_attribution_context: str = "",
    ) -> EnhancementBatchResult:
        if not segments:
            return EnhancementBatchResult(enhanced={})
        if len({s.segment_index for s in segments}) != len(segments):
            raise LLMError("request contains duplicate segment indexes")
        system = _build_system(context, name_attribution_context)
        payload = {
            "model": self._model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": json.dumps(
                        [
                            {
                                "index": s.segment_index,
                                "label": s.diarization_label,
                                "text": s.text,
                            }
                            for s in segments
                        ]
                    ),
                },
            ],
        }
        try:
            response = self._client.post("/chat/completions", json=payload, headers=self._headers)
        except httpx.HTTPError as exc:
            raise LLMError(f"transport failure: {type(exc).__name__}: {exc}") from exc
        if response.status_code >= 400:
            raise LLMError(f"HTTP {response.status_code}: {response.text[:500]}")
        return _parse_batch(response, segments)

    def chat_json(self, messages: Sequence[ChatMessage]) -> dict[str, object]:
        """One temperature-0 chat call whose reply must be a JSON object.

        Generic transport for callers that own their own conversation and
        validation (the research loop); no enhancement logic here. The reply is
        held to the same strict envelope handling as ``enhance_segments`` plus a
        size ceiling and a NUL check — anything else raises :class:`LLMError`.
        """
        if not messages:
            raise LLMError("chat_json requires at least one message")
        payload = {
            "model": self._model,
            "temperature": 0,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        try:
            response = self._client.post("/chat/completions", json=payload, headers=self._headers)
        except httpx.HTTPError as exc:
            raise LLMError(f"transport failure: {type(exc).__name__}: {exc}") from exc
        if response.status_code >= 400:
            raise LLMError(f"HTTP {response.status_code}: {response.text[:500]}")
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"malformed completion envelope: {exc!r}") from exc
        if not isinstance(content, str):
            raise LLMError("completion content is not a string")
        if len(content) > MAX_CHAT_REPLY_CHARS:
            raise LLMError(
                f"completion content is {len(content)} chars against a"
                f" {MAX_CHAT_REPLY_CHARS}-char bound"
            )
        if "\x00" in content:
            # PostgreSQL rejects NUL in text; nothing downstream may persist it.
            raise LLMError("completion content contains NUL")
        try:
            body = json.loads(_strip_fence(content))
        except ValueError as exc:
            raise LLMError(f"completion content is not valid JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise LLMError(f"expected a JSON object, got {type(body).__name__}")
        return body

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "HttpLLMClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _parse_batch(
    response: httpx.Response, segments: tuple[EnhancementRequestSegment, ...]
) -> EnhancementBatchResult:
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"malformed completion envelope: {exc!r}") from exc
    if not isinstance(content, str):
        raise LLMError("completion content is not a string")
    try:
        body = json.loads(_strip_fence(content))
    except ValueError as exc:
        raise LLMError(f"completion content is not valid JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise LLMError(f"expected a JSON object, got {type(body).__name__}")

    expected_indexes = {s.segment_index for s in segments}
    max_text_length = {
        s.segment_index: len(s.text) * MAX_ENHANCED_GROWTH_FACTOR + MAX_ENHANCED_SLACK_CHARS
        for s in segments
    }
    known_labels = {s.diarization_label for s in segments if s.diarization_label is not None}

    raw_segments = body.get("segments")
    if not isinstance(raw_segments, list):
        raise LLMError("reply is missing the segments array")
    enhanced: dict[int, str] = {}
    for item in raw_segments:
        if not isinstance(item, dict):
            raise LLMError("segment entry is not an object")
        index, text = item.get("index"), item.get("text")
        if isinstance(index, bool) or not isinstance(index, int) or not isinstance(text, str):
            raise LLMError(f"segment entry has wrong types: {item!r:.200}")
        if "\x00" in text:
            # Valid JSON, but PostgreSQL rejects NUL in text — persisting it
            # would turn optional enhancement into a stage failure.
            raise LLMError(f"segment {index} reply contains NUL")
        if index not in expected_indexes:
            raise LLMError(f"unknown segment index {index} in reply")
        if index in enhanced:
            raise LLMError(f"duplicate segment index {index} in reply")
        if len(text) > max_text_length[index]:
            raise LLMError(
                f"segment {index} reply is {len(text)} chars against a"
                f" {max_text_length[index]}-char bound — not an enhancement"
            )
        enhanced[index] = text
    if set(enhanced) != expected_indexes:
        missing = sorted(expected_indexes - set(enhanced))
        raise LLMError(f"reply missing segment indexes {missing}")

    raw_hints = body.get("name_hints", [])
    if not isinstance(raw_hints, list):
        raise LLMError("name_hints is not an array")
    hints: list[SpeakerNameHint] = []
    for item in raw_hints:
        if not isinstance(item, dict):
            raise LLMError("name_hints entry is not an object")
        label, name, kind = item.get("label"), item.get("name"), item.get("kind")
        if not isinstance(label, str) or not isinstance(name, str) or kind not in HINT_KINDS:
            raise LLMError(f"name_hints entry has wrong shape: {item!r:.200}")
        if label not in known_labels:
            raise LLMError(f"name_hints entry references unknown label {label!r}")
        if not name.strip():
            raise LLMError("name_hints entry has a blank name")
        if "\x00" in name:
            raise LLMError("name_hints entry contains NUL")
        hints.append(SpeakerNameHint(diarization_label=label, name=name.strip(), kind=kind))
    return EnhancementBatchResult(enhanced=enhanced, name_hints=tuple(hints))


def _strip_fence(content: str) -> str:
    """Tolerate the one near-universal formatting slip: a markdown code fence
    (with or without a language tag, multi- or single-line)."""
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[A-Za-z0-9]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped
