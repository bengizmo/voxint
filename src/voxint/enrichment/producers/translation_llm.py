"""LLM translation producer (#133): batched, index-echoed, fail-closed.

One generation is a sequence of strict-JSON ``chat_json`` calls over the frozen
line snapshot. Each batch sends numbered lines and demands an object envelope
(``chat_json`` rejects a bare array) whose entries echo the line numbers back:

    {"translations": [{"i": 12, "text": "..."}, ...]}

Validation is an **exact index set** — no missing, duplicate, or unknown
entries — which catches merged, dropped, AND reordered lines, where a
positional count check could not. There is deliberately no fuzzy realignment:
a misaligned reply fails the whole batch, never partially (the enhancement
doctrine). The failure ladder is retry (``llm_attempts_per_batch``), then
recursive bisection down to single-line batches — the best-ROI recovery for
small local models that lose track of long numbered lists — and an
irreducible bad line fails the generation.

Empty/whitespace-only source lines are translated deterministically to
themselves without ever reaching the model.
"""

import logging
from collections.abc import Callable, Sequence
from typing import Protocol

from voxint.clients.llm import ChatMessage, LLMError
from voxint.config import Settings
from voxint.enrichment.translations import (
    TranslationLineSource,
    translated_size_ceiling,
)

logger = logging.getLogger(__name__)

PRODUCER_NAME = "translation.llm"
PRODUCER_VERSION = "1"
PROMPT_VERSION = 1
CONFIG_SCHEMA_VERSION = 1


class TranslationProducerError(Exception):
    """The model's reply could not be turned into a valid translation."""


class TranslationCancelled(Exception):
    """The cooperative cancel flag was observed between LLM calls."""


class ChatJsonLLM(Protocol):
    """The only capability the producer needs from a client (injection seam)."""

    def chat_json(self, messages: list[ChatMessage]) -> dict[str, object]: ...


_SYSTEM = (
    "You are a professional transcript translator. Reply with a single JSON"
    " object and nothing else — no prose, no markdown fences. Translate"
    " faithfully: preserve meaning, tone, register, names, numbers, and units."
    " Never merge, drop, reorder, summarize, or add lines."
)


def _language_phrase(code_label: str | None) -> str:
    return code_label if code_label is not None else "the source language"


def _batch_prompt(
    batch: Sequence[TranslationLineSource],
    *,
    source_label: str | None,
    target_label: str,
) -> list[ChatMessage]:
    numbered = "\n".join(f"[{line.line_index}] {line.text}" for line in batch)
    instruction = (
        f"Translate the following {len(batch)} numbered transcript lines from"
        f" {_language_phrase(source_label)} to {target_label}. Reply exactly as:"
        ' {"translations": [{"i": <line number>, "text": "<translation>"}, ...]}'
        " — one entry per input line, echoing each line's exact number, nothing"
        " else.\n\nLines:\n" + numbered
    )
    return [
        ChatMessage(role="system", content=_SYSTEM),
        ChatMessage(role="user", content=instruction),
    ]


def _parse_batch_reply(
    body: dict[str, object], batch: Sequence[TranslationLineSource]
) -> dict[int, str]:
    """Fail-closed reply validation: exact index set, bounded strings."""
    entries = body.get("translations")
    if not isinstance(entries, list):
        raise TranslationProducerError("reply has no 'translations' array")
    by_index = {line.line_index: line for line in batch}
    out: dict[int, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise TranslationProducerError("a translations entry is not an object")
        index = entry.get("i")
        # bool is an int subclass — `true` must not pass as index 1.
        if isinstance(index, bool) or not isinstance(index, int):
            raise TranslationProducerError("a translations entry has a non-integer 'i'")
        if index not in by_index:
            raise TranslationProducerError(f"reply names unknown line {index}")
        if index in out:
            raise TranslationProducerError(f"reply repeats line {index}")
        value = entry.get("text")
        if not isinstance(value, str):
            raise TranslationProducerError(f"translation for line {index} is not a string")
        source_text = by_index[index].text
        if not value.strip():
            raise TranslationProducerError(f"translation for line {index} is empty")
        if len(value) > translated_size_ceiling(source_text):
            raise TranslationProducerError(
                f"translation for line {index} is {len(value)} chars against a"
                f" {translated_size_ceiling(source_text)}-char growth bound"
            )
        out[index] = value
    missing = set(by_index) - set(out)
    if missing:
        raise TranslationProducerError(
            f"reply is missing lines {sorted(missing)[:5]} (of {len(batch)})"
        )
    return out


def _make_batches(
    lines: Sequence[TranslationLineSource], *, max_segments: int, max_chars: int
) -> list[list[TranslationLineSource]]:
    batches: list[list[TranslationLineSource]] = []
    batch: list[TranslationLineSource] = []
    chars = 0
    for line in lines:
        if batch and (len(batch) >= max_segments or chars + len(line.text) > max_chars):
            batches.append(batch)
            batch = []
            chars = 0
        batch.append(line)
        chars += len(line.text)
    if batch:
        batches.append(batch)
    return batches


def _translate_batch(
    client: ChatJsonLLM,
    batch: list[TranslationLineSource],
    *,
    source_label: str | None,
    target_label: str,
    attempts: int,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[int, str]:
    last: Exception | None = None
    for _ in range(max(1, attempts)):
        # Checked before EVERY call, not just between top-level batches: the
        # retry ladder and recursive bisection can multiply one failing
        # batch into many calls, and a cancel must stop that spend at the
        # next call boundary, not after the whole ladder runs dry.
        if should_cancel is not None and should_cancel():
            raise TranslationCancelled()
        try:
            body = client.chat_json(
                _batch_prompt(batch, source_label=source_label, target_label=target_label)
            )
            return _parse_batch_reply(body, batch)
        except (LLMError, TranslationProducerError) as exc:
            last = exc
            logger.warning(
                "translation batch of %d failed (%s) — %s",
                len(batch),
                type(exc).__name__,
                exc,
            )
    if len(batch) == 1:
        # Classification only — LLMError text can embed endpoint response
        # bodies, and this message lands on the job row. The per-attempt
        # warnings above already logged the detail.
        raise TranslationProducerError(
            f"line {batch[0].line_index} could not be translated after"
            f" {max(1, attempts)} attempts ({type(last).__name__})"
        )
    # Bisect: local models lose track of long numbered lists — smaller batches
    # recover most failures without any fuzzy realignment.
    mid = len(batch) // 2
    left = _translate_batch(
        client, batch[:mid], source_label=source_label, target_label=target_label,
        attempts=attempts, should_cancel=should_cancel,
    )
    right = _translate_batch(
        client, batch[mid:], source_label=source_label, target_label=target_label,
        attempts=attempts, should_cancel=should_cancel,
    )
    return {**left, **right}


def translate_lines(
    client: ChatJsonLLM,
    lines: Sequence[TranslationLineSource],
    *,
    source_label: str | None,
    target_label: str,
    settings: Settings,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[int, str]:
    """Translate every line, returning ``{line_index: translated_text}``.

    Raises :class:`TranslationProducerError` when any line cannot be validly
    translated — partial output never leaves this function, so the writer only
    ever sees complete generations. ``should_cancel`` is checked before every
    LLM call — including retries and bisected sub-batches — so a cancel stops
    a long generation after the in-flight call, never after a whole retry
    ladder; an observed flag raises :class:`TranslationCancelled`.
    """
    out: dict[int, str] = {}
    to_translate: list[TranslationLineSource] = []
    for line in lines:
        if line.text.strip():
            to_translate.append(line)
        else:
            # Deterministic: an empty line stays exactly itself (usually "").
            out[line.line_index] = line.text
    for batch in _make_batches(
        to_translate,
        max_segments=settings.llm_batch_max_segments,
        max_chars=settings.llm_batch_max_chars,
    ):
        out.update(
            _translate_batch(
                client,
                batch,
                source_label=source_label,
                target_label=target_label,
                attempts=settings.llm_attempts_per_batch,
                should_cancel=should_cancel,
            )
        )
    return out
