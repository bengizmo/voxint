"""LLM clean-up producer (#758): batched, index-echoed, deletions proposed.

One generation is a sequence of strict-JSON ``chat_json`` calls over the frozen
clean-up source. Each batch sends numbered lines and demands an object envelope
whose entries echo the line numbers back with each line cleaned:

    {"lines": [{"i": 12, "text": "..."}, ...]}

The reply is only a proposal. The writer (``cleanups.record_cleanup``) judges
each line against its source words and builds the stored text from source
characters, so this module never decides what is deleted. It only has to
deliver, for every line, the model's text or ``None`` when no parseable reply
for that line could be had.

Reply validation is an **exact index set**, as in translation, so merged,
dropped and reordered lines fail the batch rather than pairing a proposal with
the wrong line. The failure ladder is retry (``llm_attempts_per_batch``), then
recursive bisection down to single lines. Translation's terminal rule changes:

- a single line whose every attempt ended in a reply-shape failure
  (:class:`~voxint.clients.llm.LLMReplyError` or a misaligned envelope) is
  ``None`` (``malformed``) and stays verbatim;
- a single line whose every attempt ended in a transport failure (a plain
  :class:`~voxint.clients.llm.LLMError`: unreachable, deadline, HTTP error)
  fails the generation, so a dead or refusing endpoint is never reported as a
  run of malformed lines;
- a generation in which no line got a parseable reply fails: an all-verbatim
  "success" from a broken model would be dishonest.

Lines with no words are their own proposal and never reach the model.
"""

import logging
from collections.abc import Callable, Sequence
from typing import Protocol

from voxint.clients.llm import ChatMessage, LLMError, LLMReplyError
from voxint.config import Settings
from voxint.enrichment.cleanup import proposal_keys
from voxint.enrichment.cleanups import CleanupLineSource, proposal_ceiling

logger = logging.getLogger(__name__)

PRODUCER_NAME = "cleanup.llm"
PRODUCER_VERSION = "1"
CLEANUP_PROMPT_VERSION = 1


class CleanupProducerError(Exception):
    """The generation cannot complete: a dead endpoint or a fully broken model."""


class CleanupCancelled(Exception):
    """The cooperative cancel flag was observed around an LLM call."""


class _ReplyShapeError(Exception):
    """The reply parsed as JSON but does not fit the index-echoed envelope."""


class ChatJsonLLM(Protocol):
    """The only capability the producer needs from a client (injection seam)."""

    def chat_json(self, messages: list[ChatMessage]) -> dict[str, object]: ...


_SYSTEM = (
    "You are a careful transcript copy editor. Reply with a single JSON object"
    " and nothing else: no prose, no markdown fences. You may only DELETE words."
    " Remove filler words, verbal hedges, stutters, repeated words and false"
    " starts that add nothing to the meaning. Never add, change, respell,"
    " reorder or merge words. Never delete numbers, negations or names. Keep"
    " every other word exactly as written. If a line needs no change, return it"
    " unchanged. Never merge, drop or reorder lines."
)


def _hints(hint_words: Sequence[str], kept: Sequence[str]) -> str:
    parts: list[str] = []
    if hint_words:
        parts.append(
            "Words and phrases that are usually fillers here: "
            + ", ".join(hint_words)
            + ". Delete them only where they are fillers in context."
        )
    if kept:
        parts.append("The operator asked to keep these words: " + ", ".join(kept) + ".")
    return ("\n\n" + " ".join(parts)) if parts else ""


def _batch_prompt(
    batch: Sequence[CleanupLineSource], *, hint_words: Sequence[str], kept: Sequence[str]
) -> list[ChatMessage]:
    numbered = "\n".join(f"[{line.line.line_index}] {line.line.text}" for line in batch)
    instruction = (
        f"Clean up the following {len(batch)} numbered transcript lines by deleting"
        " words only. Reply exactly as:"
        ' {"lines": [{"i": <line number>, "text": "<the line with words deleted>"}, ...]}'
        " with one entry per input line, echoing each line's exact number, nothing"
        " else." + _hints(hint_words, kept) + "\n\nLines:\n" + numbered
    )
    return [
        ChatMessage(role="system", content=_SYSTEM),
        ChatMessage(role="user", content=instruction),
    ]


def _parse_batch_reply(
    body: dict[str, object], batch: Sequence[CleanupLineSource]
) -> dict[int, str]:
    """Exact index set, string texts within the writer's proposal bound."""
    if not isinstance(body, dict):
        raise _ReplyShapeError("reply is not a JSON object")
    entries = body.get("lines")
    if not isinstance(entries, list):
        raise _ReplyShapeError("reply has no 'lines' array")
    by_index = {line.line.line_index: line for line in batch}
    out: dict[int, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise _ReplyShapeError("a lines entry is not an object")
        index = entry.get("i")
        # bool is an int subclass: `true` must not pass as index 1.
        if isinstance(index, bool) or not isinstance(index, int):
            raise _ReplyShapeError("a lines entry has a non-integer 'i'")
        if index not in by_index:
            raise _ReplyShapeError(f"reply names unknown line {index}")
        if index in out:
            raise _ReplyShapeError(f"reply repeats line {index}")
        value = entry.get("text")
        if not isinstance(value, str):
            raise _ReplyShapeError(f"text for line {index} is not a string")
        if "\x00" in value:
            raise _ReplyShapeError(f"text for line {index} contains NUL")
        ceiling = proposal_ceiling(by_index[index].words.text)
        if len(value) > ceiling:
            raise _ReplyShapeError(
                f"text for line {index} is {len(value)} chars against a {ceiling}-char bound"
            )
        out[index] = value
    missing = set(by_index) - set(out)
    if missing:
        raise _ReplyShapeError(f"reply is missing lines {sorted(missing)[:5]} (of {len(batch)})")
    return out


def _make_batches(
    lines: Sequence[CleanupLineSource], *, max_segments: int, max_chars: int
) -> list[list[CleanupLineSource]]:
    batches: list[list[CleanupLineSource]] = []
    batch: list[CleanupLineSource] = []
    chars = 0
    for line in lines:
        size = len(line.line.text)
        if batch and (len(batch) >= max_segments or chars + size > max_chars):
            batches.append(batch)
            batch = []
            chars = 0
        batch.append(line)
        chars += size
    if batch:
        batches.append(batch)
    return batches


def _check_cancel(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        raise CleanupCancelled()


def _clean_batch(
    client: ChatJsonLLM,
    batch: list[CleanupLineSource],
    *,
    hint_words: Sequence[str],
    kept: Sequence[str],
    attempts: int,
    should_cancel: Callable[[], bool] | None,
) -> dict[int, str | None]:
    transport_only = True
    last: Exception | None = None
    for _ in range(max(1, attempts)):
        # Before every call, retries and bisected halves included, so a cancel
        # stops the spend at the next call boundary.
        _check_cancel(should_cancel)
        try:
            body = client.chat_json(_batch_prompt(batch, hint_words=hint_words, kept=kept))
        except LLMReplyError as exc:
            transport_only, last = False, exc
            _check_cancel(should_cancel)
        except LLMError as exc:
            last = exc
            _check_cancel(should_cancel)
        else:
            # After the call too, whatever it returned: a cancel that landed
            # while the model was answering wins over its reply or failure.
            _check_cancel(should_cancel)
            try:
                return dict(_parse_batch_reply(body, batch))
            except _ReplyShapeError as exc:
                transport_only, last = False, exc
        logger.warning(
            "clean-up batch of %d failed (%s): %s", len(batch), type(last).__name__, last
        )
    if len(batch) == 1:
        index = batch[0].line.line_index
        if transport_only:
            # Classification only: LLMError text can embed endpoint response
            # bodies, and this message lands on the job row. Transport covers
            # an unreachable endpoint, a deadline and an HTTP error status.
            raise CleanupProducerError(
                f"the language model request failed for line {index} after"
                f" {max(1, attempts)} attempts ({type(last).__name__}); check"
                " the worker logs and the LLM endpoint"
            )
        return {index: None}
    # Bisect: local models lose track of long numbered lists, and a timeout
    # on a long batch often clears on a shorter one.
    mid = len(batch) // 2
    left = _clean_batch(
        client, batch[:mid], hint_words=hint_words, kept=kept,
        attempts=attempts, should_cancel=should_cancel,
    )
    right = _clean_batch(
        client, batch[mid:], hint_words=hint_words, kept=kept,
        attempts=attempts, should_cancel=should_cancel,
    )
    return {**left, **right}


def propose_cleanup(
    client: ChatJsonLLM,
    lines: Sequence[CleanupLineSource],
    *,
    hint_words: Sequence[str],
    kept: Sequence[str],
    settings: Settings,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[int, str | None]:
    """Return ``{line_index: proposal}`` covering every line.

    A proposal is the model's text for the line, the line's own text when it
    has no words, or ``None`` when no parseable reply for it could be had.
    ``hint_words`` (the effective filler words and phrases) and ``kept`` (the
    operator's keeps) are prompt hints only. ``should_cancel`` is checked
    before and after every LLM call; an observed flag raises
    :class:`CleanupCancelled`. Raises :class:`CleanupProducerError` when the
    endpoint stays unreachable for a line, or when no line got a parseable
    reply.
    """
    out: dict[int, str | None] = {}
    to_clean: list[CleanupLineSource] = []
    for line in lines:
        if proposal_keys(line.line.text):
            to_clean.append(line)
        else:
            out[line.line.line_index] = line.line.text
    if not to_clean:
        raise CleanupProducerError("the transcript has no words to clean up")
    for batch in _make_batches(
        to_clean,
        max_segments=settings.llm_batch_max_segments,
        max_chars=settings.llm_batch_max_chars,
    ):
        out.update(
            _clean_batch(
                client,
                batch,
                hint_words=hint_words,
                kept=kept,
                attempts=settings.llm_attempts_per_batch,
                should_cancel=should_cancel,
            )
        )
    if all(out[line.line.line_index] is None for line in to_clean):
        raise CleanupProducerError("the model gave no usable reply for any line")
    return out


__all__ = [
    "CLEANUP_PROMPT_VERSION",
    "PRODUCER_NAME",
    "PRODUCER_VERSION",
    "ChatJsonLLM",
    "CleanupCancelled",
    "CleanupProducerError",
    "propose_cleanup",
]
