"""The #758 clean-up producer: prompt, envelope, batching, bisect and failure rules."""

import re
import uuid
from collections.abc import Callable, Sequence

import pytest

from voxint.adjudication.turns import WordAnchor
from voxint.clients.llm import ChatMessage, LLMError, LLMReplyError
from voxint.config import Settings
from voxint.enrichment.cleanup import source_line
from voxint.enrichment.cleanups import PROPOSAL_SLACK_CHARS, CleanupLineSource
from voxint.enrichment.producers import cleanup_llm
from voxint.enrichment.producers.cleanup_llm import (
    CleanupCancelled,
    CleanupProducerError,
    propose_cleanup,
)
from voxint.enrichment.translations import TranslationLineSource

_SEG = uuid.UUID(int=9)
_ROW = re.compile(r"^\[(\d+)\] (.*)$")
_TWO = {"llm_attempts_per_batch": 2}


def _line(index: int, text: str) -> CleanupLineSource:
    anchors = tuple(
        WordAnchor(_SEG, i, i + 1, m.start(), m.end())
        for i, m in enumerate(re.finditer(r"\S+", text))
    )
    return CleanupLineSource(
        TranslationLineSource(index, _SEG, None, None, text), source_line(text, anchors)
    )


def _lines(*texts: str) -> list[CleanupLineSource]:
    return [_line(i, text) for i, text in enumerate(texts)]


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "llm_attempts_per_batch": 1,
        "llm_batch_max_segments": 32,
        "llm_batch_max_chars": 12000,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


def _numbered(messages: Sequence[ChatMessage]) -> dict[int, str]:
    rows = (_ROW.match(row) for row in messages[-1].content.splitlines())
    return {int(m.group(1)): m.group(2) for m in rows if m}


def _strip_um(text: str) -> str:
    return " ".join(word for word in text.split() if word.strip(",").lower() != "um")


Reply = dict[str, object] | Exception


class ScriptedLLM:
    """Answers through ``respond(lines, call_number)``; records every batch."""

    def __init__(self, respond: Callable[[dict[int, str], int], Reply] | None = None) -> None:
        self._respond = respond or (lambda lines, _n: _echo(lines))
        self.batches: list[list[int]] = []
        self.messages: list[list[ChatMessage]] = []
        self.on_call: Callable[[], None] | None = None

    def chat_json(self, messages: list[ChatMessage]) -> dict[str, object]:
        lines = _numbered(messages)
        self.batches.append(sorted(lines))
        self.messages.append(messages)
        if self.on_call is not None:
            self.on_call()
        reply = self._respond(lines, len(self.batches))
        if isinstance(reply, Exception):
            raise reply
        return reply


def _echo(lines: dict[int, str], edit: Callable[[str], str] = _strip_um) -> dict[str, object]:
    return {"lines": [{"i": i, "text": edit(text)} for i, text in lines.items()]}


def _run(
    llm: ScriptedLLM,
    lines: Sequence[CleanupLineSource],
    *,
    settings: Settings | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[int, str | None]:
    return propose_cleanup(
        llm,
        lines,
        hint_words=["um", "you know"],
        kept=["uh"],
        settings=settings or _settings(),
        should_cancel=should_cancel,
    )


def test_every_line_gets_the_model_text_and_wordless_lines_skip_the_model() -> None:
    llm = ScriptedLLM()
    out = _run(llm, _lines("Um, we went.", "", "  -- ", "So um yes."))
    assert out == {0: "we went.", 1: "", 2: "  -- ", 3: "So yes."}
    assert llm.batches == [[0, 3]]


def test_prompt_numbers_lines_and_carries_hints() -> None:
    llm = ScriptedLLM()
    _run(llm, _lines("Um, we went."))
    system, user = llm.messages[0]
    assert system.role == "system" and "DELETE" in system.content
    assert "Never delete numbers, negations or names" in system.content
    assert user.content.endswith("Lines:\n[0] Um, we went.")
    assert "um, you know" in user.content
    assert "asked to keep these words: uh." in user.content
    assert '{"lines": [{"i": <line number>' in user.content


def test_prompt_omits_empty_hints() -> None:
    llm = ScriptedLLM()
    propose_cleanup(llm, _lines("Um, we went."), hint_words=[], kept=[], settings=_settings())
    assert "usually fillers" not in llm.messages[0][1].content
    assert "asked to keep" not in llm.messages[0][1].content


def test_batches_follow_the_segment_and_char_knobs() -> None:
    llm = ScriptedLLM()
    texts = _lines("a b", "c d", "e f", "g h", "i j")
    _run(llm, texts, settings=_settings(llm_batch_max_segments=2))
    assert llm.batches == [[0, 1], [2, 3], [4]]
    llm = ScriptedLLM()
    _run(llm, _lines("aaaa", "bbbb", "cccc"), settings=_settings(llm_batch_max_chars=8))
    assert llm.batches == [[0, 1], [2]]


def test_misaligned_reply_bisects_to_single_lines() -> None:
    def respond(lines: dict[int, str], _n: int) -> Reply:
        if len(lines) > 1:
            return {"lines": [{"i": min(lines), "text": "merged"}]}
        return _echo(lines)

    llm = ScriptedLLM(respond)
    out = _run(llm, _lines("Um a", "Um b", "Um c", "Um d"))
    assert out == {0: "a", 1: "b", 2: "c", 3: "d"}
    assert llm.batches == [[0, 1, 2, 3], [0, 1], [0], [1], [2, 3], [2], [3]]


def test_a_line_that_stays_malformed_is_none_and_the_rest_survive() -> None:
    def respond(lines: dict[int, str], _n: int) -> Reply:
        if 1 in lines:
            return LLMReplyError("completion content is not valid JSON")
        return _echo(lines)

    llm = ScriptedLLM(respond)
    out = _run(llm, _lines("Um a", "Um b", "Um c"), settings=_settings(llm_attempts_per_batch=2))
    assert out == {0: "a", 1: None, 2: "c"}
    assert llm.batches.count([1]) == 2


def test_retry_recovers_before_bisecting() -> None:
    def respond(lines: dict[int, str], n: int) -> Reply:
        return LLMReplyError("garbled") if n == 1 else _echo(lines)

    llm = ScriptedLLM(respond)
    out = _run(llm, _lines("Um a", "Um b"), settings=_settings(llm_attempts_per_batch=2))
    assert out == {0: "a", 1: "b"}
    assert llm.batches == [[0, 1], [0, 1]]


def test_transport_failure_that_persists_fails_the_generation() -> None:
    secret_body = "upstream said: internal-token-xyz"
    llm = ScriptedLLM(lambda _lines, _n: LLMError(secret_body))
    with pytest.raises(CleanupProducerError) as exc:
        _run(llm, _lines("Um a", "Um b", "Um c", "Um d"), settings=_settings(**_TWO))
    assert "request failed for line 0 after 2 attempts (LLMError)" in str(exc.value)
    assert secret_body not in str(exc.value)
    # Depth-first: the dead endpoint is found at the first leaf, not every line.
    assert llm.batches == [[0, 1, 2, 3]] * 2 + [[0, 1]] * 2 + [[0]] * 2


def test_transport_failure_on_a_long_batch_can_clear_on_bisection() -> None:
    def respond(lines: dict[int, str], _n: int) -> Reply:
        return LLMError("deadline exceeded") if len(lines) > 1 else _echo(lines)

    out = _run(ScriptedLLM(respond), _lines("Um a", "Um b"))
    assert out == {0: "a", 1: "b"}


def test_a_line_with_any_reply_failure_is_malformed_not_fatal() -> None:
    def respond(lines: dict[int, str], n: int) -> Reply:
        if 0 in lines:
            return LLMError("reset") if n % 2 else LLMReplyError("not an object")
        return _echo(lines)

    out = _run(ScriptedLLM(respond), _lines("Um a", "Um b"), settings=_settings(**_TWO))
    assert out == {0: None, 1: "b"}


def test_no_parseable_reply_for_any_line_fails() -> None:
    llm = ScriptedLLM(lambda _lines, _n: {"nonsense": True})
    with pytest.raises(CleanupProducerError, match="no usable reply for any line"):
        _run(llm, _lines("Um a", "", "Um b"))


def test_a_transcript_without_words_fails_without_calling_the_model() -> None:
    llm = ScriptedLLM()
    with pytest.raises(CleanupProducerError, match="no words to clean up"):
        _run(llm, _lines("", " -- "))
    assert llm.batches == []


def test_cancel_before_the_first_call_makes_no_call() -> None:
    llm = ScriptedLLM()
    with pytest.raises(CleanupCancelled):
        _run(llm, _lines("Um a"), should_cancel=lambda: True)
    assert llm.batches == []


def test_cancel_during_a_call_discards_its_reply() -> None:
    flag = {"cancel": False}
    llm = ScriptedLLM()
    llm.on_call = lambda: flag.update(cancel=True)
    with pytest.raises(CleanupCancelled):
        _run(llm, _lines("Um a", "Um b"), should_cancel=lambda: flag["cancel"])
    assert len(llm.batches) == 1


def test_cancel_stops_a_retry_ladder() -> None:
    calls = {"n": 0}

    def cancel() -> bool:
        return calls["n"] >= 1

    def respond(_lines: dict[int, str], n: int) -> Reply:
        calls["n"] = n
        return LLMReplyError("garbled")

    llm = ScriptedLLM(respond)
    with pytest.raises(CleanupCancelled):
        _run(llm, _lines("Um a", "Um b"), settings=_settings(**_TWO), should_cancel=cancel)
    assert len(llm.batches) == 1


def test_an_oversized_line_reply_ends_malformed() -> None:
    lines = _lines("Um a", "Um b")
    too_long = "x" * (len("Um b") + PROPOSAL_SLACK_CHARS + 1)

    def respond(got: dict[int, str], _n: int) -> Reply:
        return {"lines": [{"i": i, "text": too_long if i == 1 else "a"} for i in got]}

    assert _run(ScriptedLLM(respond), lines) == {0: "a", 1: None}


def test_an_empty_text_is_a_proposal_for_the_writer_to_judge() -> None:
    assert _run(ScriptedLLM(lambda got, _n: _echo(got, lambda _t: "")), _lines("Um a")) == {0: ""}


@pytest.mark.parametrize(
    "body, message",
    [
        ([], "not a JSON object"),
        ({}, "no 'lines' array"),
        ({"lines": {"i": 0}}, "no 'lines' array"),
        ({"lines": ["text"]}, "not an object"),
        ({"lines": [{"i": True, "text": "a"}]}, "non-integer 'i'"),
        ({"lines": [{"i": "0", "text": "a"}]}, "non-integer 'i'"),
        ({"lines": [{"i": 5, "text": "a"}]}, "unknown line 5"),
        ({"lines": [{"i": 0, "text": "a"}, {"i": 0, "text": "a"}]}, "repeats line 0"),
        ({"lines": [{"i": 0, "text": 3}]}, "not a string"),
        ({"lines": [{"i": 0, "text": "a\x00"}]}, "contains NUL"),
        ({"lines": []}, "missing lines [0]"),
    ],
)
def test_reply_envelope_is_validated_exactly(body: object, message: str) -> None:
    with pytest.raises(cleanup_llm._ReplyShapeError) as exc:
        cleanup_llm._parse_batch_reply(body, _lines("Um a"))  # type: ignore[arg-type]
    assert message in str(exc.value)


@pytest.mark.parametrize("failure", [LLMError("reset"), LLMReplyError("garbled")])
def test_cancel_during_a_failing_last_call_wins_over_the_failure(failure: Exception) -> None:
    flag = {"cancel": False}
    llm = ScriptedLLM(lambda _lines, _n: failure)
    llm.on_call = lambda: flag.update(cancel=True)
    with pytest.raises(CleanupCancelled):
        _run(llm, _lines("Um a"), should_cancel=lambda: flag["cancel"])
    assert len(llm.batches) == 1
