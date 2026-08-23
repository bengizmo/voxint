"""Translation producer (#133): batch contract, validation ladder, bisection.

Pure tests over an injected fake client — no DB, no HTTP. The contract under
test: object envelope with echoed indices, exact-index-set validation (merges,
drops, and reorders all fail), retry then recursive bisection down to single
lines, deterministic empty-line short-circuit, and the growth ceiling.
"""

import uuid
from collections.abc import Sequence

import pytest

from voxint.clients.llm import LLMError
from voxint.config import Settings
from voxint.enrichment.producers.translation_llm import (
    TranslationCancelled,
    TranslationProducerError,
    _make_batches,
    translate_lines,
)
from voxint.enrichment.translations import (
    TranslationLineSource,
    translated_size_ceiling,
)


def make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {"llm_enabled": True}
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


def line(index: int, text: str) -> TranslationLineSource:
    return TranslationLineSource(
        line_index=index,
        segment_id=uuid.uuid4(),
        word_start=None,
        word_end=None,
        text=text,
    )


def body_for(entries: dict[int, str]) -> dict[str, object]:
    return {"translations": [{"i": i, "text": t} for i, t in entries.items()]}


class FakeLLM:
    """Returns canned bodies per call; a list entry may be an exception. A
    callable entry receives the prompt text and returns the body — used to
    answer bisected batches correctly without knowing the split order."""

    def __init__(self, bodies: Sequence[object]) -> None:
        self._bodies = list(bodies)
        self.calls = 0
        self.prompts: list[str] = []

    def chat_json(self, messages: object) -> dict[str, object]:
        assert isinstance(messages, Sequence)
        prompt = str(messages[-1].content)  # type: ignore[attr-defined]
        self.prompts.append(prompt)
        body = self._bodies[self.calls] if self.calls < len(self._bodies) else self._bodies[-1]
        self.calls += 1
        if isinstance(body, Exception):
            raise body
        if callable(body):
            result = body(prompt)
            assert isinstance(result, dict)
            return result
        assert isinstance(body, dict)
        return body


def echo_translator(prefix: str = "T:") -> object:
    """Answers any batch prompt correctly by echoing the numbered lines."""

    def _answer(prompt: str) -> dict[str, object]:
        entries = {}
        for row in prompt.splitlines():
            if row.startswith("[") and "]" in row:
                idx, _, rest = row.partition("] ")
                entries[int(idx[1:])] = f"{prefix}{rest}"
        return body_for(entries)

    return _answer


class TestHappyPath:
    def test_translates_all_lines_and_keeps_empty_lines_verbatim(self) -> None:
        lines = [line(0, "Hello."), line(1, ""), line(2, "Goodbye.")]
        llm = FakeLLM([echo_translator()])
        out = translate_lines(
            llm, lines, source_label="English (en)", target_label="Spanish (es)",
            settings=make_settings(),
        )
        assert out == {0: "T:Hello.", 1: "", 2: "T:Goodbye."}
        # The empty line never reached the model.
        assert llm.calls == 1
        assert "[1]" not in llm.prompts[0]

    def test_prompt_names_languages_and_lines(self) -> None:
        llm = FakeLLM([echo_translator()])
        translate_lines(
            llm, [line(0, "Hi.")], source_label=None, target_label="French (fr)",
            settings=make_settings(),
        )
        assert "the source language" in llm.prompts[0]
        assert "French (fr)" in llm.prompts[0]
        assert "[0] Hi." in llm.prompts[0]

    def test_all_empty_lines_never_call_the_model(self) -> None:
        llm = FakeLLM([AssertionError("must not be called")])
        out = translate_lines(
            llm, [line(0, ""), line(1, "  ")], source_label=None,
            target_label="Spanish (es)", settings=make_settings(),
        )
        assert out == {0: "", 1: "  "}
        assert llm.calls == 0


class TestValidation:
    def _one(self, body: object, text: str = "Hello there.") -> None:
        llm = FakeLLM([body])
        translate_lines(
            llm, [line(0, text)], source_label=None, target_label="Spanish (es)",
            settings=make_settings(llm_attempts_per_batch=1),
        )

    @pytest.mark.parametrize(
        "body",
        [
            {"nope": []},  # no translations array
            {"translations": "not a list"},
            {"translations": [["not", "an", "object"]]},
            {"translations": [{"i": "0", "text": "Hola."}]},  # string index
            {"translations": [{"i": True, "text": "Hola."}]},  # bool is not an index
            {"translations": [{"i": 7, "text": "Hola."}]},  # unknown line
            {"translations": [{"i": 0, "text": 42}]},  # non-string text
            {"translations": [{"i": 0, "text": "  "}]},  # empty translation
            {"translations": []},  # dropped line
            {  # duplicate line
                "translations": [
                    {"i": 0, "text": "Hola."},
                    {"i": 0, "text": "Hola otra vez."},
                ]
            },
        ],
    )
    def test_malformed_reply_fails_closed(self, body: object) -> None:
        with pytest.raises(TranslationProducerError):
            self._one(body)

    def test_growth_ceiling_rejects_runaway_output(self) -> None:
        source = "Hi."
        runaway = "x" * (translated_size_ceiling(source) + 1)
        with pytest.raises(TranslationProducerError):
            self._one(body_for({0: runaway}), text=source)

    def test_ceiling_allows_ordinary_expansion(self) -> None:
        source = "Hi."
        ok = "x" * translated_size_ceiling(source)
        llm = FakeLLM([body_for({0: ok})])
        out = translate_lines(
            llm, [line(0, source)], source_label=None, target_label="Spanish (es)",
            settings=make_settings(llm_attempts_per_batch=1),
        )
        assert out[0] == ok

    def test_merged_lines_fail_the_exact_index_set(self) -> None:
        # Two lines in, one entry out — the classic small-model merge.
        llm = FakeLLM([body_for({0: "Merged both lines."})])
        with pytest.raises(TranslationProducerError):
            translate_lines(
                llm, [line(0, "One."), line(1, "Two.")], source_label=None,
                target_label="Spanish (es)",
                settings=make_settings(llm_attempts_per_batch=1, llm_batch_max_segments=2),
            )


class TestFailureLadder:
    def test_retries_within_a_batch(self) -> None:
        llm = FakeLLM([LLMError("transient"), body_for({0: "Hola."})])
        out = translate_lines(
            llm, [line(0, "Hello.")], source_label=None, target_label="Spanish (es)",
            settings=make_settings(llm_attempts_per_batch=2),
        )
        assert out == {0: "Hola."}
        assert llm.calls == 2

    def test_bisection_recovers_a_failing_pair(self) -> None:
        # The 2-line batch always fails; each single-line batch succeeds.
        answer = echo_translator()

        def flaky(prompt: str) -> dict[str, object]:
            if "[0]" in prompt and "[1]" in prompt:
                raise LLMError("lost track of the list")
            return answer(prompt)  # type: ignore[operator]

        llm = FakeLLM([flaky])
        out = translate_lines(
            llm, [line(0, "One."), line(1, "Two.")], source_label=None,
            target_label="Spanish (es)",
            settings=make_settings(llm_attempts_per_batch=1, llm_batch_max_segments=2),
        )
        assert out == {0: "T:One.", 1: "T:Two."}
        # 1 failed pair call + 2 single-line calls.
        assert llm.calls == 3

    def test_irreducible_single_line_fails_the_generation(self) -> None:
        llm = FakeLLM([LLMError("down")])
        with pytest.raises(TranslationProducerError):
            translate_lines(
                llm, [line(0, "Hello.")], source_label=None, target_label="Spanish (es)",
                settings=make_settings(llm_attempts_per_batch=2),
            )
        assert llm.calls == 2  # attempts honored at size 1

    def test_cancel_between_batches(self) -> None:
        cancels = iter([False, True])
        llm = FakeLLM([echo_translator()])
        with pytest.raises(TranslationCancelled):
            translate_lines(
                llm, [line(0, "One."), line(1, "Two.")], source_label=None,
                target_label="Spanish (es)",
                settings=make_settings(llm_batch_max_segments=1),
                should_cancel=lambda: next(cancels),
            )
        assert llm.calls == 1  # first batch ran, second was cancelled


class TestBatching:
    def test_batches_respect_segment_and_char_caps(self) -> None:
        lines = [line(i, "x" * 10) for i in range(5)]
        by_segments = _make_batches(lines, max_segments=2, max_chars=10_000)
        assert [len(b) for b in by_segments] == [2, 2, 1]
        by_chars = _make_batches(lines, max_segments=100, max_chars=25)
        assert [len(b) for b in by_chars] == [2, 2, 1]

    def test_one_oversized_line_still_gets_its_own_batch(self) -> None:
        lines = [line(0, "x" * 100)]
        assert [len(b) for b in _make_batches(lines, max_segments=4, max_chars=10)] == [1]
