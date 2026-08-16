"""enhance_match internals: batching, hint selection, retry/circuit/budget —
pure objects, no database, no network."""

import uuid

from tests.fakes import FakeLLM
from voxint.clients.base import (
    EnhancementBatchResult,
    EnhancementRequestSegment,
    SpeakerNameHint,
)
from voxint.clients.llm import LLMError
from voxint.db.models import TranscriptSegment
from voxint.pipeline.stages.context import LLMPolicy
from voxint.pipeline.stages.enhance_match import _build_batches, _enhance, _select_hints

RUN_ID = uuid.uuid4()


def seg(index: int, text: str, label: str | None = "SPEAKER_00") -> TranscriptSegment:
    return TranscriptSegment(
        pipeline_run_id=RUN_ID,
        segment_index=index,
        start_seconds=float(index),
        end_seconds=float(index + 1),
        raw_text=text,
        diarization_label=label,
    )


# -------------------------------------------------------------------- batching


def test_batches_bounded_by_segment_count() -> None:
    policy = LLMPolicy(batch_max_segments=2, batch_max_chars=10_000)
    batches = _build_batches([seg(i, "x") for i in range(5)], policy)
    assert [len(b) for b in batches] == [2, 2, 1]
    # contiguity preserved
    assert [s.segment_index for b in batches for s in b] == [0, 1, 2, 3, 4]


def test_batches_bounded_by_chars() -> None:
    policy = LLMPolicy(batch_max_segments=100, batch_max_chars=10)
    batches = _build_batches([seg(0, "aaaa"), seg(1, "bbbb"), seg(2, "cccc")], policy)
    assert [len(b) for b in batches] == [2, 1]


def test_oversized_segment_travels_alone() -> None:
    policy = LLMPolicy(batch_max_segments=100, batch_max_chars=10)
    batches = _build_batches([seg(0, "y" * 50), seg(1, "z")], policy)
    assert [len(b) for b in batches] == [1, 1]


# -------------------------------------------------------------- hint selection


def hint(label: str, name: str, kind: str) -> SpeakerNameHint:
    return SpeakerNameHint(diarization_label=label, name=name, kind=kind)


def test_self_introduction_beats_other() -> None:
    chosen = _select_hints(
        [hint("A", "Misheard", "other"), hint("A", "Jane", "self"), hint("B", "Bob", "other")]
    )
    assert [(p.diarization_label, p.proposed_name) for p in chosen] == [
        ("A", "Jane"),
        ("B", "Bob"),
    ]


def test_earliest_wins_within_kind_and_self_not_displaced() -> None:
    chosen = _select_hints(
        [hint("A", "First", "self"), hint("A", "Second", "self"), hint("A", "Later", "other")]
    )
    assert [(p.diarization_label, p.proposed_name) for p in chosen] == [("A", "First")]


def test_unusable_names_dropped() -> None:
    assert _select_hints([hint("A", "   ", "self"), hint("B", "x" * 500, "self")]) == ()


# ------------------------------------------------- retry / circuit / budget


class ScriptedLLM:
    """Raises for the scripted number of leading calls, then succeeds."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def enhance_segments(
        self,
        segments: tuple[EnhancementRequestSegment, ...],
        context: str,
        *,
        name_attribution_context: str = "",
    ) -> EnhancementBatchResult:
        self.calls += 1
        if self.calls <= self.failures:
            raise LLMError("scripted failure")
        return EnhancementBatchResult(
            enhanced={s.segment_index: s.text.upper() for s in segments}
        )


def test_enhance_forwards_name_attribution_context_to_every_batch() -> None:
    # The #11 pack fragment must reach the client on every batch call, not just
    # the first — a run's attribution guidance holds for the whole transcript.
    llm = FakeLLM()
    policy = LLMPolicy(batch_max_segments=1, batch_max_chars=10_000)
    _enhance(
        llm,
        policy,
        "astronomy context",
        [seg(0, "hello"), seg(1, "world")],
        RUN_ID,
        name_attribution_context="Anchor the recurring host.",
    )
    assert llm.attribution_contexts == ["Anchor the recurring host.", "Anchor the recurring host."]
    assert llm.contexts == ["astronomy context", "astronomy context"]


def test_enhance_defaults_attribution_context_to_empty() -> None:
    llm = FakeLLM()
    _enhance(llm, LLMPolicy(), "", [seg(0, "hi")], RUN_ID)
    assert llm.attribution_contexts == [""]


def one_per_batch(**overrides: float | int) -> LLMPolicy:
    defaults: dict[str, float | int] = {"batch_max_segments": 1, "attempts_per_batch": 2}
    return LLMPolicy(**{**defaults, **overrides})  # type: ignore[arg-type]


def test_retry_recovers_single_batch() -> None:
    llm = ScriptedLLM(failures=1)
    segments = [seg(0, "hello")]
    _enhance(llm, one_per_batch(), "", segments, RUN_ID)
    assert llm.calls == 2
    assert segments[0].enhanced_text == "HELLO"


def test_failed_batch_leaves_null_but_later_batches_proceed() -> None:
    llm = ScriptedLLM(failures=2)  # batch 0 exhausts both attempts
    segments = [seg(0, "a"), seg(1, "b")]
    _enhance(llm, one_per_batch(), "", segments, RUN_ID)
    assert segments[0].enhanced_text is None
    assert segments[1].enhanced_text == "B"


def test_circuit_opens_after_consecutive_failures() -> None:
    llm = ScriptedLLM(failures=100)
    segments = [seg(i, "t") for i in range(10)]
    _enhance(llm, one_per_batch(consecutive_failure_limit=3), "", segments, RUN_ID)
    # 3 failed batches x 2 attempts, then the circuit opens — no further calls.
    assert llm.calls == 6
    assert all(s.enhanced_text is None for s in segments)


def test_budget_exhaustion_stops_before_any_call() -> None:
    llm = ScriptedLLM(failures=0)
    segments = [seg(0, "t")]
    _enhance(llm, one_per_batch(run_budget_seconds=-1.0), "", segments, RUN_ID)
    assert llm.calls == 0
    assert segments[0].enhanced_text is None
