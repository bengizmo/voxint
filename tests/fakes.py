"""Deterministic fake providers satisfying the client protocols."""

from pathlib import Path

from voxint.clients.base import (
    DiarizationResult,
    DiarizationTurn,
    EmbeddingEntry,
    EmbeddingResult,
    EnhancementBatchResult,
    EnhancementRequestSegment,
    SpeakerNameHint,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
)
from voxint.clients.llm import LLMError

FAKE_EMBEDDING_SPACE = "fake-192-v1"


def _fake_words(text: str, start: float, end: float) -> tuple[TranscriptionWord, ...]:
    """Evenly-spaced word timings spanning [start, end) for a segment's text —
    deterministic, so the transcribe stage's word bucketing runs on the fakes."""
    tokens = text.split()
    if not tokens:
        return ()
    step = (end - start) / len(tokens)
    return tuple(
        TranscriptionWord(
            start_seconds=start + i * step,
            end_seconds=start + (i + 1) * step,
            word=token,
            confidence=0.9,
        )
        for i, token in enumerate(tokens)
    )


class FakeASR:
    def __init__(self) -> None:
        # Records the initial_prompt of the most recent transcribe call so tests
        # can assert which vocabulary actually reached ASR.
        self.last_initial_prompt: str | None = None

    def transcribe(
        self, audio_path: Path, initial_prompt: str | None = None
    ) -> TranscriptionResult:
        self.last_initial_prompt = initial_prompt
        segments = (
            TranscriptionSegment(0.0, 4.0, "hello and welcome to the show"),
            TranscriptionSegment(4.0, 8.0, "thanks for having me"),
            TranscriptionSegment(8.0, 9.0, "mm", suspect=True),
        )
        words = tuple(
            w
            for seg in segments
            for w in _fake_words(seg.text, seg.start_seconds, seg.end_seconds)
        )
        return TranscriptionResult(segments=segments, language="en", words=words)


class FakeDiarizer:
    def diarize(self, audio_path: Path) -> DiarizationResult:
        return DiarizationResult(
            turns=(
                DiarizationTurn(0.0, 4.0, "SPEAKER_00"),
                DiarizationTurn(4.0, 9.0, "SPEAKER_01"),
                # Repeated label + sub-second window: exercises the skip path
                # and proves labels may repeat across turns (real diarization
                # always produces many turns per speaker).
                DiarizationTurn(8.5, 9.0, "SPEAKER_00", overlap=True, overlap_seconds=0.5),
            )
        )


class FakeEmbedder:
    def embed(
        self, audio_path: Path, windows: tuple[tuple[float, float], ...]
    ) -> EmbeddingResult:
        return EmbeddingResult(
            embedding_space=FAKE_EMBEDDING_SPACE,
            entries=tuple(
                # Sub-second windows are skipped, mirroring the titanet service's
                # quality gate, so pipeline code must handle skipped entries.
                EmbeddingEntry(embedding=None, snr_db=None, skip_reason="too_short")
                if (end - start) < 1.0
                else EmbeddingEntry(
                    embedding=tuple(float(i + 1) / 192.0 for _ in range(192)),
                    snr_db=20.0,
                )
                for i, (start, end) in enumerate(windows)
            ),
        )


class FakeLLM:
    """Capitalizes each segment; optionally emits canned name hints once."""

    def __init__(self, name_hints: tuple[SpeakerNameHint, ...] = ()) -> None:
        self._name_hints = name_hints
        self.calls: list[tuple[EnhancementRequestSegment, ...]] = []
        # Per-call context strings, so tests can assert the #11 pack fragments
        # reach the client (context = enhancement_context; attribution = the
        # name_attribution_context block).
        self.contexts: list[str] = []
        self.attribution_contexts: list[str] = []

    def enhance_segments(
        self,
        segments: tuple[EnhancementRequestSegment, ...],
        context: str,
        *,
        name_attribution_context: str = "",
    ) -> EnhancementBatchResult:
        self.calls.append(segments)
        self.contexts.append(context)
        self.attribution_contexts.append(name_attribution_context)
        hints = self._name_hints if len(self.calls) == 1 else ()
        return EnhancementBatchResult(
            enhanced={s.segment_index: s.text.capitalize() for s in segments},
            name_hints=hints,
        )


class FailingLLM:
    """Every call raises — exercises the degraded-success path."""

    def enhance_segments(
        self,
        segments: tuple[EnhancementRequestSegment, ...],
        context: str,
        *,
        name_attribution_context: str = "",
    ) -> EnhancementBatchResult:
        raise LLMError("endpoint down")
