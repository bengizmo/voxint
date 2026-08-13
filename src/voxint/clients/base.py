"""Provider protocols — the modularity seams.

The pipeline depends only on these protocols; concrete HTTP clients (with the
pipeline wiring, P3) and the
test fakes both satisfy them. Result types are deliberately minimal: they carry
what downstream stages consume, nothing provider-specific.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class TranscriptionSegment:
    start_seconds: float
    end_seconds: float
    text: str
    # Provider-flagged suspect content (e.g. hallucination soft-tags); gates consume this.
    suspect: bool = False


@dataclass(frozen=True)
class TranscriptionResult:
    segments: tuple[TranscriptionSegment, ...]
    language: str | None = None


@dataclass(frozen=True)
class DiarizationTurn:
    start_seconds: float
    end_seconds: float
    label: str  # local label within one file, e.g. "SPEAKER_00"
    # Summed intersection with other speakers' turns; quality gates threshold on it.
    overlap: bool = False
    overlap_seconds: float = 0.0


@dataclass(frozen=True)
class DiarizationResult:
    turns: tuple[DiarizationTurn, ...]


@dataclass(frozen=True)
class EmbeddingEntry:
    """Per-window embedding outcome. Exactly one of embedding / skip_reason is set:
    a skipped window (too short, too noisy) carries the reason instead of a
    low-quality vector, so downstream quality policy stays auditable."""

    embedding: tuple[float, ...] | None
    snr_db: float | None = None
    skip_reason: str | None = None  # "too_short" | "low_snr" when embedding is None


@dataclass(frozen=True)
class EmbeddingResult:
    embedding_space: str
    # Exactly one entry per requested (start, end) window, same order.
    entries: tuple[EmbeddingEntry, ...] = field(default_factory=tuple)


class ASRClient(Protocol):
    def transcribe(
        self, audio_path: Path, initial_prompt: str | None = None
    ) -> TranscriptionResult: ...


class DiarizerClient(Protocol):
    def diarize(self, audio_path: Path) -> DiarizationResult: ...


class EmbedderClient(Protocol):
    def embed(
        self, audio_path: Path, windows: tuple[tuple[float, float], ...]
    ) -> EmbeddingResult: ...


@dataclass(frozen=True)
class EnhancementRequestSegment:
    """One transcript segment sent for enhancement, identified by its index."""

    segment_index: int
    text: str
    diarization_label: str | None = None


@dataclass(frozen=True)
class SpeakerNameHint:
    """A name the LLM heard attached to a diarization label — evidence for a
    human, never an identity claim (``llm_hint`` proposals are never grounded)."""

    diarization_label: str
    name: str
    kind: str  # "self" (speaker states own name) | "other" (someone names them)


@dataclass(frozen=True)
class EnhancementBatchResult:
    """ID-keyed batch outcome: ``enhanced`` holds exactly one entry per
    requested ``segment_index`` — adapters must reject misaligned responses
    outright rather than return a partially trusted batch."""

    enhanced: dict[int, str]
    name_hints: tuple[SpeakerNameHint, ...] = ()


class LLMClient(Protocol):
    def enhance_segments(
        self, segments: tuple[EnhancementRequestSegment, ...], context: str
    ) -> EnhancementBatchResult: ...
