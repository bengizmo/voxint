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
    def transcribe(self, audio_path: Path) -> TranscriptionResult: ...


class DiarizerClient(Protocol):
    def diarize(self, audio_path: Path) -> DiarizationResult: ...


class EmbedderClient(Protocol):
    def embed(
        self, audio_path: Path, windows: tuple[tuple[float, float], ...]
    ) -> EmbeddingResult: ...


class LLMClient(Protocol):
    def enhance(self, text: str, context: str) -> str: ...
