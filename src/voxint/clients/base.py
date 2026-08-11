"""Provider protocols — the modularity seams.

The pipeline depends only on these protocols; concrete HTTP clients (P2) and the
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


@dataclass(frozen=True)
class DiarizationResult:
    turns: tuple[DiarizationTurn, ...]


@dataclass(frozen=True)
class EmbeddingResult:
    embedding_space: str
    # One embedding per requested (start, end) window, same order.
    embeddings: tuple[tuple[float, ...], ...] = field(default_factory=tuple)


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
