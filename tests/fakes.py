"""Deterministic fake providers satisfying the client protocols."""

from pathlib import Path

from voxint.clients.base import (
    DiarizationResult,
    DiarizationTurn,
    EmbeddingResult,
    TranscriptionResult,
    TranscriptionSegment,
)

FAKE_EMBEDDING_SPACE = "fake-192-v1"


class FakeASR:
    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        return TranscriptionResult(
            segments=(
                TranscriptionSegment(0.0, 4.0, "hello and welcome to the show"),
                TranscriptionSegment(4.0, 8.0, "thanks for having me"),
                TranscriptionSegment(8.0, 9.0, "mm", suspect=True),
            ),
            language="en",
        )


class FakeDiarizer:
    def diarize(self, audio_path: Path) -> DiarizationResult:
        return DiarizationResult(
            turns=(
                DiarizationTurn(0.0, 4.0, "SPEAKER_00"),
                DiarizationTurn(4.0, 9.0, "SPEAKER_01"),
            )
        )


class FakeEmbedder:
    def embed(
        self, audio_path: Path, windows: tuple[tuple[float, float], ...]
    ) -> EmbeddingResult:
        return EmbeddingResult(
            embedding_space=FAKE_EMBEDDING_SPACE,
            embeddings=tuple(
                tuple(float(i + 1) / 192.0 for _ in range(192))
                for i, _ in enumerate(windows)
            ),
        )


class FakeLLM:
    def enhance(self, text: str, context: str) -> str:
        return text.capitalize()
