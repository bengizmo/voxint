"""Whisper service contract schemas — torch-free by design.

Repo-level contract tests import this module without GPU dependencies; keep it
pydantic-only. See docs/gpu-contracts.md for the authoritative contract.
"""

from pydantic import BaseModel, ConfigDict, Field

SERVICE_NAME = "whisper"
CONTRACT_VERSION = "v1"


class TranscribeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., description="Audio path relative to MEDIA_ROOT")
    language: str | None = Field(
        default="en", description="Target language code; null → auto-detect"
    )
    initial_prompt: str | None = Field(
        default=None, max_length=2000, description="Optional vocabulary/context prompt"
    )
    vad_filter: bool = Field(
        default=True,
        description="Silero VAD via BatchedInferencePipeline; disable for audio VAD misreads",
    )


class Word(BaseModel):
    start_seconds: float
    end_seconds: float
    word: str
    confidence: float


class Segment(BaseModel):
    start_seconds: float
    end_seconds: float
    text: str
    confidence: float | None = None
    # Hallucination soft-tag: text is preserved verbatim; downstream gates
    # decide how to weight flagged spans.
    suspect: bool = False
    suspect_score: float | None = None
    suspect_span: str | None = None


class TranscribeResponse(BaseModel):
    language: str
    duration_seconds: float
    transcript: str
    confidence: float
    segments: list[Segment]
    words: list[Word]
    suspect_segment_count: int = 0


class HealthResponse(BaseModel):
    status: str
    service: str = SERVICE_NAME
    version: str
    contract_version: str = CONTRACT_VERSION
    model: str | None
    device: str
    model_loaded: bool
