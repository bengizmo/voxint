"""Whisper service contract schemas — torch-free by design.

Repo-level contract tests import this module without GPU dependencies; keep it
pydantic-only. See docs/gpu-contracts.md for the authoritative contract.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Torch-free by design: import ONLY the pydantic models, never the sampler
# (app.resource_probe pulls in the lazy torch/NVML machinery). The contract
# tests import this module without a GPU stack present.
from app.resource_models import Resources

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
    # Additive v1 fields: inference engine vs the compute runtime it runs on
    # (docs/gpu-contracts.md "Health"). Versions are null until model load.
    engine: str
    engine_version: str | None = None
    runtime: str | None = None
    runtime_version: str | None = None
    model_loaded: bool
    # Additive v1 fields (#33 Slice 2b): the effective decode identity so two
    # deployments are distinguishable and a numerics change is visible on
    # /healthz. Computed once at load and cached (never hashes weights).
    # ``decode_config_hash`` digests the effective decode config (engine, the
    # canonical compute device, model, compute_type, batch_size, engine/runtime
    # versions, vad params + plan version) — NOT the kwargs
    # BatchedInferencePipeline silently ignores.
    # Null until the model is loaded.
    vad_plan_version: str | None = None
    vad_params: dict[str, Any] | None = None
    decode_config_hash: str | None = None
    model_revision: str | None = None
    # Additive v1 field (hardware-aware processing, W1): optional nested
    # hardware telemetry. Absent on older services; an upgraded service always
    # emits it (GPU tri-state + always-present admission). Consumers tolerate
    # absence exactly as they do the engine/runtime fields.
    resources: Resources | None = None
