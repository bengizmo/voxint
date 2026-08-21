"""Pyannote service contract schemas — torch-free by design.

Repo-level contract tests import this module without GPU dependencies; keep it
pydantic-only. See docs/gpu-contracts.md for the authoritative contract.
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Torch-free by design: import ONLY the pydantic models, never the sampler
# (app.resource_probe pulls in the lazy torch/NVML machinery). The contract
# tests import this module without a GPU stack present.
from app.resource_models import Resources

SERVICE_NAME = "pyannote"
CONTRACT_VERSION = "v1"


class DiarizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., description="Audio path relative to MEDIA_ROOT")
    min_speakers: int = Field(default=1, ge=1, le=20)
    max_speakers: int = Field(default=10, ge=1, le=20)
    min_turn_seconds: float = Field(
        default=0.5, ge=0.1, le=10.0, description="Raw turns shorter than this are dropped"
    )

    @model_validator(mode="after")
    def _speaker_bounds(self) -> "DiarizeRequest":
        if self.min_speakers > self.max_speakers:
            raise ValueError("min_speakers must be <= max_speakers")
        return self


class Turn(BaseModel):
    start_seconds: float
    end_seconds: float
    label: str  # local to the file (SPEAKER_NN); global identity is the pipeline's job
    overlap: bool = False
    # Summed intersection with all other speakers' turns — lets callers
    # distinguish a grazing overlap from a fully-overlapped turn.
    overlap_seconds: float = 0.0


class SpeakerSummary(BaseModel):
    label: str
    total_seconds: float
    num_turns: int


class DiarizeResponse(BaseModel):
    duration_seconds: float
    num_speakers: int
    turns: list[Turn]
    speakers: list[SpeakerSummary]


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
    # Additive v1 field (hardware-aware processing, W1): optional nested
    # hardware telemetry. Absent on older services; an upgraded service always
    # emits it (GPU tri-state + always-present admission). Consumers tolerate
    # absence exactly as they do the engine/runtime fields.
    resources: Resources | None = None
