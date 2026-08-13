"""TitaNet service contract schemas — torch-free by design.

Repo-level contract tests import this module without GPU dependencies; keep it
pydantic-only. See docs/gpu-contracts.md for the authoritative contract.
"""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SERVICE_NAME = "titanet"
CONTRACT_VERSION = "v1"

# Versions the vector semantics: any model OR preprocessing change requires a
# new space id, never a silent swap.
EMBEDDING_SPACE = "titanet-large-v1"
EMBEDDING_DIM = 192


class Window(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_seconds: float = Field(..., ge=0.0)
    end_seconds: float

    @field_validator("start_seconds", "end_seconds")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("Window bounds must be finite")
        return v

    @model_validator(mode="after")
    def _ordered(self) -> "Window":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be > start_seconds")
        return self


class EmbedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., description="Audio path relative to MEDIA_ROOT")
    windows: list[Window] = Field(..., min_length=1, max_length=512)


class WindowResult(BaseModel):
    # Invariant: embedding is non-null iff skip_reason is null.
    embedding: list[float] | None = None
    snr_db: float | None = None
    skip_reason: Literal["too_short", "low_snr"] | None = None

    @field_validator("snr_db")
    @classmethod
    def _snr_finite(cls, v: float | None) -> float | None:
        if v is not None and not math.isfinite(v):
            raise ValueError("snr_db must be finite")
        return v

    @model_validator(mode="after")
    def _exclusive(self) -> "WindowResult":
        if (self.embedding is None) == (self.skip_reason is None):
            raise ValueError("exactly one of embedding / skip_reason must be set")
        if self.embedding is not None:
            if len(self.embedding) != EMBEDDING_DIM:
                raise ValueError(f"embedding must have exactly {EMBEDDING_DIM} dimensions")
            if not all(math.isfinite(x) for x in self.embedding):
                raise ValueError("embedding values must be finite")
        return self


class EmbedResponse(BaseModel):
    embedding_space: str = EMBEDDING_SPACE
    # Exactly one entry per requested window, same order.
    results: list[WindowResult]


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
