"""Synthdetect service contract schemas -- torch-free by design.

Repo-level contract tests import this module without GPU dependencies; keep it
pydantic-only. See docs/gpu-contracts.md for the authoritative contract.
"""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.resource_models import Resources

SERVICE_NAME = "synthdetect"
CONTRACT_VERSION = "v1"

INFERENCE_SPACE = "w2v2-aasist-df-m2-s0e11"

MODEL_WINDOW_SAMPLES = 64_600
SAMPLE_RATE = 16_000
MODEL_WINDOW_SECONDS = MODEL_WINDOW_SAMPLES / SAMPLE_RATE  # 4.0375

MIN_SCORABLE_SAMPLES = 8_000
MIN_SCORABLE_SECONDS = MIN_SCORABLE_SAMPLES / SAMPLE_RATE  # 0.5


class Interval(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_seconds: float = Field(..., ge=0.0)
    end_seconds: float

    @field_validator("start_seconds", "end_seconds")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("Interval bounds must be finite")
        return v

    @model_validator(mode="after")
    def _ordered(self) -> "Interval":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be > start_seconds")
        return self


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., description="Audio path relative to MEDIA_ROOT")
    intervals: list[Interval] = Field(..., min_length=1, max_length=4096)


class IntervalResult(BaseModel):
    raw_score: float | None = None
    window_count: int = Field(..., ge=0)
    skip_reason: Literal["too_short"] | None = None

    @field_validator("raw_score")
    @classmethod
    def _score_finite(cls, v: float | None) -> float | None:
        if v is not None and not math.isfinite(v):
            raise ValueError("raw_score must be finite")
        return v

    @model_validator(mode="after")
    def _exclusive(self) -> "IntervalResult":
        if (self.raw_score is None) == (self.skip_reason is None):
            raise ValueError("exactly one of raw_score / skip_reason must be set")
        if self.skip_reason is not None and self.window_count != 0:
            raise ValueError("window_count must be 0 when skip_reason is set")
        if self.raw_score is not None and self.window_count < 1:
            raise ValueError("window_count must be >= 1 when raw_score is set")
        return self


class ScoreResponse(BaseModel):
    inference_space: str = INFERENCE_SPACE
    results: list[IntervalResult]


class HealthResponse(BaseModel):
    status: str
    service: str = SERVICE_NAME
    version: str
    contract_version: str = CONTRACT_VERSION
    inference_space: str = INFERENCE_SPACE
    model_window_samples: int = MODEL_WINDOW_SAMPLES
    sample_rate: int = SAMPLE_RATE
    min_scorable_samples: int = MIN_SCORABLE_SAMPLES
    model: str | None
    device: str
    engine: str
    engine_version: str | None = None
    runtime: str | None = None
    runtime_version: str | None = None
    model_loaded: bool
    resources: Resources | None = None
