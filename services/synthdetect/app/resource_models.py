"""Torch-free resource-telemetry contract models (hardware-aware processing, W1).

Vendored BYTE-IDENTICALLY into each GPU service image (whisper / pyannote /
titanet); a contract test asserts the three copies match. Pydantic-only by
design: the repo's contract tests import each service's ``schemas.py`` without
torch or NVML present, and ``schemas.py`` imports :class:`Resources` from here,
so this module must never import torch, ``pynvml``, or any GPU library, even
lazily. The background sampler that fills these models lives in the separate
``resource_probe.py`` (which ``schemas.py`` must never import).

Wire discipline (docs/gpu-contracts.md): integer bytes on the wire; utilization
bounded 0-100; ``vram_used_bytes <= vram_total_bytes``; NaN/inf rejected at the
source before construction; throttle reasons are normalized labels, never a raw
bitmask; ``availability`` is a tri-state validity discriminator, so every
hardware field stays optional and a null value is honest, not an error.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Tri-state telemetry availability. ``disabled`` = operator turned it off;
# ``unsupported`` = no GPU / no NVML / resolution failed; ``ok`` = live values.
Availability = Literal["ok", "unsupported", "disabled"]

# Normalized throttle labels. NVML exposes a bitmask of clock-throttle reasons;
# the sampler decodes known bits to these labels and OMITS unknown/future bits
# (never surfaces a raw number). ``thermal_sw``/``thermal_hw`` separate the
# driver's software slowdown from the hardware thermal cutoff; ``power`` covers
# power-cap and power-brake; ``clock`` covers applications-clock / sync-boost /
# display-clock settings; ``idle`` is the benign idle state.
ThrottleReason = Literal["thermal_sw", "thermal_hw", "power", "clock", "idle"]


class GpuTelemetry(BaseModel):
    """One physical GPU's cached telemetry as seen by this service.

    Every hardware field is optional and gated by ``availability``: a consumer
    reads ``availability`` first and treats null values as "not measured", never
    as zero. Bounds are enforced here as a backstop; the sampler sanitizes each
    value (finite, in range, integer bytes) before construction.
    """

    availability: Availability
    gpu_uuid: str | None = None
    utilization_percent: int | None = Field(default=None, ge=0, le=100)
    vram_used_bytes: int | None = Field(default=None, ge=0)
    vram_total_bytes: int | None = Field(default=None, ge=0)
    temperature_celsius: int | None = None
    throttle_active: bool | None = None
    throttle_reasons: list[ThrottleReason] = Field(default_factory=list)
    # Cumulative-since-process-start signals the sampler tracks across ticks
    # (they cannot be reconstructed from a single instantaneous read).
    max_temperature_celsius: int | None = None
    throttle_events_since_start: int | None = Field(default=None, ge=0)
    # Staleness of the cached read, computed at serve time from a monotonic
    # clock (never wall time), clamped to >= 0.
    sample_age_seconds: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _drop_impossible_used(self) -> "GpuTelemetry":
        used, total = self.vram_used_bytes, self.vram_total_bytes
        if used is not None and total is not None and used > total:
            # An impossible pair is telemetry corruption. Drop the used figure
            # rather than emit used>total; never raise (a validator raising here
            # would let a corrupt sample turn /healthz into a 500).
            object.__setattr__(self, "vram_used_bytes", None)
        return self


class CpuTelemetry(BaseModel):
    """Host-visible CPU advisory. ``logical_cores`` / ``load_average_1m`` come
    from the stdlib (``os.cpu_count`` / ``os.getloadavg``); they report the
    HOST, ignoring any cgroup CPU quota on this container, so they are advisory
    only and must never drive a sizing decision on their own.
    """

    availability: Availability
    logical_cores: int | None = Field(default=None, ge=0)
    load_average_1m: float | None = Field(default=None, ge=0)


class Admission(BaseModel):
    """In-process admission state (contention sourced honestly, not inferred).

    ``pending`` is admitted-plus-waiting inference calls at read time,
    ``max_pending`` the bound past which the service returns a retryable
    ``503 saturated``, ``rejected_since_start`` a monotonic count of those
    rejections, ``process_started_at`` an ISO-8601 UTC timestamp so a consumer
    can rate the rejection count. Read instantaneously under the service's
    admission lock, never sampled on the background cadence.
    """

    pending: int = Field(ge=0)
    max_pending: int = Field(ge=0)
    rejected_since_start: int = Field(ge=0)
    process_started_at: str


class Resources(BaseModel):
    """The additive, optional ``resources`` block on each service /healthz.

    Optional on :class:`HealthResponse` for rolling-version compatibility (an
    older service omits it entirely), but an upgraded service ALWAYS emits it,
    even when GPU telemetry is ``disabled``/``unsupported`` -- so ``admission``
    (which is always meaningful) never disappears with the GPU.
    """

    gpu: GpuTelemetry
    admission: Admission
    cpu: CpuTelemetry | None = None
