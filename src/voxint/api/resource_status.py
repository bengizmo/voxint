"""App-side aggregation of the model services' /healthz hardware telemetry (W2).

The services each report an optional ``resources`` block on /healthz (see
``docs/gpu-contracts.md`` and ``voxint.api.health_probe``). This module turns the
three per-service reports into one operator-facing :class:`ResourceSnapshot`:

- **Parse defensively.** A body without ``resources`` (an older service), or with
  a malformed value, degrades to "telemetry unavailable" for that service, never
  an exception. Numeric bounds are re-checked here so a hostile or buggy source
  cannot push an out-of-range value into the UI; unknown throttle labels are
  tolerated (forward-compatible with a future service).
- **Deduplicate by GPU UUID.** The three services usually share one physical
  card, so their GPU reports are aggregated by ``gpu_uuid`` into a single device
  (NVML memory/utilization is device-global, never one service's usage). The
  freshest reading wins for the instantaneous values; cumulative counters take
  the max across the services sharing the card.
- **Probe concurrently, cache briefly.** A short-TTL single-flight cache means a
  15-second browser poll across several tabs never fans out concurrent live
  probes; the three probes themselves run concurrently, not in sequence.

This is the shared snapshot the visibility surfaces (dashboard strip, resource
page, ``voxint doctor``) render from.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from voxint.api.health_probe import ServiceHealth, _probe_one, _service_targets
from voxint.config import Settings


class GpuInfo(BaseModel):
    """Lenient parse of a service's GPU telemetry. Numeric bounds are enforced
    (a bad value fails the parse -> the service's telemetry is treated as
    unavailable); ``throttle_reasons`` is a plain str list so a future label
    parses rather than crashing."""

    model_config = ConfigDict(extra="ignore")

    availability: str
    gpu_uuid: str | None = None
    utilization_percent: int | None = Field(default=None, ge=0, le=100)
    vram_used_bytes: int | None = Field(default=None, ge=0)
    vram_total_bytes: int | None = Field(default=None, ge=0)
    temperature_celsius: int | None = None
    throttle_active: bool | None = None
    throttle_reasons: list[str] = Field(default_factory=list)
    max_temperature_celsius: int | None = None
    throttle_events_since_start: int | None = Field(default=None, ge=0)
    sample_age_seconds: float | None = Field(default=None, ge=0)


class CpuInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    availability: str
    logical_cores: int | None = Field(default=None, ge=0)
    load_average_1m: float | None = Field(default=None, ge=0)


class AdmissionInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pending: int = Field(ge=0)
    max_pending: int = Field(ge=0)
    rejected_since_start: int = Field(ge=0)
    process_started_at: str


class ServiceResources(BaseModel):
    model_config = ConfigDict(extra="ignore")

    gpu: GpuInfo
    admission: AdmissionInfo
    cpu: CpuInfo | None = None


def parse_resources(raw: object) -> ServiceResources | None:
    """Best-effort parse of a service's ``resources`` block. None if absent or
    malformed (never raises)."""
    if not isinstance(raw, dict):
        return None
    try:
        return ServiceResources.model_validate(raw)
    except ValidationError:
        return None


@dataclass(frozen=True)
class AggregatedGpu:
    """One physical GPU, aggregated across the services that report it."""

    gpu_uuid: str
    utilization_percent: int | None
    vram_used_bytes: int | None
    vram_total_bytes: int | None
    temperature_celsius: int | None
    throttle_active: bool | None
    throttle_reasons: tuple[str, ...]
    max_temperature_celsius: int | None
    throttle_events_since_start: int | None
    sample_age_seconds: float | None
    services: tuple[str, ...]  # service names sharing this card


@dataclass(frozen=True)
class ServiceResourceView:
    """One service's reachability plus its own admission/CPU telemetry."""

    name: str
    up: bool
    device: str | None
    telemetry_available: bool
    gpu_uuid: str | None
    admission: AdmissionInfo | None
    cpu: CpuInfo | None


@dataclass(frozen=True)
class ResourceSnapshot:
    """The aggregated resource view: distinct GPUs plus per-service admission."""

    gpus: tuple[AggregatedGpu, ...]
    services: tuple[ServiceResourceView, ...]
    collected_age_seconds: float  # staleness of this snapshot (cache age)


def _merge_max(a: int | None, b: int | None) -> int | None:
    values = [v for v in (a, b) if v is not None]
    return max(values) if values else None


def _build_snapshot(healths: list[ServiceHealth]) -> ResourceSnapshot:
    """Aggregate per-service health into distinct GPUs (by UUID) + service views."""
    views: list[ServiceResourceView] = []
    # uuid -> the freshest GpuInfo seen + the set of service names sharing it.
    gpu_by_uuid: dict[str, tuple[GpuInfo, list[str]]] = {}

    for health in healths:
        parsed = parse_resources(health.resources)
        gpu = parsed.gpu if parsed else None
        admission = parsed.admission if parsed else None
        cpu = parsed.cpu if parsed else None
        available = gpu is not None and gpu.availability == "ok"
        uuid = gpu.gpu_uuid if (available and gpu is not None) else None
        views.append(
            ServiceResourceView(
                name=health.name,
                up=health.up,
                device=health.device,
                telemetry_available=available,
                gpu_uuid=uuid,
                admission=admission,
                cpu=cpu,
            )
        )
        if available and gpu is not None and uuid is not None:
            existing = gpu_by_uuid.get(uuid)
            if existing is None:
                gpu_by_uuid[uuid] = (gpu, [health.name])
            else:
                prev, names = existing
                names.append(health.name)
                # Freshest instantaneous reading wins (None age = oldest).
                fresher = _is_fresher(gpu, prev)
                base = gpu if fresher else prev
                merged = base.model_copy(
                    update={
                        "max_temperature_celsius": _merge_max(
                            prev.max_temperature_celsius, gpu.max_temperature_celsius
                        ),
                        "throttle_events_since_start": _merge_max(
                            prev.throttle_events_since_start, gpu.throttle_events_since_start
                        ),
                    }
                )
                gpu_by_uuid[uuid] = (merged, names)

    gpus = tuple(
        AggregatedGpu(
            gpu_uuid=uuid,
            utilization_percent=gpu.utilization_percent,
            vram_used_bytes=gpu.vram_used_bytes,
            vram_total_bytes=gpu.vram_total_bytes,
            temperature_celsius=gpu.temperature_celsius,
            throttle_active=gpu.throttle_active,
            throttle_reasons=tuple(gpu.throttle_reasons),
            max_temperature_celsius=gpu.max_temperature_celsius,
            throttle_events_since_start=gpu.throttle_events_since_start,
            sample_age_seconds=gpu.sample_age_seconds,
            services=tuple(names),
        )
        for uuid, (gpu, names) in sorted(gpu_by_uuid.items())
    )
    return ResourceSnapshot(gpus=gpus, services=tuple(views), collected_age_seconds=0.0)


def _is_fresher(candidate: GpuInfo, current: GpuInfo) -> bool:
    """True if ``candidate``'s reading is fresher (smaller sample age)."""
    if candidate.sample_age_seconds is None:
        return False
    if current.sample_age_seconds is None:
        return True
    return candidate.sample_age_seconds < current.sample_age_seconds


def _probe_all_concurrent(
    settings: Settings, client: httpx.Client
) -> list[ServiceHealth]:
    """Probe every model service concurrently (httpx.Client is thread-safe)."""
    targets = _service_targets(settings)
    with ThreadPoolExecutor(max_workers=max(1, len(targets))) as pool:
        return list(
            pool.map(lambda t: _probe_one(client, t[0], t[1]), targets)
        )


# Module-level short-TTL single-flight cache. Serializing refreshes under the
# lock IS the single-flight: a second caller arriving during a refresh waits,
# then sees the just-populated cache instead of firing its own probes.
_cache_lock = threading.Lock()
_cache: tuple[ResourceSnapshot, float] | None = None


def collect_resource_status(
    settings: Settings, *, client: httpx.Client | None = None, force: bool = False
) -> ResourceSnapshot:
    """Aggregated resource telemetry, served from a short-TTL single-flight cache.

    Within ``resource_status_ttl_seconds`` of the last collection the cached
    snapshot is returned (with a refreshed ``collected_age_seconds``), so a
    15-second poll across several browser tabs never triggers concurrent live
    probes. ``force`` bypasses the cache; ``client`` injects a transport in tests.
    """
    global _cache
    ttl = settings.resource_status_ttl_seconds
    now = time.monotonic()
    with _cache_lock:
        if not force and _cache is not None and (now - _cache[1]) < ttl:
            snap, ts = _cache
            return replace(snap, collected_age_seconds=now - ts)
        own_client = client is None
        probe_client = client or httpx.Client(
            timeout=httpx.Timeout(settings.health_probe_timeout_seconds)
        )
        try:
            healths = _probe_all_concurrent(settings, probe_client)
        finally:
            if own_client:
                probe_client.close()
        snap = _build_snapshot(healths)
        _cache = (snap, now)
        return snap


def _reset_cache_for_tests() -> None:
    """Drop the module cache so a test starts from a clean slate."""
    global _cache
    with _cache_lock:
        _cache = None


def _short_uuid(uuid: str) -> str:
    """The first UUID group after the ``GPU-`` tag, for compact display."""
    body = uuid[4:] if uuid.lower().startswith("gpu-") else uuid
    return body.split("-", 1)[0] or uuid


def _gib(nbytes: int | None) -> str:
    return f"{nbytes / (1024**3):.1f}GiB" if nbytes is not None else "?"


def format_resource_status_text(snapshot: ResourceSnapshot) -> str:
    """Plain-text render of the aggregated resource view for ``voxint doctor``."""
    lines = ["Resource telemetry:"]
    if not snapshot.gpus:
        lines.append("  GPU: unavailable (no service reported GPU telemetry)")
    for gpu in snapshot.gpus:
        util = f"{gpu.utilization_percent}%" if gpu.utilization_percent is not None else "?"
        temp = f"{gpu.temperature_celsius}C" if gpu.temperature_celsius is not None else "?"
        peak = (
            f", peak {gpu.max_temperature_celsius}C"
            if gpu.max_temperature_celsius is not None
            else ""
        )
        if gpu.throttle_active:
            reasons = ", ".join(gpu.throttle_reasons) if gpu.throttle_reasons else "unknown"
            throttle = f"throttling ({reasons})"
        else:
            throttle = "no throttle"
        shared = ", ".join(gpu.services)
        lines.append(
            f"  GPU {_short_uuid(gpu.gpu_uuid)} [{shared}]: util {util}, "
            f"VRAM {_gib(gpu.vram_used_bytes)}/{_gib(gpu.vram_total_bytes)}, "
            f"{temp}{peak}, {throttle}"
        )
    for view in snapshot.services:
        if view.admission is not None:
            adm = view.admission
            lines.append(
                f"  {view.name}: pending {adm.pending}/{adm.max_pending}, "
                f"rejected {adm.rejected_since_start}"
            )
    return "\n".join(lines)
