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
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

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

    @model_validator(mode="after")
    def _drop_impossible_used(self) -> "GpuInfo":
        # Mirror the service-side backstop: a stale or buggy upstream must not
        # paint used>total into the UI. Drop the used figure, never raise.
        used, total = self.vram_used_bytes, self.vram_total_bytes
        if used is not None and total is not None and used > total:
            object.__setattr__(self, "vram_used_bytes", None)
        return self


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
    with _cache_lock:
        now = time.monotonic()
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
        # Stamp AFTER probing, so the cache age counts from when the data was
        # actually collected, not from before a multi-second probe.
        ts = time.monotonic()
        snap = _build_snapshot(healths)
        _cache = (snap, ts)
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
        if gpu.throttle_active is None:
            throttle = "throttle unknown"
        elif gpu.throttle_active:
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


# --------------------------------------------------------------------------- #
# Curated visibility layer (W3): one snapshot -> strip / page / json / metrics.
# --------------------------------------------------------------------------- #

# Device-activity thresholds. Fixed internal constants, NOT operator knobs: the
# strip shows *activity*, never an alarm, so these only pick the wording of a
# neutral pill and a few percent either way changes nothing an operator acts on.
_IDLE_UTIL_MAX = 15  # <= this is "idle"
_BUSY_UTIL_MIN = 90  # >= this is "busy"

# The only throttle reasons that raise a thermal warning. Power/clock/idle
# throttling is normal governor behaviour and is never surfaced as a problem
# (guardrail from the plan consult: thermal is warn-only in v1, driven solely by
# the driver's own thermal verdict, never by an invented temperature threshold).
_THERMAL_REASONS = frozenset({"thermal_sw", "thermal_hw"})


@dataclass(frozen=True)
class GpuActivity:
    """One GPU's neutral, informational activity state for the strip."""

    gpu_uuid: str
    short_uuid: str
    state: str  # "idle" | "working" | "busy" | "unknown"
    utilization_percent: int | None
    services: tuple[str, ...]


@dataclass(frozen=True)
class ResourceWarning:
    """A single curated, actionable warning. Amber, never red: the NVIDIA driver
    already protects the hardware, so v1 warns and advises rather than acting."""

    kind: str  # "thermal" | "queue_full"
    scope: str  # which GPU (short uuid) or service the warning is about
    message: str  # plain-language statement of what is happening
    remedy: str  # one concrete next step a non-technical operator can take


@dataclass(frozen=True)
class ResourceStripView:
    """The compact dashboard strip: neutral GPU activity plus active warnings.

    ``telemetry_present`` separates "no warnings" (calm, telemetry was seen) from
    "status unavailable" (no service reported anything) so the strip never claims
    all-clear on missing data.
    """

    telemetry_present: bool
    gpus: tuple[GpuActivity, ...]
    warnings: tuple[ResourceWarning, ...]
    collected_age_seconds: float


def device_state(util: int | None) -> str:
    """Neutral activity label for a utilization reading (idle/working/busy), or
    "unknown" when there is no reading. Informational, never an alarm."""
    if util is None:
        return "unknown"
    if util <= _IDLE_UTIL_MAX:
        return "idle"
    if util >= _BUSY_UTIL_MIN:
        return "busy"
    return "working"


# Back-compat private alias (kept so existing call sites / tests need no churn).
_device_state = device_state


def _is_thermal_throttling(gpu: AggregatedGpu) -> bool:
    """True only when the driver reports it is throttling *for heat* — a
    non-thermal throttle (power/clock/idle) is normal and never warns."""
    return bool(gpu.throttle_active) and not _THERMAL_REASONS.isdisjoint(gpu.throttle_reasons)


def build_resource_strip(snapshot: ResourceSnapshot) -> ResourceStripView:
    """Curate the snapshot into the compact dashboard strip (pure).

    Only two conditions raise an (amber) warning: the driver reports thermal
    throttling, or a service's admission queue is currently full. High VRAM,
    100% utilization, and cumulative-since-restart counters are deliberately NOT
    warnings (they are normal during a run, and cumulative counts are the wrong
    tense for a live strip); those live on the resource page as instantaneous /
    cumulative context instead.
    """
    gpus = tuple(
        GpuActivity(
            gpu_uuid=g.gpu_uuid,
            short_uuid=_short_uuid(g.gpu_uuid),
            state=device_state(g.utilization_percent),
            utilization_percent=g.utilization_percent,
            services=g.services,
        )
        for g in snapshot.gpus
    )
    warnings: list[ResourceWarning] = []
    for g in snapshot.gpus:
        if _is_thermal_throttling(g):
            warnings.append(
                ResourceWarning(
                    kind="thermal",
                    scope=_short_uuid(g.gpu_uuid),
                    message="The GPU is slowing itself down to stay cool.",
                    remedy=(
                        "Check that the computer's fans and air vents are clear, "
                        "then let it cool down."
                    ),
                )
            )
    for view in snapshot.services:
        adm = view.admission
        if adm is not None and adm.max_pending > 0 and adm.pending >= adm.max_pending:
            warnings.append(
                ResourceWarning(
                    kind="queue_full",
                    scope=view.name,
                    message=f"The {view.name} queue is full ({adm.pending}/{adm.max_pending}).",
                    remedy="Wait for the current work to finish before adding more audio.",
                )
            )
    telemetry_present = bool(snapshot.gpus) or any(
        v.admission is not None for v in snapshot.services
    )
    return ResourceStripView(
        telemetry_present=telemetry_present,
        gpus=gpus,
        warnings=tuple(warnings),
        collected_age_seconds=snapshot.collected_age_seconds,
    )


def short_uuid(uuid: str) -> str:
    """Public alias of :func:`_short_uuid` for template display."""
    return _short_uuid(uuid)


def gib(nbytes: int | None) -> str:
    """Bytes as a GiB number string (no unit suffix), for template display."""
    return f"{nbytes / (1024**3):.1f}" if nbytes is not None else "?"


def vram_percent(used: int | None, total: int | None) -> int | None:
    """Used-VRAM percentage, or None when either figure is missing."""
    if used is None or total is None or total <= 0:
        return None
    return round(100 * used / total)


def resource_snapshot_to_json(snapshot: ResourceSnapshot) -> dict[str, object]:
    """The snapshot as JSON-serialisable primitives (stable key set)."""
    return {
        "collected_age_seconds": snapshot.collected_age_seconds,
        "gpus": [
            {
                "gpu_uuid": g.gpu_uuid,
                "utilization_percent": g.utilization_percent,
                "vram_used_bytes": g.vram_used_bytes,
                "vram_total_bytes": g.vram_total_bytes,
                "temperature_celsius": g.temperature_celsius,
                "throttle_active": g.throttle_active,
                "throttle_reasons": list(g.throttle_reasons),
                "max_temperature_celsius": g.max_temperature_celsius,
                "throttle_events_since_start": g.throttle_events_since_start,
                "sample_age_seconds": g.sample_age_seconds,
                "services": list(g.services),
            }
            for g in snapshot.gpus
        ],
        "services": [
            {
                "name": v.name,
                "up": v.up,
                "device": v.device,
                "telemetry_available": v.telemetry_available,
                "gpu_uuid": v.gpu_uuid,
                "admission": (
                    {
                        "pending": v.admission.pending,
                        "max_pending": v.admission.max_pending,
                        "rejected_since_start": v.admission.rejected_since_start,
                        "process_started_at": v.admission.process_started_at,
                    }
                    if v.admission is not None
                    else None
                ),
                "cpu": (
                    {
                        "availability": v.cpu.availability,
                        "logical_cores": v.cpu.logical_cores,
                        "load_average_1m": v.cpu.load_average_1m,
                    }
                    if v.cpu is not None
                    else None
                ),
            }
            for v in snapshot.services
        ],
    }


def _escape_prom_label(value: str) -> str:
    """Escape a Prometheus label value (backslash, double-quote, newline)."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render_resource_prometheus(snapshot: ResourceSnapshot) -> str:
    """Prometheus text exposition (format 0.0.4) for the hardware telemetry.

    Series are keyed by GPU uuid / service name, discovered at runtime, so unlike
    the DB metrics there is no fixed label set to zero-fill; a metric line is
    emitted only when its value is present (a null reading omits the sample rather
    than inventing a zero). All gauges: every value is a current reading or a
    counter that resets on service restart, so none carries a monotonic-counter
    ``_total`` suffix (``promtool check metrics`` would flag that on these).
    Returns an empty string when no telemetry is present so ``/metrics`` appends
    nothing rather than empty metric families.
    """
    out: list[str] = []

    def _gpu_gauge(name: str, help_text: str, pick: object) -> None:
        assert callable(pick)
        out.append(f"# HELP {name} {help_text}")
        out.append(f"# TYPE {name} gauge")
        for g in snapshot.gpus:
            value = pick(g)
            if value is None:
                continue
            out.append(f'{name}{{gpu="{_escape_prom_label(g.gpu_uuid)}"}} {value}')

    if snapshot.gpus:
        _gpu_gauge(
            "voxint_gpu_utilization_percent",
            "Instantaneous GPU utilization (0-100).",
            lambda g: g.utilization_percent,
        )
        _gpu_gauge(
            "voxint_gpu_vram_used_bytes",
            "GPU memory in use, device-global (bytes).",
            lambda g: g.vram_used_bytes,
        )
        _gpu_gauge(
            "voxint_gpu_vram_total_bytes",
            "Total GPU memory (bytes).",
            lambda g: g.vram_total_bytes,
        )
        _gpu_gauge(
            "voxint_gpu_temperature_celsius",
            "Instantaneous GPU temperature (Celsius).",
            lambda g: g.temperature_celsius,
        )
        _gpu_gauge(
            "voxint_gpu_max_temperature_celsius",
            "Peak GPU temperature since service start (Celsius).",
            lambda g: g.max_temperature_celsius,
        )
        _gpu_gauge(
            "voxint_gpu_throttle_active",
            "1 if the GPU is currently throttling, else 0.",
            lambda g: None if g.throttle_active is None else int(g.throttle_active),
        )
        _gpu_gauge(
            "voxint_gpu_throttle_events_since_start",
            "Throttle events counted since service start.",
            lambda g: g.throttle_events_since_start,
        )

    admission_services = [v for v in snapshot.services if v.admission is not None]
    if admission_services:
        for metric, help_text, field in (
            (
                "voxint_service_admission_pending",
                "In-flight and queued requests at the service.",
                "pending",
            ),
            (
                "voxint_service_admission_max_pending",
                "Admission ceiling before the service rejects.",
                "max_pending",
            ),
            (
                "voxint_service_admission_rejected_since_start",
                "Requests rejected since service start.",
                "rejected_since_start",
            ),
        ):
            out.append(f"# HELP {metric} {help_text}")
            out.append(f"# TYPE {metric} gauge")
            for v in admission_services:
                assert v.admission is not None  # narrowed by the filter above
                value = getattr(v.admission, field)
                out.append(f'{metric}{{service="{_escape_prom_label(v.name)}"}} {value}')

    return "\n".join(out) + "\n" if out else ""
