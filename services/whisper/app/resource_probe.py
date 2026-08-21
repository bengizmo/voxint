"""Torch-free-IMPORT background GPU/CPU sampler feeding the /healthz resources block.

Vendored BYTE-IDENTICALLY into each GPU service image (whisper / pyannote /
titanet); a contract test asserts the three copies match. This is the sampler
half of the telemetry pair; the wire models live in ``resource_models.py``.

Two hard invariants (docs/gpu-contracts.md):

- **Import-time pure.** ``torch``, ``pynvml`` (nvidia-ml-py), and the ctypes
  CUDA-driver calls are imported/loaded lazily inside methods, never at module
  scope. Importing this file requires no GPU stack. ``schemas.py`` still must
  not import it (only ``resource_models``), keeping the contract-test boundary
  structural.
- **Telemetry never changes readiness.** Every NVML/torch/ctypes touch is
  wrapped so a failure downgrades ``availability`` and keeps serving the last
  good snapshot. ``/healthz`` requires only that a snapshot object exists, never
  that the latest sample succeeded. The sampler thread catches ``Exception`` per
  tick and never dies.

GPU-by-UUID resolution avoids the NVML-index-vs-torch-index trap: NVML physical
indices differ from torch's ``CUDA_VISIBLE_DEVICES``-remapped ordinals, so a
naive "GPU 0" read can report the WRONG card. Torch 2.1.x exposes neither
``get_device_uuid`` nor ``get_device_properties().uuid`` (added ~2.5), so the
UUID is resolved in layers: torch device-property UUID -> CUDA driver API
(``cuDeviceGetUuid_v2`` via ctypes, which honors CUDA visibility) -> a
single-GPU index-0 safety net (unambiguous when exactly one card is visible) ->
else ``unsupported``. The maintainer GPU gate verifies the live path; CI runs
the logic against fakes.
"""

import contextlib
import ctypes
import logging
import math
import os
import threading
import time
import uuid as uuidlib
from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.resource_models import Admission, CpuTelemetry, GpuTelemetry, Resources

logger = logging.getLogger(__name__)

# Captured once at import ~= process start; the admission block reports it so a
# consumer can rate ``rejected_since_start``.
PROCESS_STARTED_AT = datetime.now(tz=UTC).isoformat()

# Service-env knobs (fail-soft; a malformed value must never crash-loop a
# service or fail /healthz -- it falls back to the default).
_ENABLED_ENV = "VOXINT_TELEMETRY_ENABLED"
_INTERVAL_ENV = "VOXINT_TELEMETRY_INTERVAL_SECONDS"
_DEFAULT_INTERVAL = 5.0
_MIN_INTERVAL = 0.5
_MAX_INTERVAL = 3600.0

# NVML clock-throttle bits -> normalized labels. Names are resolved by getattr
# at sample time against BOTH the modern ``nvmlClocksEventReason*`` spelling and
# the legacy ``nvmlClocksThrottleReason*`` alias, so a version skew in the
# installed nvidia-ml-py wheel drops the unknown bit instead of raising.
_THROTTLE_BIT_NAMES: list[tuple[tuple[str, ...], str]] = [
    (("SwThermalSlowdown",), "thermal_sw"),
    (("HwThermalSlowdown",), "thermal_hw"),
    (("HwPowerBrakeSlowdown", "SwPowerCap"), "power"),
    (("GpuIdle",), "idle"),
    (
        (
            "ApplicationsClocksSetting",
            "SyncBoost",
            "DisplayClockSetting",
        ),
        "clock",
    ),
]


def _parse_enabled(raw: str | None) -> bool:
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _parse_interval(raw: str | None) -> float:
    if raw is None or not raw.strip():
        return _DEFAULT_INTERVAL
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_INTERVAL
    if not math.isfinite(value) or value <= 0:
        return _DEFAULT_INTERVAL
    return max(_MIN_INTERVAL, min(_MAX_INTERVAL, value))


def _canonical_uuid(raw: object) -> str | None:
    """Normalize a torch/driver UUID to ``GPU-<8-4-4-4-12>`` (lowercase)."""
    try:
        if isinstance(raw, uuidlib.UUID):
            hex_form = str(raw)
        elif isinstance(raw, bytes):
            hex_form = str(uuidlib.UUID(bytes=raw))
        else:
            text = str(raw).strip()
            if text.lower().startswith("gpu-"):
                text = text[4:]
            hex_form = str(uuidlib.UUID(text))
    except (ValueError, TypeError):
        return None
    return f"GPU-{hex_form.lower()}"


def _normalize_uuid(raw: object) -> str | None:
    """Comparison key: the canonical lowercase form without the ``GPU-`` tag."""
    canonical = _canonical_uuid(raw)
    return canonical[4:] if canonical else None


def _finite_int(value: object) -> int | None:
    """Coerce to a non-negative int; reject None/NaN/inf/negative/garbage."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return int(number)


def _cuda_driver_uuid(ordinal: int) -> str | None:
    """CUDA driver API UUID for a logical device ordinal, via ctypes.

    ``ordinal`` is torch's ``current_device()`` -- an index into the
    CUDA-visible set. The driver API honors ``CUDA_VISIBLE_DEVICES`` /
    ``CUDA_DEVICE_ORDER`` the same way, so ``cuDeviceGet(ordinal)`` returns the
    same physical device torch is on; we never parse those env vars and never
    treat an NVML physical index as a logical ordinal. Best-effort and
    fail-soft: any failure returns None. MIG devices whose driver lacks the
    v2 UUID call return None (reported ``unsupported``), never the parent card.
    """
    try:
        lib = ctypes.CDLL("libcuda.so.1")
    except OSError:
        try:
            lib = ctypes.CDLL("libcuda.so")
        except OSError:
            return None
    try:
        # Declare ctypes signatures. Without argtypes/restype, ctypes guesses the
        # marshalling of these pointer arguments, which is undefined behaviour: it
        # can return a garbage UUID or fault natively (a native crash would take
        # the whole process down, breaking the "sampler never dies" invariant).
        # CUresult is a C int; CUdevice is a C int.
        uuid16 = ctypes.c_ubyte * 16
        get_uuid = getattr(lib, "cuDeviceGetUuid_v2", None) or getattr(
            lib, "cuDeviceGetUuid", None
        )
        if get_uuid is None:
            return None
        lib.cuInit.restype = ctypes.c_int
        lib.cuInit.argtypes = [ctypes.c_uint]
        lib.cuDeviceGet.restype = ctypes.c_int
        lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        get_uuid.restype = ctypes.c_int
        get_uuid.argtypes = [ctypes.POINTER(uuid16), ctypes.c_int]
        # cuInit is idempotent: torch initialises the driver at model load, well
        # before the sampler starts, so this just no-ops. Do NOT gate on its
        # return code (a repeated init does not reliably report plain success and
        # the exact "already initialised" code varies) -- rely on the typed
        # cuDeviceGet / UUID calls below for the real error signal.
        lib.cuInit(0)
        dev = ctypes.c_int(0)
        if lib.cuDeviceGet(ctypes.byref(dev), int(ordinal)) != 0:
            return None
        raw = uuid16()
        if get_uuid(ctypes.byref(raw), dev) != 0:
            return None
        return _canonical_uuid(bytes(raw))
    except (OSError, AttributeError, ValueError):
        return None


def _torch_gpu_uuid() -> str | None:
    """Best-effort UUID of the GPU this process's torch is using. Never raises."""
    try:
        import torch
    except Exception:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        ordinal = int(torch.cuda.current_device())
    except Exception:
        return None
    # Fast path: torch device-property UUID (torch >= ~2.5).
    try:
        raw = getattr(torch.cuda.get_device_properties(ordinal), "uuid", None)
        if raw:
            canonical = _canonical_uuid(raw)
            if canonical:
                return canonical
    except Exception:
        pass
    # Fallback: CUDA driver API for torch 2.1.x (no property UUID).
    return _cuda_driver_uuid(ordinal)


@dataclass
class _GpuSample:
    """Immutable per-tick GPU reading, assembled fully then atomically swapped
    into the cache. ``sample_age_seconds`` is deliberately absent: it is
    computed at serve time from ``monotonic_at``, never stored."""

    availability: str
    monotonic_at: float
    gpu_uuid: str | None = None
    utilization_percent: int | None = None
    vram_used_bytes: int | None = None
    vram_total_bytes: int | None = None
    temperature_celsius: int | None = None
    throttle_active: bool | None = None
    throttle_reasons: list[str] = field(default_factory=list)
    max_temperature_celsius: int | None = None
    throttle_events_since_start: int | None = None


def _decode_throttle(pynvml: object, mask: int) -> list[str]:
    """Decode an NVML throttle bitmask to sorted normalized labels.

    Unknown/future bits are omitted (never surfaced). Constant names are
    resolved by getattr against both the modern event-reason and legacy
    throttle-reason spellings so a wheel skew degrades gracefully.
    """
    labels: set[str] = set()
    for suffixes, label in _THROTTLE_BIT_NAMES:
        for suffix in suffixes:
            bit = getattr(pynvml, f"nvmlClocksEventReason{suffix}", None)
            if bit is None:
                bit = getattr(pynvml, f"nvmlClocksThrottleReason{suffix}", None)
            if bit is not None and (mask & int(bit)):
                labels.add(label)
    return sorted(labels)


class ResourceSampler:
    """Background NVML/CPU sampler; ``/healthz`` serves its cached snapshot.

    Lifecycle: construct once at import, :meth:`start` in the FastAPI lifespan
    (which takes one synchronous fail-soft sample before spawning the daemon
    loop, so the first ``/healthz`` after readiness never reports ``ok`` with
    all-null values), :meth:`stop` in the lifespan ``finally``.
    """

    def __init__(
        self, *, enabled: bool | None = None, interval_seconds: float | None = None
    ) -> None:
        self._enabled = (
            _parse_enabled(os.getenv(_ENABLED_ENV)) if enabled is None else enabled
        )
        self._interval = (
            _parse_interval(os.getenv(_INTERVAL_ENV))
            if interval_seconds is None
            else _parse_interval(str(interval_seconds))
        )
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Cumulative-since-start state (survives across ticks).
        self._max_temp: int | None = None
        self._throttle_events = 0
        self._prev_throttled = False
        # Seed with a disabled/unsupported snapshot so /healthz always has one.
        seed = "disabled" if not self._enabled else "unsupported"
        self._sample = _GpuSample(availability=seed, monotonic_at=time.monotonic())

    def start(self) -> None:
        if not self._enabled:
            return
        self._sample_once()  # synchronous first read (fail-soft)
        self._thread = threading.Thread(
            target=self._loop, name="resource-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample_once()

    def _sample_once(self) -> None:
        """One fail-soft tick. Assembles a complete sample, then atomically
        swaps it in; a mid-tick failure keeps the last good snapshot."""
        try:
            sample = self._read_gpu()
        except Exception:
            logger.debug("resource sample failed; keeping last snapshot", exc_info=True)
            return
        with self._lock:
            self._sample = sample

    def _read_gpu(self) -> _GpuSample:
        now = time.monotonic()
        try:
            import pynvml
        except Exception:
            return _GpuSample(availability="unsupported", monotonic_at=now)
        try:
            pynvml.nvmlInit()
        except Exception:
            return _GpuSample(availability="unsupported", monotonic_at=now)
        try:
            handle, gpu_uuid = self._resolve_handle(pynvml)
            if handle is None or gpu_uuid is None:
                # No resolvable UUID: the app dedups a shared card by UUID, so an
                # "ok" sample without one would be dropped by aggregation while
                # still claiming telemetry is available. Report unsupported so the
                # wire stays honest end to end.
                return _GpuSample(availability="unsupported", monotonic_at=now)
            util = self._read_util(pynvml, handle)
            used, total = self._read_memory(pynvml, handle)
            temp = self._read_temp(pynvml, handle)
            throttle = self._read_throttle(pynvml, handle)
            if throttle is None:
                throttle_active, reasons = None, []
            else:
                # throttle_active reflects the RAW mask (any bit set), so an
                # unknown/future throttle bit is still surfaced as "throttling"
                # even though it decodes to no known label.
                throttle_active, reasons = throttle
            # Cumulative bookkeeping: count rising edges of a real slowdown
            # (a known non-idle reason), not every throttled tick and not the
            # benign idle bit; track the peak temperature since start.
            throttled = any(r != "idle" for r in reasons)
            if throttled and not self._prev_throttled:
                self._throttle_events += 1
            self._prev_throttled = throttled
            if temp is not None:
                self._max_temp = temp if self._max_temp is None else max(self._max_temp, temp)
            return _GpuSample(
                availability="ok",
                monotonic_at=now,
                gpu_uuid=gpu_uuid,
                utilization_percent=util,
                vram_used_bytes=used,
                vram_total_bytes=total,
                temperature_celsius=temp,
                throttle_active=throttle_active,
                throttle_reasons=reasons,
                max_temperature_celsius=self._max_temp,
                throttle_events_since_start=self._throttle_events,
            )
        finally:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()

    def _resolve_handle(self, pynvml: object) -> tuple[object | None, str | None]:
        """(handle, canonical_uuid) for the GPU this service uses, or (None, None).

        Resolution order: torch/driver UUID -> NVML handle by that UUID; else a
        single-visible-GPU index-0 net (unambiguous, so honest even without a
        UUID); else unsupported. The returned UUID always comes from NVML so it
        matches the app-layer dedup key.
        """
        count_fn = getattr(pynvml, "nvmlDeviceGetCount", None)
        by_index = getattr(pynvml, "nvmlDeviceGetHandleByIndex", None)
        get_uuid = getattr(pynvml, "nvmlDeviceGetUUID", None)
        if count_fn is None or by_index is None or get_uuid is None:
            return None, None

        target = _torch_gpu_uuid()
        if target is not None:
            by_uuid = getattr(pynvml, "nvmlDeviceGetHandleByUUID", None)
            if by_uuid is not None:
                try:
                    handle = by_uuid(target.encode())
                    return handle, self._nvml_uuid(pynvml, handle) or target
                except Exception:
                    pass
            key = _normalize_uuid(target)
            try:
                for i in range(int(count_fn())):
                    handle = by_index(i)
                    if _normalize_uuid(self._nvml_uuid(pynvml, handle)) == key:
                        return handle, self._nvml_uuid(pynvml, handle) or target
            except Exception:
                pass

        # Single visible GPU: index 0 is unambiguous, no remap trap possible.
        try:
            if int(count_fn()) == 1:
                handle = by_index(0)
                return handle, self._nvml_uuid(pynvml, handle)
        except Exception:
            pass
        return None, None

    @staticmethod
    def _nvml_uuid(pynvml: object, handle: object) -> str | None:
        try:
            raw = pynvml.nvmlDeviceGetUUID(handle)  # type: ignore[attr-defined]
        except Exception:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode(errors="ignore")
        return _canonical_uuid(raw)

    @staticmethod
    def _read_util(pynvml: object, handle: object) -> int | None:
        try:
            rates = pynvml.nvmlDeviceGetUtilizationRates(handle)  # type: ignore[attr-defined]
            value = _finite_int(getattr(rates, "gpu", None))
        except Exception:
            return None
        return None if value is None else min(100, value)

    @staticmethod
    def _read_memory(pynvml: object, handle: object) -> tuple[int | None, int | None]:
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)  # type: ignore[attr-defined]
            used = _finite_int(getattr(mem, "used", None))
            total = _finite_int(getattr(mem, "total", None))
        except Exception:
            return None, None
        if used is not None and total is not None and used > total:
            used = None  # impossible pair -> drop the used figure, keep total
        return used, total

    @staticmethod
    def _read_temp(pynvml: object, handle: object) -> int | None:
        try:
            sensor = getattr(pynvml, "NVML_TEMPERATURE_GPU", 0)
            return _finite_int(pynvml.nvmlDeviceGetTemperature(handle, sensor))  # type: ignore[attr-defined]
        except Exception:
            return None

    def _read_throttle(self, pynvml: object, handle: object) -> tuple[bool, list[str]] | None:
        """(throttle_active, normalized_reasons) or None if NVML can't report it.

        ``throttle_active`` is ``mask != 0`` (honest even when a bit is unknown);
        reasons are the decoded subset of known bits.
        """
        fn = getattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None)
        if fn is None:
            fn = getattr(pynvml, "nvmlDeviceGetCurrentClocksThrottleReasons", None)
        if fn is None:
            return None
        try:
            mask = int(fn(handle))
        except Exception:
            return None
        return mask != 0, _decode_throttle(pynvml, mask)

    def gpu(self) -> GpuTelemetry:
        """The cached GPU telemetry with a fresh, clamped ``sample_age_seconds``."""
        with self._lock:
            sample = self._sample
        age = max(0.0, time.monotonic() - sample.monotonic_at)
        return GpuTelemetry(
            availability=sample.availability,  # type: ignore[arg-type]
            gpu_uuid=sample.gpu_uuid,
            utilization_percent=sample.utilization_percent,
            vram_used_bytes=sample.vram_used_bytes,
            vram_total_bytes=sample.vram_total_bytes,
            temperature_celsius=sample.temperature_celsius,
            throttle_active=sample.throttle_active,
            throttle_reasons=list(sample.throttle_reasons),  # type: ignore[arg-type]
            max_temperature_celsius=sample.max_temperature_celsius,
            throttle_events_since_start=sample.throttle_events_since_start,
            sample_age_seconds=age if sample.availability == "ok" else None,
        )

    def cpu(self) -> CpuTelemetry:
        """Host-visible CPU advisory from the stdlib (ignores cgroup quota)."""
        cores = os.cpu_count()
        load: float | None
        try:
            load = float(os.getloadavg()[0])
            if not math.isfinite(load) or load < 0:
                load = None
        except (OSError, AttributeError, ValueError):
            load = None
        available = cores is not None or load is not None
        return CpuTelemetry(
            availability="ok" if available else "unsupported",
            logical_cores=cores,
            load_average_1m=load,
        )


def build_admission(pending: int, max_pending: int, rejected_since_start: int) -> Admission:
    """Assemble the instantaneous admission block (read under the service lock)."""
    return Admission(
        pending=max(0, pending),
        max_pending=max(0, max_pending),
        rejected_since_start=max(0, rejected_since_start),
        process_started_at=PROCESS_STARTED_AT,
    )


def build_resources(sampler: ResourceSampler, admission: Admission) -> Resources:
    """Assemble the full /healthz resources block. Fail-soft: any error yields a
    minimal ``unsupported`` GPU block so telemetry can never fail readiness."""
    try:
        gpu = sampler.gpu()
    except Exception:
        logger.debug("gpu telemetry assembly failed", exc_info=True)
        gpu = GpuTelemetry(availability="unsupported")
    try:
        cpu = sampler.cpu()
    except Exception:
        cpu = None
    return Resources(gpu=gpu, admission=admission, cpu=cpu)
