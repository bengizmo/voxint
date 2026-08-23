"""Contract + unit tests for the /healthz hardware-telemetry block (W1).

Coverage:
- the two vendored files stay byte-identical across the three service images,
  and schemas.py imports only the torch-free models, never the sampler;
- the wire models import without torch/NVML and enforce their bounds;
- the sampler resolves the GPU by UUID (never by NVML index), degrades every
  torch/NVML failure to honest nulls without touching readiness, sanitizes
  NaN/inf, tracks cumulative counters with rising-edge throttle semantics, and
  parses its env fail-soft;
- compose forwards the telemetry env to every model service.
"""

import hashlib
import sys
from types import SimpleNamespace
from typing import ClassVar

import pytest

from tests.contracts.conftest import REPO_ROOT, load_service_module

SERVICES = ("whisper", "pyannote", "titanet")

# Loaded once via the service-package swap harness; identical across services.
models = load_service_module("whisper", "resource_models")
probe = load_service_module("whisper", "resource_probe")


# --------------------------------------------------------------------------- #
# Vendoring: the pair stays byte-identical and the torch-free boundary is
# structural (schemas.py imports the models, never the sampler).
# --------------------------------------------------------------------------- #
class TestVendoredIdentity:
    @pytest.mark.parametrize("fname", ["resource_models.py", "resource_probe.py"])
    def test_identical_across_services(self, fname: str) -> None:
        digests = {
            svc: hashlib.sha256(
                (REPO_ROOT / "services" / svc / "app" / fname).read_bytes()
            ).hexdigest()
            for svc in SERVICES
        }
        assert len(set(digests.values())) == 1, f"{fname} diverged: {digests}"

    @staticmethod
    def _import_lines(path: object) -> list[str]:
        return [
            ln for ln in path.read_text().splitlines()  # type: ignore[attr-defined]
            if ln.strip().startswith(("import ", "from "))
        ]

    def test_schemas_imports_models_not_probe(self) -> None:
        # The whole point of splitting the pair: schemas.py (imported torch-free
        # by the contract suite) must never pull in the lazy torch/NVML sampler.
        for svc in SERVICES:
            path = REPO_ROOT / "services" / svc / "app" / "schemas.py"
            imports = self._import_lines(path)
            assert any("from app.resource_models import" in ln for ln in imports), svc
            assert not any("resource_probe" in ln for ln in imports), (
                f"{svc}/schemas.py imports the sampler"
            )

    def test_models_module_has_no_gpu_imports(self) -> None:
        path = REPO_ROOT / "services" / "whisper" / "app" / "resource_models.py"
        imports = "\n".join(self._import_lines(path))
        for banned in ("torch", "pynvml", "ctypes", "nvidia"):
            assert banned not in imports, f"resource_models.py must stay pure: {banned!r}"

    def test_probe_has_no_module_level_gpu_imports(self) -> None:
        # ctypes is stdlib and fine at module scope; torch/pynvml must be lazy
        # so importing the sampler never needs a GPU stack. Only column-0 (i.e.
        # module-level) import lines are flagged; the in-method lazy imports are
        # the whole point.
        src = (REPO_ROOT / "services" / "whisper" / "app" / "resource_probe.py").read_text()
        for line in src.splitlines():
            if line.startswith(("import torch", "import pynvml", "from torch", "from pynvml")):
                raise AssertionError(f"module-level GPU import in sampler: {line!r}")


class TestPython310Compatibility:
    """The model-service images build on ``nvidia/cuda:*-ubuntu22.04`` (system
    Python 3.10). A 3.11+-only construct in the service app code crash-loops the
    CUDA image at import: v0.22.0 shipped ``from datetime import UTC`` in
    ``resource_probe.py`` and every CUDA model service failed to boot. The
    ``-cpu``/``-rocm`` bases are 3.11+ so only the CUDA lane broke, and the host
    test runner is 3.12 so importing the module here never caught it. Guard the
    source statically instead. When a new 3.11+-only name bites, add it here.
    """

    # substring -> why it is 3.11+-only
    _BANNED: ClassVar[dict[str, str]] = {
        "from datetime import UTC": "datetime.UTC is 3.11+; use timezone.utc",
        "datetime.UTC": "datetime.UTC is 3.11+; use timezone.utc",
    }

    @pytest.mark.parametrize("svc", SERVICES)
    def test_service_app_is_py310_safe(self, svc: str) -> None:
        app_dir = REPO_ROOT / "services" / svc / "app"
        offenders = [
            f"{path.relative_to(REPO_ROOT)}: {needle!r} ({reason})"
            for path in sorted(app_dir.rglob("*.py"))
            for needle, reason in self._BANNED.items()
            if needle in path.read_text()
        ]
        assert not offenders, (
            "model-service code must stay Python 3.10-compatible (CUDA base):\n"
            + "\n".join(offenders)
        )


class TestTorchFreeImport:
    def test_models_import_without_torch_or_nvml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # sys.modules[name] = None makes `import name` raise ImportError.
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(sys.modules, "pynvml", None)
        reloaded = load_service_module("pyannote", "resource_models")
        assert reloaded.GpuTelemetry(availability="unsupported").gpu_uuid is None


# --------------------------------------------------------------------------- #
# Wire models: bounds and the used<=total backstop.
# --------------------------------------------------------------------------- #
class TestWireModels:
    def test_utilization_bounded(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            models.GpuTelemetry(availability="ok", utilization_percent=101)
        with pytest.raises(ValidationError):
            models.GpuTelemetry(availability="ok", utilization_percent=-1)

    def test_negative_bytes_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            models.GpuTelemetry(availability="ok", vram_used_bytes=-1)

    def test_used_over_total_dropped_not_raised(self) -> None:
        gpu = models.GpuTelemetry(
            availability="ok", vram_used_bytes=100, vram_total_bytes=50
        )
        assert gpu.vram_used_bytes is None  # dropped
        assert gpu.vram_total_bytes == 50  # total kept

    def test_unknown_throttle_label_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            models.GpuTelemetry(availability="ok", throttle_reasons=["meltdown"])

    def test_resources_requires_gpu_and_admission_cpu_optional(self) -> None:
        adm = models.Admission(
            pending=0, max_pending=8, rejected_since_start=0, process_started_at="t"
        )
        res = models.Resources(gpu=models.GpuTelemetry(availability="unsupported"), admission=adm)
        assert res.cpu is None


# --------------------------------------------------------------------------- #
# Env parsing: fail-soft, never raises.
# --------------------------------------------------------------------------- #
class TestEnvParsing:
    @pytest.mark.parametrize(
        "raw,expected",
        [(None, True), ("1", True), ("true", True), ("0", False), ("false", False), ("", False)],
    )
    def test_enabled(self, raw: str | None, expected: bool) -> None:
        assert probe._parse_enabled(raw) is expected

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (None, 5.0),
            ("", 5.0),
            ("garbage", 5.0),
            ("-5", 5.0),
            ("0", 5.0),
            ("nan", 5.0),
            ("inf", 5.0),
            ("12", 12.0),
            ("0.1", 0.5),  # clamped up to the floor
            ("99999", 3600.0),  # clamped to the ceiling
        ],
    )
    def test_interval(self, raw: str | None, expected: float) -> None:
        assert probe._parse_interval(raw) == expected


# --------------------------------------------------------------------------- #
# Fakes for the NVML/torch surface the sampler touches.
# --------------------------------------------------------------------------- #
class _FakeNVMLError(Exception):
    pass


def _fake_pynvml(
    *,
    uuids: list[str],
    util: object = 55,
    used: int = 4_000_000_000,
    total: int = 12_000_000_000,
    temp: object = 61,
    throttle_mask: int = 0,
    by_uuid_works: bool = True,
    event_reasons: bool = True,
) -> SimpleNamespace:
    """A minimal NVML stand-in. ``uuids`` are the physical cards, index-ordered.
    Handles are their index; UUID lookups compare normalized UUIDs."""

    def norm(u: str) -> str:
        u = u.strip().lower()
        return u[4:] if u.startswith("gpu-") else u

    ns = SimpleNamespace()
    ns.NVMLError = _FakeNVMLError
    ns.NVML_TEMPERATURE_GPU = 0
    ns.nvmlInit = lambda: None
    ns.nvmlShutdown = lambda: None
    ns.nvmlDeviceGetCount = lambda: len(uuids)
    ns.nvmlDeviceGetHandleByIndex = lambda i: i
    ns.nvmlDeviceGetUUID = lambda h: uuids[h]

    def by_uuid(raw: object) -> int:
        if not by_uuid_works:
            raise _FakeNVMLError("not supported")
        key = norm(raw.decode() if isinstance(raw, bytes) else str(raw))
        for i, u in enumerate(uuids):
            if norm(u) == key:
                return i
        raise _FakeNVMLError("not found")

    ns.nvmlDeviceGetHandleByUUID = by_uuid
    ns.nvmlDeviceGetUtilizationRates = lambda h: SimpleNamespace(gpu=util)
    ns.nvmlDeviceGetMemoryInfo = lambda h: SimpleNamespace(used=used, total=total)
    ns.nvmlDeviceGetTemperature = lambda h, s: temp
    if event_reasons:
        ns.nvmlDeviceGetCurrentClocksEventReasons = lambda h: throttle_mask
        ns.nvmlClocksEventReasonSwThermalSlowdown = 0x1
        ns.nvmlClocksEventReasonHwThermalSlowdown = 0x2
        ns.nvmlClocksEventReasonSwPowerCap = 0x4
        ns.nvmlClocksEventReasonGpuIdle = 0x8
    return ns


def _fake_torch(uuid: str | None) -> SimpleNamespace:
    """torch whose current device reports ``uuid`` via device properties
    (the torch>=2.5 fast path). ``None`` models no-CUDA / no-UUID."""
    cuda = SimpleNamespace(
        is_available=lambda: uuid is not None,
        current_device=lambda: 0,
        get_device_properties=lambda i: SimpleNamespace(uuid=uuid),
    )
    return SimpleNamespace(cuda=cuda)


UUID_A = "GPU-aaaaaaaa-1111-2222-3333-444444444444"
UUID_B = "GPU-bbbbbbbb-5555-6666-7777-888888888888"


def _install(monkeypatch: pytest.MonkeyPatch, pynvml: object, torch: object | None) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)
    if torch is None:
        monkeypatch.setitem(sys.modules, "torch", None)
    else:
        monkeypatch.setitem(sys.modules, "torch", torch)


# --------------------------------------------------------------------------- #
# Sampler: resolution paths, degradation, sanitation, cumulative counters.
# --------------------------------------------------------------------------- #
class TestSamplerResolution:
    def test_by_uuid_direct_multi_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Two cards; torch is on B. The UUID path must pick B, never index 0.
        _install(monkeypatch, _fake_pynvml(uuids=[UUID_A, UUID_B]), _fake_torch(UUID_B))
        smp = probe.ResourceSampler(enabled=True, interval_seconds=999)
        smp._sample_once()
        gpu = smp.gpu()
        assert gpu.availability == "ok"
        assert probe._normalize_uuid(gpu.gpu_uuid) == probe._normalize_uuid(UUID_B)
        assert gpu.utilization_percent == 55
        assert gpu.vram_used_bytes == 4_000_000_000
        assert gpu.sample_age_seconds is not None and gpu.sample_age_seconds >= 0

    def test_enumerate_fallback_when_by_uuid_unsupported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(
            monkeypatch,
            _fake_pynvml(uuids=[UUID_A, UUID_B], by_uuid_works=False),
            _fake_torch(UUID_B),
        )
        smp = probe.ResourceSampler(enabled=True, interval_seconds=999)
        smp._sample_once()
        assert probe._normalize_uuid(smp.gpu().gpu_uuid) == probe._normalize_uuid(UUID_B)

    def test_single_gpu_net_when_torch_has_no_uuid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # torch<2.5 with no driver UUID, but exactly one visible card: index 0
        # is unambiguous, so telemetry is still honest.
        _install(monkeypatch, _fake_pynvml(uuids=[UUID_A]), _fake_torch(None))
        smp = probe.ResourceSampler(enabled=True, interval_seconds=999)
        smp._sample_once()
        gpu = smp.gpu()
        assert gpu.availability == "ok"
        assert probe._normalize_uuid(gpu.gpu_uuid) == probe._normalize_uuid(UUID_A)

    def test_multi_gpu_no_uuid_is_unsupported_not_wrong(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Cannot disambiguate two cards without a UUID: honest "unsupported"
        # beats guessing index 0 and fabricating the wrong card's telemetry.
        _install(monkeypatch, _fake_pynvml(uuids=[UUID_A, UUID_B]), _fake_torch(None))
        smp = probe.ResourceSampler(enabled=True, interval_seconds=999)
        smp._sample_once()
        assert smp.gpu().availability == "unsupported"

    def test_single_gpu_but_unresolvable_uuid_is_unsupported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Single card, but NVML returns a UUID we cannot canonicalize: an "ok"
        # sample with no UUID would be dropped by the app's dedup while still
        # claiming telemetry is available, so the source reports unsupported.
        _install(monkeypatch, _fake_pynvml(uuids=["not-a-uuid"]), _fake_torch(None))
        smp = probe.ResourceSampler(enabled=True, interval_seconds=999)
        smp._sample_once()
        assert smp.gpu().availability == "unsupported"

    def test_nvml_absent_is_unsupported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "pynvml", None)  # import raises
        smp = probe.ResourceSampler(enabled=True, interval_seconds=999)
        smp._sample_once()
        assert smp.gpu().availability == "unsupported"

    def test_disabled_never_starts_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        smp = probe.ResourceSampler(enabled=False)
        smp.start()
        assert smp._thread is None
        assert smp.gpu().availability == "disabled"

    def test_injected_interval_is_clamped(self) -> None:
        # A caller/test passing 0 (or NaN) must not make the loop busy-spin: the
        # injected value runs through the same fail-soft clamp as the env value.
        assert probe.ResourceSampler(interval_seconds=0)._interval == probe._DEFAULT_INTERVAL
        assert probe.ResourceSampler(interval_seconds=float("nan"))._interval == (
            probe._DEFAULT_INTERVAL
        )
        assert probe.ResourceSampler(interval_seconds=0.01)._interval == probe._MIN_INTERVAL


class TestSamplerSanitation:
    def test_nan_util_and_temp_become_null_not_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(
            monkeypatch,
            _fake_pynvml(uuids=[UUID_A], util=float("nan"), temp=float("inf")),
            _fake_torch(UUID_A),
        )
        smp = probe.ResourceSampler(enabled=True, interval_seconds=999)
        smp._sample_once()
        gpu = smp.gpu()
        assert gpu.availability == "ok"
        assert gpu.utilization_percent is None
        assert gpu.temperature_celsius is None

    def test_used_over_total_dropped_at_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(
            monkeypatch,
            _fake_pynvml(uuids=[UUID_A], used=99, total=10),
            _fake_torch(UUID_A),
        )
        smp = probe.ResourceSampler(enabled=True, interval_seconds=999)
        smp._sample_once()
        gpu = smp.gpu()
        assert gpu.vram_used_bytes is None
        assert gpu.vram_total_bytes == 10

    def test_util_clamped_to_100(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, _fake_pynvml(uuids=[UUID_A], util=150), _fake_torch(UUID_A))
        smp = probe.ResourceSampler(enabled=True, interval_seconds=999)
        smp._sample_once()
        assert smp.gpu().utilization_percent == 100


class TestThrottleDecoding:
    def test_known_bits_decode_to_labels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # SwThermalSlowdown (0x1) | SwPowerCap (0x4) -> thermal_sw + power.
        _install(
            monkeypatch,
            _fake_pynvml(uuids=[UUID_A], throttle_mask=0x1 | 0x4),
            _fake_torch(UUID_A),
        )
        smp = probe.ResourceSampler(enabled=True, interval_seconds=999)
        smp._sample_once()
        gpu = smp.gpu()
        assert gpu.throttle_active is True
        assert set(gpu.throttle_reasons) == {"thermal_sw", "power"}

    def test_unknown_bit_sets_active_but_omits_label(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 0x8000 is an unmapped bit: throttle_active True, no fabricated label.
        _install(
            monkeypatch, _fake_pynvml(uuids=[UUID_A], throttle_mask=0x8000), _fake_torch(UUID_A)
        )
        smp = probe.ResourceSampler(enabled=True, interval_seconds=999)
        smp._sample_once()
        gpu = smp.gpu()
        assert gpu.throttle_active is True
        assert gpu.throttle_reasons == []

    def test_idle_bit_does_not_count_as_throttle_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # GpuIdle (0x8) is benign: it surfaces as a reason but must not inflate
        # throttle_events_since_start (which counts real slowdown rising edges).
        _install(
            monkeypatch, _fake_pynvml(uuids=[UUID_A], throttle_mask=0x8), _fake_torch(UUID_A)
        )
        smp = probe.ResourceSampler(enabled=True, interval_seconds=999)
        smp._sample_once()
        gpu = smp.gpu()
        assert "idle" in gpu.throttle_reasons
        assert gpu.throttle_events_since_start == 0


class TestCumulativeCounters:
    def test_max_temp_and_rising_edge_throttle_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = {"temp": 40, "mask": 0x0}
        pynvml = _fake_pynvml(uuids=[UUID_A])
        pynvml.nvmlDeviceGetTemperature = lambda h, s: state["temp"]
        pynvml.nvmlDeviceGetCurrentClocksEventReasons = lambda h: state["mask"]
        _install(monkeypatch, pynvml, _fake_torch(UUID_A))
        smp = probe.ResourceSampler(enabled=True, interval_seconds=999)

        smp._sample_once()  # 40C, not throttled
        state["temp"], state["mask"] = 75, 0x1  # rising edge -> event 1
        smp._sample_once()
        state["temp"], state["mask"] = 60, 0x1  # still throttled -> no new event
        smp._sample_once()
        state["temp"], state["mask"] = 55, 0x0  # clears
        smp._sample_once()
        state["temp"], state["mask"] = 70, 0x1  # rising edge -> event 2
        smp._sample_once()

        gpu = smp.gpu()
        assert gpu.max_temperature_celsius == 75  # peak retained
        assert gpu.throttle_events_since_start == 2


class TestBuildResources:
    def test_admission_survives_gpu_unsupported(self) -> None:
        smp = probe.ResourceSampler(enabled=False)  # gpu -> disabled
        adm = probe.build_admission(3, 8, 5)
        res = probe.build_resources(smp, adm)
        assert res.gpu.availability == "disabled"
        assert res.admission.pending == 3
        assert res.admission.max_pending == 8
        assert res.admission.rejected_since_start == 5
        assert res.admission.process_started_at  # ISO string present

    def test_negative_admission_clamped(self) -> None:
        adm = probe.build_admission(-1, -1, -1)
        assert (adm.pending, adm.max_pending, adm.rejected_since_start) == (0, 0, 0)


# --------------------------------------------------------------------------- #
# HealthResponse tolerance + compose env propagation.
# --------------------------------------------------------------------------- #
class TestHealthResponseTolerance:
    @pytest.mark.parametrize("svc", SERVICES)
    def test_old_body_without_resources_still_parses(self, svc: str) -> None:
        schemas = load_service_module(svc, "schemas")
        body = {
            "status": "ok",
            "version": "1.0.0",
            "model": "m",
            "device": "cuda",
            "engine": "e",
            "model_loaded": True,
        }
        assert schemas.HealthResponse.model_validate(body).resources is None

    @pytest.mark.parametrize("svc", SERVICES)
    def test_body_with_resources_parses(self, svc: str) -> None:
        schemas = load_service_module(svc, "schemas")
        body = {
            "status": "ok",
            "version": "1.0.0",
            "model": "m",
            "device": "cuda",
            "engine": "e",
            "model_loaded": True,
            "resources": {
                "gpu": {"availability": "ok", "utilization_percent": 50},
                "admission": {
                    "pending": 0,
                    "max_pending": 8,
                    "rejected_since_start": 0,
                    "process_started_at": "2026-01-01T00:00:00+00:00",
                },
            },
        }
        parsed = schemas.HealthResponse.model_validate(body)
        assert parsed.resources is not None
        assert parsed.resources.gpu.utilization_percent == 50


_TELEMETRY_OVERLAYS = ["compose.gpu.yaml", "compose.cpu.yaml", "compose.rocm.yaml"]


class TestComposeEnvPropagation:
    @pytest.mark.parametrize("overlay", _TELEMETRY_OVERLAYS)
    def test_telemetry_env_forwarded_to_every_model_service(self, overlay: str) -> None:
        import yaml

        config = yaml.safe_load((REPO_ROOT / overlay).read_text())
        for svc in SERVICES:
            env = config["services"][svc]["environment"]
            assert "VOXINT_TELEMETRY_ENABLED" in env, f"{overlay}:{svc}"
            assert "VOXINT_TELEMETRY_INTERVAL_SECONDS" in env, f"{overlay}:{svc}"


class TestWhisperOverridePassthrough:
    """Every whisper-bearing overlay forwards the model-override knobs and mounts
    the separate alternate-model cache (configurable models A2). HF_HUB_OFFLINE is
    deliberately NOT a pass-through: an empty compose default would flip the baked
    offline behavior, so the resolver owns that flag instead."""

    # A compose ``${KEY:-}`` pass-through does NOT leave a baked image ENV
    # untouched: it SETS the container var to the empty string, shadowing the
    # bake. That is why WHISPER_MODEL carries a real ``large-v2`` default (the
    # startup resolver rejects an empty model), and why the resolver restores the
    # baked revision when WHISPER_REVISION comes through empty. WHISPER_ALLOW_DOWNLOAD
    # and HF_TOKEN are genuine opt-ins whose correct default IS empty.
    _EMPTY_DEFAULTED_KEYS = ("WHISPER_REVISION", "WHISPER_ALLOW_DOWNLOAD", "HF_TOKEN")
    _ALT_CACHE = "/app/model-cache/whisper"

    @pytest.mark.parametrize("overlay", _TELEMETRY_OVERLAYS)
    def test_override_env_forwarded_to_whisper(self, overlay: str) -> None:
        import yaml

        config = yaml.safe_load((REPO_ROOT / overlay).read_text())
        env = config["services"]["whisper"]["environment"]
        # The model must default to the validated large-v2, never empty: an empty
        # ${WHISPER_MODEL:-} fails the startup resolver closed (blank != baked default).
        assert env.get("WHISPER_MODEL") == "${WHISPER_MODEL:-large-v2}", (
            f"{overlay}:whisper WHISPER_MODEL must default to large-v2, not empty"
        )
        for key in self._EMPTY_DEFAULTED_KEYS:
            assert key in env, f"{overlay}:whisper missing {key} pass-through"
            assert env[key] == f"${{{key}:-}}", f"{overlay}:whisper {key} not empty-defaulted"
        # HF_HUB_OFFLINE must not leak in as a pass-through (would flip the baked
        # default when unset in .env).
        assert "HF_HUB_OFFLINE" not in env, f"{overlay}:whisper must not pass HF_HUB_OFFLINE"

    @pytest.mark.parametrize("overlay", _TELEMETRY_OVERLAYS)
    def test_whisper_models_cache_volume_declared_and_mounted(self, overlay: str) -> None:
        import yaml

        config = yaml.safe_load((REPO_ROOT / overlay).read_text())
        assert "whisper-models" in (config.get("volumes") or {}), (
            f"{overlay} missing top-level whisper-models volume"
        )
        mounts = config["services"]["whisper"]["volumes"]
        assert any(m == f"whisper-models:{self._ALT_CACHE}" for m in mounts), (
            f"{overlay}:whisper does not mount whisper-models at {self._ALT_CACHE}"
        )
        # The alternate cache must never shadow the baked large-v2 download root.
        assert self._ALT_CACHE != "/app/.cache/whisper"

    _WHISPER_DOCKERFILES = (
        "services/whisper/Dockerfile",
        "services/whisper/Dockerfile.cpu",
        "services/whisper/Dockerfile.rocm",
    )

    @pytest.mark.parametrize("dockerfile", _WHISPER_DOCKERFILES)
    def test_alt_cache_mountpoint_is_owned_by_runtime_user(self, dockerfile: str) -> None:
        # A fresh named volume mounted at the alt-cache path inherits that path's
        # ownership FROM THE IMAGE. If the image never creates it, Docker makes the
        # mountpoint (and the empty volume) root-owned and the non-root ``voxint``
        # service cannot write an alternate model's download. The image must create
        # the path and chown it to voxint before dropping privileges.
        text = (REPO_ROOT / dockerfile).read_text()
        assert self._ALT_CACHE in text, (
            f"{dockerfile} never creates the alt-cache mountpoint {self._ALT_CACHE}; "
            "a fresh named volume there will be root-owned and unwritable by voxint"
        )
        chown_at = text.find("chown -R voxint:voxint /app")
        user_at = text.find("USER voxint")
        mkdir_at = text.find(self._ALT_CACHE)
        assert chown_at != -1, f"{dockerfile} missing chown -R voxint:voxint /app"
        assert user_at != -1, f"{dockerfile} missing USER voxint"
        # The mountpoint must be created, then chowned, all before USER voxint.
        assert mkdir_at < user_at and chown_at < user_at, (
            f"{dockerfile} must create+chown {self._ALT_CACHE} before USER voxint"
        )


class TestDiarizerOverridePassthrough:
    """Every pyannote-bearing overlay forwards the diarization model-override knobs
    so a deployment-set DIARIZER_REVISION (configurable models A3) actually reaches
    the container. Omitting the revision silently floats the pin at the repo
    default while the operator believes it is pinned."""

    _DIARIZER_KEYS = ("DIARIZER_MODEL_NAME", "DIARIZER_REVISION", "HF_TOKEN")

    @pytest.mark.parametrize("overlay", _TELEMETRY_OVERLAYS)
    def test_override_env_forwarded_to_pyannote(self, overlay: str) -> None:
        import yaml

        config = yaml.safe_load((REPO_ROOT / overlay).read_text())
        env = config["services"]["pyannote"]["environment"]
        for key in self._DIARIZER_KEYS:
            assert key in env, f"{overlay}:pyannote missing {key} pass-through"
            assert env[key] == f"${{{key}:-}}", (
                f"{overlay}:pyannote {key} not empty-defaulted"
            )
