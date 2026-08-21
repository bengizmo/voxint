"""App-side resource-telemetry aggregation (W2) against httpx.MockTransport.

Covers: /healthz telemetry extraction (including the degraded 503 path), the
defensive parse, dedup-by-UUID aggregation of a shared card, the per-service
admission view, the short-TTL single-flight cache, and the text renderer. No
real services involved.
"""

import httpx
import pytest

from voxint.api import resource_status as rs
from voxint.api.health_probe import probe_services
from voxint.config import Settings

_ASR_PORT = 8022  # transcription (whisper)
_DIARIZER_PORT = 8024  # diarization (pyannote)
_EMBEDDER_PORT = 8021  # speaker embedding (titanet)

UUID_A = "GPU-aaaaaaaa-1111-2222-3333-444444444444"
UUID_B = "GPU-bbbbbbbb-5555-6666-7777-888888888888"


@pytest.fixture(autouse=True)
def _clean_cache() -> None:
    rs._reset_cache_for_tests()


def _settings(**kw: object) -> Settings:
    return Settings(voxint_user="u", voxint_password="p", **kw)


def _resources(
    uuid: str,
    *,
    util: int = 50,
    used: int = 4_000_000_000,
    total: int = 12_000_000_000,
    temp: int = 60,
    max_temp: int = 70,
    throttle_events: int = 0,
    sample_age: float = 1.0,
    pending: int = 0,
    rejected: int = 0,
    availability: str = "ok",
) -> dict:
    return {
        "gpu": {
            "availability": availability,
            "gpu_uuid": uuid,
            "utilization_percent": util,
            "vram_used_bytes": used,
            "vram_total_bytes": total,
            "temperature_celsius": temp,
            "throttle_active": False,
            "throttle_reasons": [],
            "max_temperature_celsius": max_temp,
            "throttle_events_since_start": throttle_events,
            "sample_age_seconds": sample_age,
        },
        "admission": {
            "pending": pending,
            "max_pending": 8,
            "rejected_since_start": rejected,
            "process_started_at": "2026-01-01T00:00:00+00:00",
        },
        "cpu": {"availability": "ok", "logical_cores": 24, "load_average_1m": 2.5},
    }


def _healthy(service: str, resources: dict | None) -> httpx.Response:
    body = {"status": "ok", "service": service, "model": f"{service}-m", "model_loaded": True}
    if resources is not None:
        body["resources"] = resources
    return httpx.Response(200, json=body)


def _client(by_port: dict[int, object], counter: list[int] | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/healthz"
        if counter is not None:
            counter[0] += 1
        outcome = by_port[request.url.port or 0]
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, httpx.Response)
        return outcome

    return httpx.Client(transport=httpx.MockTransport(handler))


# --------------------------------------------------------------------------- #
# health_probe telemetry extraction.
# --------------------------------------------------------------------------- #
class TestHealthProbeTelemetry:
    def test_resources_extracted_on_ready(self) -> None:
        client = _client(
            {
                _ASR_PORT: _healthy("whisper", _resources(UUID_A)),
                _DIARIZER_PORT: _healthy("pyannote", _resources(UUID_A)),
                _EMBEDDER_PORT: _healthy("titanet", _resources(UUID_A)),
            }
        )
        by_name = {r.name: r for r in probe_services(_settings(), client=client)}
        assert by_name["transcription"].resources is not None
        assert by_name["transcription"].resources["gpu"]["gpu_uuid"] == UUID_A

    def test_resources_extracted_on_degraded_503(self) -> None:
        # The 503 path used to return before reading the body; telemetry is most
        # useful exactly when a service is degraded.
        degraded = httpx.Response(
            503,
            json={"status": "degraded", "model": None, "resources": _resources(UUID_A, util=0)},
        )
        client = _client(
            {
                _ASR_PORT: degraded,
                _DIARIZER_PORT: _healthy("pyannote", _resources(UUID_A)),
                _EMBEDDER_PORT: _healthy("titanet", _resources(UUID_A)),
            }
        )
        by_name = {r.name: r for r in probe_services(_settings(), client=client)}
        asr = by_name["transcription"]
        assert asr.up is False
        assert asr.resources is not None
        assert asr.resources["gpu"]["gpu_uuid"] == UUID_A

    def test_absent_resources_is_none(self) -> None:
        client = _client(
            {
                _ASR_PORT: _healthy("whisper", None),
                _DIARIZER_PORT: _healthy("pyannote", None),
                _EMBEDDER_PORT: _healthy("titanet", None),
            }
        )
        by_name = {r.name: r for r in probe_services(_settings(), client=client)}
        assert by_name["transcription"].resources is None

    def test_resources_kept_on_parseable_not_ready_2xx(self) -> None:
        # A 200 body that is neither "ok" nor model_loaded=false (e.g. still
        # starting) must still surface the telemetry it carried.
        not_ready = httpx.Response(
            200, json={"status": "starting", "model_loaded": None, "resources": _resources(UUID_A)}
        )
        client = _client(
            {
                _ASR_PORT: not_ready,
                _DIARIZER_PORT: _healthy("pyannote", _resources(UUID_A)),
                _EMBEDDER_PORT: _healthy("titanet", _resources(UUID_A)),
            }
        )
        by_name = {r.name: r for r in probe_services(_settings(), client=client)}
        asr = by_name["transcription"]
        assert asr.up is False
        assert asr.detail == "not ready"
        assert asr.resources is not None and asr.resources["gpu"]["gpu_uuid"] == UUID_A


# --------------------------------------------------------------------------- #
# Defensive parse.
# --------------------------------------------------------------------------- #
class TestParseResources:
    def test_none_and_non_dict(self) -> None:
        assert rs.parse_resources(None) is None
        assert rs.parse_resources("nope") is None
        assert rs.parse_resources(42) is None

    def test_malformed_value_becomes_none(self) -> None:
        bad = _resources(UUID_A)
        bad["gpu"]["utilization_percent"] = 150  # out of range -> whole block unavailable
        assert rs.parse_resources(bad) is None

    def test_missing_admission_becomes_none(self) -> None:
        no_adm = {"gpu": {"availability": "unsupported"}}
        assert rs.parse_resources(no_adm) is None

    def test_unknown_throttle_label_tolerated(self) -> None:
        # Forward compatibility: a future service may add a new reason label.
        body = _resources(UUID_A)
        body["gpu"]["throttle_reasons"] = ["some_future_reason"]
        parsed = rs.parse_resources(body)
        assert parsed is not None
        assert parsed.gpu.throttle_reasons == ["some_future_reason"]

    def test_valid_parses(self) -> None:
        parsed = rs.parse_resources(_resources(UUID_A, util=33))
        assert parsed is not None
        assert parsed.gpu.utilization_percent == 33
        assert parsed.cpu is not None and parsed.cpu.logical_cores == 24

    def test_used_over_total_dropped_on_app_parse(self) -> None:
        # Defence in depth: a stale/buggy upstream sending used>total must not
        # paint the impossible pair into the UI.
        body = _resources(UUID_A, used=99, total=10)
        parsed = rs.parse_resources(body)
        assert parsed is not None
        assert parsed.gpu.vram_used_bytes is None
        assert parsed.gpu.vram_total_bytes == 10


# --------------------------------------------------------------------------- #
# Aggregation.
# --------------------------------------------------------------------------- #
def _collect(client: httpx.Client, **settings_kw: object) -> rs.ResourceSnapshot:
    return rs.collect_resource_status(_settings(**settings_kw), client=client, force=True)


class TestAggregation:
    def test_shared_card_deduped_to_one_gpu(self) -> None:
        # All three services report the same physical card at different sample
        # ages and cumulative counters. Expect one GPU, all three service names,
        # the freshest instantaneous reading, and the max cumulative counters.
        client = _client(
            {
                _ASR_PORT: _healthy(
                    "whisper",
                    _resources(UUID_A, util=80, sample_age=0.5, max_temp=72, throttle_events=1),
                ),
                _DIARIZER_PORT: _healthy(
                    "pyannote",
                    _resources(UUID_A, util=10, sample_age=3.0, max_temp=68, throttle_events=0),
                ),
                _EMBEDDER_PORT: _healthy(
                    "titanet",
                    _resources(UUID_A, util=40, sample_age=2.0, max_temp=70, throttle_events=2),
                ),
            }
        )
        snap = _collect(client)
        assert len(snap.gpus) == 1
        gpu = snap.gpus[0]
        assert rs._short_uuid(gpu.gpu_uuid) == "aaaaaaaa"
        assert set(gpu.services) == {"transcription", "diarization", "speaker embedding"}
        assert gpu.utilization_percent == 80  # freshest (0.5s) reading
        assert gpu.max_temperature_celsius == 72  # max across services
        assert gpu.throttle_events_since_start == 2  # max across services

    def test_distinct_cards_stay_separate(self) -> None:
        client = _client(
            {
                _ASR_PORT: _healthy("whisper", _resources(UUID_A)),
                _DIARIZER_PORT: _healthy("pyannote", _resources(UUID_B)),
                _EMBEDDER_PORT: _healthy("titanet", _resources(UUID_A)),
            }
        )
        snap = _collect(client)
        assert len(snap.gpus) == 2
        uuids = {rs._short_uuid(g.gpu_uuid) for g in snap.gpus}
        assert uuids == {"aaaaaaaa", "bbbbbbbb"}

    def test_service_without_telemetry_still_a_view(self) -> None:
        client = _client(
            {
                _ASR_PORT: _healthy("whisper", None),  # no telemetry
                _DIARIZER_PORT: _healthy("pyannote", _resources(UUID_A)),
                _EMBEDDER_PORT: _healthy("titanet", _resources(UUID_A)),
            }
        )
        snap = _collect(client)
        by_name = {v.name: v for v in snap.services}
        assert by_name["transcription"].telemetry_available is False
        assert by_name["transcription"].admission is None
        assert by_name["diarization"].telemetry_available is True
        assert by_name["diarization"].admission is not None

    def test_unsupported_gpu_not_aggregated_but_admission_kept(self) -> None:
        unsupported = _resources(UUID_A, availability="unsupported")
        client = _client(
            {
                _ASR_PORT: _healthy("whisper", unsupported),
                _DIARIZER_PORT: _healthy("pyannote", unsupported),
                _EMBEDDER_PORT: _healthy("titanet", unsupported),
            }
        )
        snap = _collect(client)
        assert snap.gpus == ()
        # admission still surfaces even though GPU telemetry is unsupported
        assert all(v.admission is not None for v in snap.services)


# --------------------------------------------------------------------------- #
# TTL single-flight cache.
# --------------------------------------------------------------------------- #
class TestCache:
    def test_second_call_within_ttl_is_cached(self) -> None:
        counter = [0]
        client = _client(
            {
                _ASR_PORT: _healthy("whisper", _resources(UUID_A)),
                _DIARIZER_PORT: _healthy("pyannote", _resources(UUID_A)),
                _EMBEDDER_PORT: _healthy("titanet", _resources(UUID_A)),
            },
            counter=counter,
        )
        settings = _settings(resource_status_ttl_seconds=100)
        rs.collect_resource_status(settings, client=client)
        first = counter[0]
        assert first == 3  # one probe per service
        rs.collect_resource_status(settings, client=client)
        assert counter[0] == first  # served from cache, no new probes

    def test_force_bypasses_cache(self) -> None:
        counter = [0]
        client = _client(
            {
                _ASR_PORT: _healthy("whisper", _resources(UUID_A)),
                _DIARIZER_PORT: _healthy("pyannote", _resources(UUID_A)),
                _EMBEDDER_PORT: _healthy("titanet", _resources(UUID_A)),
            },
            counter=counter,
        )
        settings = _settings(resource_status_ttl_seconds=100)
        rs.collect_resource_status(settings, client=client, force=True)
        rs.collect_resource_status(settings, client=client, force=True)
        assert counter[0] == 6  # both calls probed

    def test_cached_age_advances(self) -> None:
        client = _client(
            {
                _ASR_PORT: _healthy("whisper", _resources(UUID_A)),
                _DIARIZER_PORT: _healthy("pyannote", _resources(UUID_A)),
                _EMBEDDER_PORT: _healthy("titanet", _resources(UUID_A)),
            }
        )
        settings = _settings(resource_status_ttl_seconds=100)
        first = rs.collect_resource_status(settings, client=client)
        assert first.collected_age_seconds == pytest.approx(0.0, abs=0.5)
        second = rs.collect_resource_status(settings, client=client)
        assert second.collected_age_seconds >= 0.0


# --------------------------------------------------------------------------- #
# Text renderer.
# --------------------------------------------------------------------------- #
class TestTextRenderer:
    def test_renders_gpu_and_admission(self) -> None:
        client = _client(
            {
                _ASR_PORT: _healthy("whisper", _resources(UUID_A, util=77, pending=2, rejected=1)),
                _DIARIZER_PORT: _healthy("pyannote", _resources(UUID_A)),
                _EMBEDDER_PORT: _healthy("titanet", _resources(UUID_A)),
            }
        )
        text = rs.format_resource_status_text(_collect(client))
        assert "Resource telemetry:" in text
        assert "GPU aaaaaaaa" in text
        assert "transcription: pending 2/8, rejected 1" in text

    def test_empty_snapshot_says_unavailable(self) -> None:
        empty = rs.ResourceSnapshot(gpus=(), services=(), collected_age_seconds=0.0)
        assert "unavailable" in rs.format_resource_status_text(empty)


# --------------------------------------------------------------------------- #
# Visibility layer (W3): strip curation, JSON, prometheus, trio agreement.
# Fixtures build the frozen snapshot directly so the pure renderers unit-test
# without any HTTP.
# --------------------------------------------------------------------------- #
def _gpu(
    uuid: str = UUID_A,
    *,
    util: int | None = 50,
    used: int | None = 4_000_000_000,
    total: int | None = 12_000_000_000,
    temp: int | None = 60,
    throttle_active: bool | None = False,
    throttle_reasons: tuple[str, ...] = (),
    max_temp: int | None = 70,
    throttle_events: int | None = 0,
    sample_age: float | None = 1.0,
    services: tuple[str, ...] = ("transcription",),
) -> rs.AggregatedGpu:
    return rs.AggregatedGpu(
        gpu_uuid=uuid,
        utilization_percent=util,
        vram_used_bytes=used,
        vram_total_bytes=total,
        temperature_celsius=temp,
        throttle_active=throttle_active,
        throttle_reasons=throttle_reasons,
        max_temperature_celsius=max_temp,
        throttle_events_since_start=throttle_events,
        sample_age_seconds=sample_age,
        services=services,
    )


def _admission(pending: int = 0, max_pending: int = 8, rejected: int = 0) -> rs.AdmissionInfo:
    return rs.AdmissionInfo(
        pending=pending,
        max_pending=max_pending,
        rejected_since_start=rejected,
        process_started_at="2026-01-01T00:00:00+00:00",
    )


def _view(
    name: str = "transcription",
    *,
    up: bool = True,
    device: str | None = "cuda",
    admission: rs.AdmissionInfo | None = None,
) -> rs.ServiceResourceView:
    return rs.ServiceResourceView(
        name=name,
        up=up,
        device=device,
        telemetry_available=admission is not None,
        gpu_uuid=UUID_A if admission is not None else None,
        admission=admission,
        cpu=None,
    )


def _snapshot(
    gpus: tuple[rs.AggregatedGpu, ...] = (),
    services: tuple[rs.ServiceResourceView, ...] = (),
    age: float = 0.0,
) -> rs.ResourceSnapshot:
    return rs.ResourceSnapshot(gpus=gpus, services=services, collected_age_seconds=age)


class TestDeviceState:
    @pytest.mark.parametrize(
        "util,expected",
        [(None, "unknown"), (0, "idle"), (15, "idle"), (16, "working"),
         (89, "working"), (90, "busy"), (100, "busy")],
    )
    def test_boundaries(self, util: int | None, expected: str) -> None:
        assert rs.device_state(util) == expected


class TestStrip:
    def test_gpu_activity_and_no_warnings_is_calm_not_unavailable(self) -> None:
        strip = rs.build_resource_strip(_snapshot(gpus=(_gpu(util=40),)))
        assert strip.telemetry_present is True
        assert strip.warnings == ()
        assert [g.state for g in strip.gpus] == ["working"]

    def test_no_telemetry_at_all_is_unavailable(self) -> None:
        # No GPU and every service lacks admission -> "status unavailable", the
        # honest state that must never be rendered as all-clear.
        strip = rs.build_resource_strip(_snapshot(services=(_view(admission=None),)))
        assert strip.telemetry_present is False
        assert strip.gpus == ()
        assert strip.warnings == ()

    def test_thermal_throttle_warns_with_remedy(self) -> None:
        gpu = _gpu(throttle_active=True, throttle_reasons=("thermal_sw",))
        strip = rs.build_resource_strip(_snapshot(gpus=(gpu,)))
        assert [w.kind for w in strip.warnings] == ["thermal"]
        w = strip.warnings[0]
        assert "cool" in w.message.lower()
        assert w.remedy  # a non-technical one-step remedy is always present

    def test_non_thermal_throttle_does_not_warn(self) -> None:
        # Power/clock/idle throttling is normal governor behaviour, not a problem.
        gpu = _gpu(throttle_active=True, throttle_reasons=("power", "clock", "idle"))
        assert rs.build_resource_strip(_snapshot(gpus=(gpu,))).warnings == ()

    def test_not_throttling_does_not_warn_even_when_hot(self) -> None:
        gpu = _gpu(throttle_active=False, temp=95, throttle_reasons=("thermal_sw",))
        assert rs.build_resource_strip(_snapshot(gpus=(gpu,))).warnings == ()

    def test_high_vram_and_full_util_are_not_warnings(self) -> None:
        # Resident model footprint + 100% util during a run are healthy, never amber.
        gpu = _gpu(util=100, used=11_900_000_000, total=12_000_000_000)
        strip = rs.build_resource_strip(_snapshot(gpus=(gpu,)))
        assert strip.warnings == ()
        assert strip.gpus[0].state == "busy"

    def test_queue_full_warns_with_remedy(self) -> None:
        svc = _view("transcription", admission=_admission(pending=8, max_pending=8))
        strip = rs.build_resource_strip(_snapshot(services=(svc,)))
        assert [w.kind for w in strip.warnings] == ["queue_full"]
        assert "transcription" in strip.warnings[0].message
        assert strip.warnings[0].remedy

    def test_queue_below_max_does_not_warn(self) -> None:
        svc = _view("transcription", admission=_admission(pending=7, max_pending=8))
        assert rs.build_resource_strip(_snapshot(services=(svc,))).warnings == ()

    def test_cumulative_rejects_are_not_a_live_warning(self) -> None:
        # rejected_since_start is monotonic-since-boot: the wrong tense for a live
        # strip. A past rejection with a currently non-full queue raises nothing.
        svc = _view("transcription", admission=_admission(pending=1, max_pending=8, rejected=42))
        assert rs.build_resource_strip(_snapshot(services=(svc,))).warnings == ()

    def test_zero_capacity_queue_warns(self) -> None:
        # max_pending 0 means the service accepts nothing and rejects every
        # request; a 0/0 queue is turning work away and must not read as calm.
        svc = _view("transcription", admission=_admission(pending=0, max_pending=0))
        strip = rs.build_resource_strip(_snapshot(services=(svc,)))
        assert [w.kind for w in strip.warnings] == ["queue_full"]

    def test_partial_telemetry_loss_is_named_not_all_clear(self) -> None:
        # One service reports, another reported nothing (down or older build).
        # The reporting service keeps telemetry_present true, but the missing one
        # is named as unavailable rather than swallowed into "no warnings".
        present = _view("transcription", admission=_admission(pending=1))
        missing = _view("speaker embedding", admission=None)
        strip = rs.build_resource_strip(_snapshot(services=(present, missing)))
        assert strip.telemetry_present is True
        assert strip.warnings == ()
        assert strip.unavailable_services == ("speaker embedding",)


class TestJsonRenderer:
    def test_shape_and_values(self) -> None:
        snap = _snapshot(
            gpus=(_gpu(util=77),),
            services=(_view("transcription", admission=_admission(pending=2, rejected=1)),),
            age=3.5,
        )
        data = rs.resource_snapshot_to_json(snap)
        assert data["collected_age_seconds"] == 3.5
        assert data["gpus"][0]["utilization_percent"] == 77
        assert data["gpus"][0]["services"] == ["transcription"]
        svc = data["services"][0]
        assert svc["admission"]["pending"] == 2
        assert svc["admission"]["rejected_since_start"] == 1
        assert svc["cpu"] is None

    def test_service_without_admission_serialises_null(self) -> None:
        data = rs.resource_snapshot_to_json(_snapshot(services=(_view(admission=None),)))
        assert data["services"][0]["admission"] is None


class TestPrometheusRenderer:
    def test_gpu_and_admission_gauges_present(self) -> None:
        snap = _snapshot(
            gpus=(_gpu(util=77, temp=61),),
            services=(_view("transcription", admission=_admission(pending=2, max_pending=8)),),
        )
        text = rs.render_resource_prometheus(snap)
        assert f'voxint_gpu_utilization_percent{{gpu="{UUID_A}"}} 77' in text
        assert f'voxint_gpu_temperature_celsius{{gpu="{UUID_A}"}} 61' in text
        assert 'voxint_service_admission_pending{service="transcription"} 2' in text
        assert 'voxint_service_admission_max_pending{service="transcription"} 8' in text
        assert "# TYPE voxint_gpu_utilization_percent gauge" in text
        assert text.endswith("\n")

    def test_rejected_is_a_counter_not_a_gauge(self) -> None:
        # rejected_since_start is monotonic within a process, so it is exposed as
        # a counter (a restart is a normal reset) with the `_total` name.
        snap = _snapshot(
            services=(_view("transcription", admission=_admission(rejected=5)),),
        )
        text = rs.render_resource_prometheus(snap)
        assert "# TYPE voxint_service_admission_rejected_total counter" in text
        assert 'voxint_service_admission_rejected_total{service="transcription"} 5' in text
        assert "voxint_service_admission_rejected_since_start" not in text

    def test_throttle_active_bool_becomes_one_zero(self) -> None:
        on = rs.render_resource_prometheus(_snapshot(gpus=(_gpu(throttle_active=True),)))
        off = rs.render_resource_prometheus(_snapshot(gpus=(_gpu(throttle_active=False),)))
        assert f'voxint_gpu_throttle_active{{gpu="{UUID_A}"}} 1' in on
        assert f'voxint_gpu_throttle_active{{gpu="{UUID_A}"}} 0' in off

    def test_null_values_omit_their_sample(self) -> None:
        text = rs.render_resource_prometheus(_snapshot(gpus=(_gpu(util=None, temp=None),)))
        assert "voxint_gpu_utilization_percent{" not in text  # no sample for null util
        assert f'voxint_gpu_vram_used_bytes{{gpu="{UUID_A}"}} 4000000000' in text

    def test_empty_snapshot_renders_nothing(self) -> None:
        assert rs.render_resource_prometheus(_snapshot()) == ""

    def test_no_total_suffix_on_gauges(self) -> None:
        # The guardrail is the monotonic-counter `_total` *suffix* (which promtool
        # flags on a gauge), not the substring — `vram_total_bytes` is fine.
        text = rs.render_resource_prometheus(_snapshot(gpus=(_gpu(),)))
        assert "_total{" not in text and "_total " not in text


class TestTrioAgreement:
    def test_all_renderers_agree_on_the_same_numbers(self) -> None:
        # One snapshot -> json / prometheus / strip / text must reflect the same
        # utilization and queue figures (the plan's core W3 invariant).
        snap = _snapshot(
            gpus=(_gpu(util=83),),
            services=(_view("transcription", admission=_admission(pending=8, max_pending=8)),),
        )
        data = rs.resource_snapshot_to_json(snap)
        prom = rs.render_resource_prometheus(snap)
        strip = rs.build_resource_strip(snap)
        text = rs.format_resource_status_text(snap)

        assert data["gpus"][0]["utilization_percent"] == 83
        assert f'voxint_gpu_utilization_percent{{gpu="{UUID_A}"}} 83' in prom
        assert "util 83%" in text
        assert strip.gpus[0].state == "working"  # 83 -> working, consistently
        # The full queue is the one warning, and prometheus carries the same 8/8.
        assert [w.kind for w in strip.warnings] == ["queue_full"]
        assert 'voxint_service_admission_pending{service="transcription"} 8' in prom
        assert data["services"][0]["admission"]["pending"] == 8
