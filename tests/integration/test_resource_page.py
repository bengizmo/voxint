"""The hardware resource surfaces (hardware-aware W3) end to end.

The pure curation/rendering is covered in ``tests/unit/test_resource_status.py``;
these pin the wiring the unit tests cannot see: that the hardware strip, the
``/settings/status`` page it now lives on (full + htmx fragment), and ``/metrics``
all render one cached ``ResourceSnapshot``, that a probe failure degrades honestly
instead of 500-ing, and that a mixed-version deploy (a GPU-reporting service
beside an old service without a ``resources`` block) renders without crashing.

The hardware view folded from ``/resources`` into ``/settings/status`` at Console
2.0 P6b (#161); ``/resources`` now issues a permanent 303 there, and the status
page answers an ``HX-Request`` poll with just the hardware fragment (the plain
303 is pinned by the REDIRECT_MAP contract in
``tests/contracts/test_console2_characterization.py``).

The route probes the model services live (behind a cache); with none running in
the test env that would just degrade to empty, so each test instead patches
``voxint.api.resource_status.collect_resource_status`` to return a crafted
snapshot — the route still runs ``build_resource_strip`` /
``render_resource_prometheus`` for real over it.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.resource_status import (
    AdmissionInfo,
    AggregatedGpu,
    ResourceSnapshot,
    ServiceResourceView,
)
from voxint.config import Settings

CREDS = ("reviewer", "s3cret")
UUID_A = "GPU-aaaaaaaa-1111-2222-3333-444444444444"


@pytest.fixture()
def client(session_factory: sessionmaker[Session], tmp_path: Path) -> TestClient:
    settings = Settings(voxint_user=CREDS[0], voxint_password=CREDS[1], media_root=tmp_path)
    test_client = TestClient(create_app(settings=settings, session_factory=session_factory))
    test_client.auth = CREDS
    seed_onboarded(session_factory)
    return test_client


def _gpu(
    *,
    util: int | None = 42,
    throttle_active: bool | None = False,
    throttle_reasons: tuple[str, ...] = (),
    services: tuple[str, ...] = ("transcription", "diarization"),
) -> AggregatedGpu:
    return AggregatedGpu(
        gpu_uuid=UUID_A,
        utilization_percent=util,
        vram_used_bytes=4_000_000_000,
        vram_total_bytes=12_000_000_000,
        temperature_celsius=61,
        throttle_active=throttle_active,
        throttle_reasons=throttle_reasons,
        max_temperature_celsius=70,
        throttle_events_since_start=0,
        sample_age_seconds=1.0,
        services=services,
    )


def _admission(pending: int = 1, max_pending: int = 8, rejected: int = 0) -> AdmissionInfo:
    return AdmissionInfo(
        pending=pending,
        max_pending=max_pending,
        rejected_since_start=rejected,
        process_started_at="2026-01-01T00:00:00+00:00",
    )


def _view(name: str, admission: AdmissionInfo | None) -> ServiceResourceView:
    return ServiceResourceView(
        name=name,
        up=admission is not None,
        device="cuda" if admission is not None else None,
        telemetry_available=admission is not None,
        gpu_uuid=UUID_A if admission is not None else None,
        admission=admission,
        cpu=None,
    )


def _patch_snapshot(monkeypatch: pytest.MonkeyPatch, snapshot: ResourceSnapshot) -> None:
    # Patch the canonical source (resource_status). The guarded reader every page
    # shares — collect_resource_status_or_empty (#160) — resolves this module
    # global at call time, so one patch covers the resource page, /metrics, and
    # the Jobs strip alike.
    monkeypatch.setattr(
        "voxint.api.resource_status.collect_resource_status",
        lambda settings, **kw: snapshot,
    )


def test_resources_page_renders_gpu_gauges_and_components(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    snap = ResourceSnapshot(
        gpus=(_gpu(util=42),),
        services=(_view("transcription", _admission(pending=2, max_pending=8)),),
        collected_age_seconds=1.0,
    )
    _patch_snapshot(monkeypatch, snap)
    resp = client.get("/settings/status")
    assert resp.status_code == 200
    body = resp.text
    assert "PARTS OF VOXINT" in body
    assert "THIS COMPUTER RIGHT NOW" in body
    assert "42%" in body
    assert "Graphics card" in body
    assert "Graphics memory" in body
    assert "GB" in body


def test_resources_htmx_fragment_omits_chrome(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_snapshot(
        monkeypatch,
        ResourceSnapshot(gpus=(_gpu(),), services=(), collected_age_seconds=0.0),
    )
    resp = client.get("/settings/status", headers={"HX-Request": "true"})
    assert resp.status_code == 200
    body = resp.text
    assert "<nav" not in body and "<h1" not in body
    assert "Graphics card" in body


def test_stale_resources_poll_survives_the_redirect(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A Resources tab left open across the P6b deploy keeps polling
    # hx-get="/resources" (#161). The 303 to /settings/status carries the
    # HX-Request header through, and the status page must answer that poll with
    # just the hardware fragment, so the already-open page keeps refreshing in
    # place instead of swapping a whole page shell into its poll container.
    _patch_snapshot(
        monkeypatch,
        ResourceSnapshot(gpus=(_gpu(),), services=(), collected_age_seconds=0.0),
    )
    resp = client.get("/resources", headers={"HX-Request": "true"})
    assert resp.status_code == 200  # followed the 303 to the status fragment
    assert str(resp.url).endswith("/settings/status")
    body = resp.text
    assert "<nav" not in body and "<h1" not in body
    assert "Graphics card" in body


def test_resource_strip_shows_thermal_warning_with_remedy(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    snap = ResourceSnapshot(
        gpus=(_gpu(throttle_active=True, throttle_reasons=("thermal_sw",)),),
        services=(),
        collected_age_seconds=0.0,
    )
    _patch_snapshot(monkeypatch, snap)
    body = client.get("/settings/status").text
    assert "slowing itself down to stay cool" in body
    assert "fans and air vents" in body  # the one-step remedy


def test_resource_strip_unavailable_when_empty(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_snapshot(
        monkeypatch,
        ResourceSnapshot(gpus=(), services=(), collected_age_seconds=0.0),
    )
    body = client.get("/settings/status").text
    assert "Hardware status unavailable" in body


def test_probe_failure_degrades_to_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A probe that raises must never break the page: collect_resource_status_or_empty
    # swallows it into an empty snapshot, so the strip shows "unavailable", not a 500.
    def _boom(settings: object, **kw: object) -> ResourceSnapshot:
        raise RuntimeError("nvml exploded")

    monkeypatch.setattr("voxint.api.resource_status.collect_resource_status", _boom)
    resp = client.get("/settings/status")
    assert resp.status_code == 200
    assert "Hardware status unavailable" in resp.text


def test_metrics_carries_gpu_and_admission_gauges(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    snap = ResourceSnapshot(
        gpus=(_gpu(util=77),),
        services=(_view("transcription", _admission(pending=3, max_pending=8)),),
        collected_age_seconds=0.0,
    )
    _patch_snapshot(monkeypatch, snap)
    body = client.get("/metrics").text
    assert f'voxint_gpu_utilization_percent{{gpu="{UUID_A}"}} 77' in body
    assert 'voxint_service_admission_pending{service="transcription"} 3' in body
    # The DB stats are still present: the resource gauges are appended, not swapped.
    assert "voxint_runs{" in body


def test_metrics_and_resources_agree_on_one_snapshot(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    snap = ResourceSnapshot(
        gpus=(_gpu(util=63),), services=(), collected_age_seconds=0.0
    )
    _patch_snapshot(monkeypatch, snap)
    metrics = client.get("/metrics").text
    page = client.get("/settings/status").text
    assert f'voxint_gpu_utilization_percent{{gpu="{UUID_A}"}} 63' in metrics
    assert "63%" in page  # same reading, both surfaces


def test_mixed_version_old_service_without_resources(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # New app + one old service (no `resources` block -> admission None). The GPU
    # (reported by the newer services) still renders; the page does not crash.
    snap = ResourceSnapshot(
        gpus=(_gpu(services=("transcription", "diarization")),),
        services=(
            _view("transcription", _admission(pending=1)),
            _view("speaker embedding", None),  # old service, no telemetry
        ),
        collected_age_seconds=0.0,
    )
    _patch_snapshot(monkeypatch, snap)
    resp = client.get("/settings/status")
    assert resp.status_code == 200
    body = resp.text
    assert "GPU aaaaaaaa" in body  # aggregated card still shown
    assert "transcription" in body  # the telemetry-bearing service's queue row
    # The old/down service is NOT hidden: it appears in the queue table with its
    # telemetry marked unavailable rather than silently dropped.
    assert "speaker embedding" in body
    assert "unavailable" in body


def test_resource_strip_names_partial_telemetry_loss(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One service reports, another reported nothing: the strip must not claim
    # "no hardware warnings" without qualifying that a service is unavailable.
    snap = ResourceSnapshot(
        gpus=(_gpu(util=30),),
        services=(
            _view("transcription", _admission(pending=1)),
            _view("speaker embedding", None),
        ),
        collected_age_seconds=0.0,
    )
    _patch_snapshot(monkeypatch, snap)
    body = client.get("/settings/status").text
    assert "Telemetry unavailable for: speaker embedding" in body


def test_resources_page_renders_unknown_readings(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A GPU that reports a UUID but null util/vram/temp (a partial-telemetry
    # driver state) must render "unknown" cells, not error at template time.
    gpu = AggregatedGpu(
        gpu_uuid=UUID_A,
        utilization_percent=None,
        vram_used_bytes=None,
        vram_total_bytes=None,
        temperature_celsius=None,
        throttle_active=None,
        throttle_reasons=(),
        max_temperature_celsius=None,
        throttle_events_since_start=None,
        sample_age_seconds=None,
        services=("transcription",),
    )
    _patch_snapshot(
        monkeypatch,
        ResourceSnapshot(gpus=(gpu,), services=(), collected_age_seconds=0.0),
    )
    resp = client.get("/settings/status")
    assert resp.status_code == 200
    assert "unknown" in resp.text  # null readings degrade to "unknown", no crash


def test_resources_page_cpu_only_says_no_gpu(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A CPU-only install reports admission but no GPU: the page says so plainly
    # rather than showing an empty GPU section.
    _patch_snapshot(
        monkeypatch,
        ResourceSnapshot(
            gpus=(),
            services=(_view("transcription", _admission(pending=0)),),
            collected_age_seconds=0.0,
        ),
    )
    body = client.get("/settings/status").text
    assert "No GPU telemetry reported" in body
    assert "transcription" in body  # admission queue still shown
