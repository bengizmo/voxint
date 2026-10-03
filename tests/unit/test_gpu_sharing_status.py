"""GPU sharing visibility (#748 slice 4): copy, expected stops, status rows.

An expected stop is a dependency GPU sharing stopped on purpose. It must read
as off with its reason everywhere (doctor, Status page, Runs page) and never
as a failure or a nonzero doctor exit.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from sqlalchemy import create_engine

from tests.unit.test_gpu_phase import phase_settings
from voxint.api.pipeline_dashboard_query import degraded_stages
from voxint.api.routers.jobs import _detect_degraded, _pipeline_summary
from voxint.api.routers.settings import _build_components
from voxint.api.service_control import NoopController, ServiceState
from voxint.cli import main
from voxint.diagnostics import (
    GPU_SHARING_CHECK,
    CheckResult,
    apply_gpu_phase,
    check_gpu_phase,
    check_state,
    exit_code,
    gpu_phase_stale_after_seconds,
    run_diagnostics,
)
from voxint.gpu_phase.state import GpuPhase, GpuPhaseSnapshot, OperatorRequest
from voxint.gpu_phase.visibility import (
    LLM_EXPECTED_DOWN_DETAIL,
    NO_ROW_SUMMARY,
    PHASE_SUMMARY,
    build_view,
    llm_expected_down,
    model_service_stop_detail,
    phase_summary,
    snapshot_phase,
    stage_pause_reason,
)

NOW = datetime(2026, 10, 2, 18, 0, tzinfo=UTC)
MODEL_CHECKS = ("transcription", "diarization", "speaker embedding")


def snap(phase: GpuPhase, **fields: Any) -> GpuPhaseSnapshot:
    values: dict[str, Any] = {
        "id": 1,
        "phase": phase,
        "phase_since": NOW,
        "lease_id": None,
        "lease_expires_at": None,
        "last_error": None,
        "failures": 0,
        "retry_after": None,
        "operator_request": None,
        "updated_at": NOW,
    }
    values.update(fields)
    return GpuPhaseSnapshot(**values)


def gpu_check(**kwargs: Any) -> CheckResult | None:
    """check_gpu_phase at NOW with the default 30 s tick; the row is fresh."""
    kwargs.setdefault("now", NOW)
    kwargs.setdefault("tick_seconds", 30)
    return check_gpu_phase(**kwargs)


def down_models() -> list[CheckResult]:
    return [CheckResult(name, False, True, "unreachable (ConnectError)") for name in MODEL_CHECKS]


# ---- phase copy ------------------------------------------------------------


def test_every_phase_has_a_summary() -> None:
    assert set(PHASE_SUMMARY) == set(GpuPhase)
    assert snapshot_phase(snap(GpuPhase.AUDIO)) == GpuPhase.AUDIO
    assert phase_summary(GpuPhase.AUDIO) == "the GPU is doing audio work"


def test_missing_row_closes_both_lanes_in_every_helper() -> None:
    # A missing row closes both lanes while the feature is on (the orchestrator
    # repairs it): services and the language model are both treated as waiting.
    assert snapshot_phase(None) is None
    assert phase_summary(None) == NO_ROW_SUMMARY
    assert model_service_stop_detail(None) == "stopped; GPU sharing has not recorded a phase yet"
    assert stage_pause_reason(None) == "GPU sharing has not started"
    assert llm_expected_down(None) is False
    view = build_view(None, 2)
    assert view.headline == "Waiting for GPU sharing to start."
    assert view.detail == (
        "2 recordings are queued. No GPU phase is recorded yet, so audio and "
        "language-model work wait until the GPU sharing task records one."
    )
    assert view.note == "waiting for GPU sharing to start" and not view.is_error
    assert build_view(None, 0).detail == (
        "No GPU phase is recorded yet, so audio and language-model work wait "
        "until the GPU sharing task records one."
    )


@pytest.mark.parametrize("phase", list(GpuPhase))
def test_services_expected_down_outside_audio_and_draining(phase: GpuPhase) -> None:
    up = phase in {GpuPhase.AUDIO, GpuPhase.DRAINING}
    assert (model_service_stop_detail(phase) is None) is up
    assert (stage_pause_reason(phase) is None) is up


@pytest.mark.parametrize("phase", list(GpuPhase))
def test_llm_expected_down_only_while_voxint_holds_or_moves_the_gpu(phase: GpuPhase) -> None:
    expected = phase not in {GpuPhase.LLM, GpuPhase.DRAINING_POST, GpuPhase.ERROR}
    assert llm_expected_down(phase) is expected


def test_waiting_banner_copy_and_plural() -> None:
    view = build_view(GpuPhase.LLM, 3)
    assert view.headline == "Waiting for the GPU."
    assert view.detail == (
        "3 recordings are queued; the GPU is serving the language model "
        "until the next audio window."
    )
    assert view.note == "waiting for the GPU" and not view.is_error
    assert build_view(GpuPhase.LLM, 1).detail is not None
    assert "1 recording is queued;" in str(build_view(GpuPhase.LLM, 1).detail)


def test_llm_phase_with_nothing_waiting_and_audio_have_no_banner() -> None:
    for view in (build_view(GpuPhase.LLM, 0), build_view(GpuPhase.AUDIO, 0)):
        assert view.headline is None and view.detail is None and view.note is None


@pytest.mark.parametrize(
    "phase", [GpuPhase.DRAINING_POST, GpuPhase.ACQUIRING, GpuPhase.STARTING_SERVICES]
)
def test_switching_to_audio_copy(phase: GpuPhase) -> None:
    view = build_view(phase, 2)
    assert view.headline == "Switching the GPU to audio work."
    assert view.detail == "2 recordings are queued and will start once the model services are up."
    assert build_view(phase, 0).detail == "Audio work starts once the model services are up."


@pytest.mark.parametrize(
    "phase", [GpuPhase.DRAINING, GpuPhase.STOPPING_SERVICES, GpuPhase.RELEASING]
)
def test_switching_back_copy(phase: GpuPhase) -> None:
    assert build_view(phase, 0).headline == "Switching the GPU back."
    assert build_view(phase, 0).detail == (
        "Language-model work resumes once the GPU is handed back."
    )
    assert build_view(phase, 1).detail == (
        "Language-model work resumes once the GPU is handed back. "
        "1 recording is queued for the next audio window."
    )


def test_error_copy() -> None:
    view = build_view(GpuPhase.ERROR, 4)
    assert view.is_error
    assert view.headline == "Could not return the GPU to the language model."
    assert view.detail == "LLM work is paused."
    assert view.note == "GPU sharing needs attention"


# ---- diagnostics -------------------------------------------------------------


def test_check_gpu_phase_off_is_absent() -> None:
    assert gpu_check(enabled=False, snapshot=snap(GpuPhase.ERROR)) is None


def test_check_gpu_phase_llm_is_ok_advisory() -> None:
    result = gpu_check(enabled=True, snapshot=snap(GpuPhase.LLM))
    assert result == CheckResult(
        GPU_SHARING_CHECK,
        True,
        False,
        "the GPU is serving the language model since 2026-10-02 18:00 UTC",
    )


def test_check_gpu_phase_missing_row_needs_attention() -> None:
    result = gpu_check(enabled=True, snapshot=None)
    assert result == CheckResult(
        GPU_SHARING_CHECK,
        False,
        False,
        f"{NO_ROW_SUMMARY}; check that the gpu-phase worker is running",
    )
    assert exit_code([result]) == 0


def test_apply_gpu_phase_missing_row() -> None:
    adjusted = apply_gpu_phase(down_models(), None)
    assert all(r.expected_stop for r in adjusted) and exit_code(adjusted) == 0
    endpoint = [CheckResult("llm endpoint", False, False, "unreachable (ConnectError)")]
    assert apply_gpu_phase(endpoint, None) == endpoint


def test_check_gpu_phase_reports_request_and_backoff() -> None:
    result = gpu_check(
        enabled=True,
        snapshot=snap(
            GpuPhase.LLM,
            operator_request=OperatorRequest.AUDIO,
            failures=2,
            retry_after=NOW + timedelta(minutes=1),
            last_error="broker refused (409)",
        ),
    )
    assert result is not None and result.ok
    assert result.detail == (
        "the GPU is serving the language model since 2026-10-02 18:00 UTC; "
        "operator asked for audio work now; "
        "last attempt failed, next try after 2026-10-02 18:01 UTC (broker refused (409))"
    )


def test_check_gpu_phase_error_is_an_advisory_failure_with_last_error() -> None:
    result = gpu_check(
        enabled=True,
        snapshot=snap(GpuPhase.ERROR, last_error="could not return the GPU: release timed out"),
    )
    assert result is not None and not result.ok and not result.hard
    assert "could not return the GPU to the language model" in result.detail
    assert "release timed out" in result.detail
    assert "voxint gpu-phase release" in result.detail
    assert check_state(result) == "unverified"
    assert exit_code([result]) == 0


def test_stale_after_is_three_ticks_with_a_two_minute_floor() -> None:
    assert gpu_phase_stale_after_seconds(5) == 120
    assert gpu_phase_stale_after_seconds(30) == 120
    assert gpu_phase_stale_after_seconds(60) == 180


@pytest.mark.parametrize("phase", [GpuPhase.LLM, GpuPhase.AUDIO, GpuPhase.ERROR])
def test_check_gpu_phase_reports_a_task_that_stopped_writing(phase: GpuPhase) -> None:
    written = NOW - timedelta(seconds=121)
    result = gpu_check(snapshot=snap(phase, updated_at=written), enabled=True)
    assert result == CheckResult(
        GPU_SHARING_CHECK,
        False,
        False,
        "the GPU sharing task has not run since 2026-10-02 17:57 UTC; check that the "
        "gpu-phase service from compose.gpu-phase.yaml is running",
    )
    assert exit_code([result]) == 0


def test_check_gpu_phase_at_the_stale_boundary_is_still_fresh() -> None:
    at_limit = snap(GpuPhase.LLM, updated_at=NOW - timedelta(seconds=120))
    result = gpu_check(snapshot=at_limit, enabled=True)
    assert result is not None and result.ok
    slow_tick = snap(GpuPhase.LLM, updated_at=NOW - timedelta(seconds=150))
    result = gpu_check(snapshot=slow_tick, enabled=True, tick_seconds=60)
    assert result is not None and result.ok


def test_check_gpu_phase_read_error() -> None:
    result = gpu_check(enabled=True, snapshot=None, read_error="ProgrammingError")
    assert result == CheckResult(
        GPU_SHARING_CHECK, False, False, "could not read the phase (ProgrammingError)"
    )


@pytest.mark.parametrize("phase", list(GpuPhase))
def test_apply_gpu_phase_marks_down_models_as_expected_stops(phase: GpuPhase) -> None:
    adjusted = apply_gpu_phase(down_models(), phase)
    if phase in {GpuPhase.AUDIO, GpuPhase.DRAINING}:
        assert adjusted == down_models()
        assert exit_code(adjusted) == 1
    else:
        assert all(r.expected_stop and not r.ok and not r.hard for r in adjusted)
        assert {r.detail for r in adjusted} == {model_service_stop_detail(phase)}
        assert exit_code(adjusted) == 0


def test_apply_gpu_phase_reports_running_services_as_they_are() -> None:
    ready = [CheckResult("transcription", True, True, "ready (cuda)")]
    assert apply_gpu_phase(ready, GpuPhase.LLM) == ready


@pytest.mark.parametrize("phase", list(GpuPhase))
def test_apply_gpu_phase_llm_endpoint(phase: GpuPhase) -> None:
    down = [
        CheckResult("llm endpoint", False, False, "unreachable (ConnectError)"),
        CheckResult("llm bundled", False, False, "unreachable (ConnectError)"),
    ]
    endpoint, bundled = apply_gpu_phase(down, phase)
    assert bundled == down[1]  # a Voxint container GPU sharing never stops
    if llm_expected_down(phase):
        assert endpoint == CheckResult(
            "llm endpoint", False, False, LLM_EXPECTED_DOWN_DETAIL, expected_stop=True
        )
    else:
        assert endpoint == down[0]


def test_run_diagnostics_off_adds_no_gpu_row() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    results = run_diagnostics(
        phase_settings(gpu_phase_enabled=False, llm_enabled=False),
        create_engine("sqlite://"),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        redis_client=type("P", (), {"ping": lambda self: True})(),
    )
    assert GPU_SHARING_CHECK not in [r.name for r in results]
    assert not any(r.expected_stop for r in results)


def test_run_diagnostics_unreadable_phase_reports_and_does_not_relabel() -> None:
    # sqlite has no gpu_phase table: the read fails, the GPU row says so, and
    # down services keep their honest hard failure (no guess at the phase).
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    results = run_diagnostics(
        phase_settings(llm_enabled=False),
        create_engine("sqlite://"),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        redis_client=type("P", (), {"ping": lambda self: True})(),
    )
    gpu = next(r for r in results if r.name == GPU_SHARING_CHECK)
    assert not gpu.ok and gpu.detail.startswith("could not read the phase (")
    assert not any(r.expected_stop for r in results)
    assert exit_code(results) == 1


# ---- doctor CLI --------------------------------------------------------------


def test_doctor_prints_expected_stops_as_off_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import voxint.db.session as db_session
    import voxint.diagnostics as diagnostics

    class _Engine:
        def dispose(self) -> None:
            pass

    monkeypatch.setattr(db_session, "build_engine", lambda *a, **k: _Engine())
    canned = [
        CheckResult("postgres", True, True, "connected"),
        *apply_gpu_phase(down_models(), GpuPhase.LLM),
        CheckResult(GPU_SHARING_CHECK, True, False, "the GPU is serving the language model"),
    ]
    monkeypatch.setattr(diagnostics, "run_diagnostics", lambda *a, **k: canned)

    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "[off ] transcription: stopped while the GPU serves the language model" in out
    assert "[ok  ] gpu sharing: the GPU is serving the language model" in out
    assert "all hard dependencies OK" in out


# ---- Status page rows ----------------------------------------------------------


def _check(result: CheckResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "state": check_state(result),
        "detail": result.detail,
        "remediation": "",
        "expected_stop": result.expected_stop,
    }


def _row(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    return next(r for r in rows if r["label"] == label)


class _DockerController(NoopController):
    """A controllable controller that would offer Start on a stopped service."""

    @property
    def controllable(self) -> bool:
        return True

    def inspect(self, service_key: str) -> ServiceState:
        return ServiceState.STOPPED


def test_status_rows_off_have_no_gpu_sharing_row() -> None:
    rows = _build_components(
        [_check(CheckResult("transcription", True, True, "ready"))], NoopController("test")
    )
    assert all(r["label"] != "GPU sharing" for r in rows)


def test_status_rows_expected_stop_is_off_without_controls() -> None:
    checks = [_check(r) for r in apply_gpu_phase(down_models(), GpuPhase.LLM)]
    checks.append(
        _check(CheckResult(GPU_SHARING_CHECK, True, False, "the GPU is serving the language model"))
    )
    rows = _build_components(checks, _DockerController("test"))
    transcriber = _row(rows, "Transcriber")
    assert transcriber["dot"] == "off"
    assert transcriber["state_text"] == "stopped while the GPU serves the language model"
    assert transcriber["controllable"] is False and transcriber["terminal_hint"] is None
    assert transcriber["state"] is None
    sharing = _row(rows, "GPU sharing")
    assert sharing["dot"] == "ok"
    assert sharing["state_text"] == "the GPU is serving the language model"
    assert all(r["dot"] != "warn" for r in rows)  # the banner stays green


def test_status_rows_gpu_sharing_error_warns() -> None:
    result = gpu_check(enabled=True, snapshot=snap(GpuPhase.ERROR, last_error="boom"))
    assert result is not None
    rows = _build_components([_check(result)], NoopController("test"))
    sharing = _row(rows, "GPU sharing")
    assert sharing["dot"] == "warn" and "boom" in sharing["state_text"]


def test_status_rows_expected_llm_stop_is_off() -> None:
    (endpoint,) = apply_gpu_phase(
        [CheckResult("llm endpoint", False, False, "unreachable (ConnectError)")], GpuPhase.AUDIO
    )
    rows = _build_components([_check(endpoint)], NoopController("test"))
    byo = _row(rows, "Your own AI endpoint")
    assert byo["dot"] == "off" and byo["state_text"] == LLM_EXPECTED_DOWN_DETAIL


# ---- Runs page helpers ---------------------------------------------------------


def test_degraded_stages_uses_the_gpu_sharing_reason_once() -> None:
    services = [("transcription", False), ("diarization", False), ("speaker embedding", False)]
    reason = stage_pause_reason(GpuPhase.LLM)
    assert degraded_stages(services, llm_enabled=True, expected_stop_reason=reason) == {
        "transcribe": "the GPU is serving the language model",
        "diarize_embed": "the GPU is serving the language model",
    }
    # Without GPU sharing the per-service reasons are unchanged.
    assert degraded_stages(services, llm_enabled=True)["transcribe"] == "transcriber is down"


def test_detect_degraded_suppresses_model_banners_for_expected_stops() -> None:
    services = [("transcription", False), ("diarization", False)]
    assert _detect_degraded(services, llm_enabled=True, model_services_expected_down=True) == []
    names = [
        d.name
        for d in _detect_degraded(services, llm_enabled=False, model_services_expected_down=True)
    ]
    assert names == ["enrichment"]
    assert len(_detect_degraded(services, llm_enabled=True)) == 2


def test_pipeline_summary_carries_the_gpu_note() -> None:
    summary = _pipeline_summary({"queued": 2}, gpu_sharing_note="waiting for the GPU")
    assert summary == "waiting for the GPU · 2 queued"
    assert _pipeline_summary({"queued": 2}) == "2 queued"
