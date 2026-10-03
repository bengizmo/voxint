"""GPU sharing visibility against real Postgres (#748 slice 4).

The CLI (`voxint gpu-phase`), doctor's checks, the Status page and the Runs
page all read the stored phase. Model services are pointed at a closed port so
they are down, as they are while GPU sharing has them stopped.
"""

import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from tests.integration.test_gpu_phase import NOW, seed_run
from tests.integration.test_jobs_pages import _prime_resource_cache
from tests.unit.test_gpu_phase import phase_settings
from voxint.api.app import create_app
from voxint.api.resource_status import _reset_cache_for_tests
from voxint.cli import main
from voxint.config import Settings
from voxint.db.models import Stage
from voxint.diagnostics import GPU_SHARING_CHECK, CheckResult, exit_code, run_diagnostics
from voxint.gpu_phase.state import GpuPhase, OperatorRequest, read_phase, set_phase

CREDS = ("reviewer", "s3cret")
CLOSED = "http://127.0.0.1:9"  # discard port: connection refused


def _set(session_factory: sessionmaker[Session], phase: GpuPhase, **fields: object) -> None:
    with session_factory() as session:
        # A fresh write, as the phase task makes each tick (doctor flags a stale row).
        set_phase(session, phase, now=datetime.now(UTC), **fields)  # type: ignore[arg-type]
        session.commit()


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "voxint_user": CREDS[0],
        "voxint_password": CREDS[1],
        "csrf_secret": "gpu-sharing-test-csrf-key",
        "asr_url": CLOSED,
        "diarizer_url": CLOSED,
        "embedder_url": CLOSED,
        "resource_status_ttl_seconds": 3600.0,
    }
    values.update(overrides)
    return phase_settings(**values)


def _runs_settings(**overrides: object) -> Settings:
    """Default service URLs: the primed resource cache is keyed by their ports."""
    defaults = Settings.model_fields
    return _settings(
        asr_url=defaults["asr_url"].default,
        diarizer_url=defaults["diarizer_url"].default,
        embedder_url=defaults["embedder_url"].default,
        **overrides,
    )


def _client(
    session_factory: sessionmaker[Session], settings: Settings, *, llm_enabled: bool = False
) -> TestClient:
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory, llm_enabled=llm_enabled)
    return client


@pytest.fixture()
def clean_resource_cache() -> Iterator[None]:
    _reset_cache_for_tests()
    try:
        yield
    finally:
        _reset_cache_for_tests()


# ---- CLI ---------------------------------------------------------------------


@pytest.fixture()
def cli_env(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> Iterator[list[Settings]]:
    holder = [phase_settings()]
    monkeypatch.setattr("voxint.config.get_settings", lambda: holder[0])
    monkeypatch.setattr("voxint.db.session.build_engine", lambda: None)
    monkeypatch.setattr("voxint.db.session.build_session_factory", lambda _: session_factory)
    yield holder


def test_status_off_says_so_and_exits_zero(
    cli_env: list[Settings], capsys: pytest.CaptureFixture[str]
) -> None:
    cli_env[0] = phase_settings(gpu_phase_enabled=False)
    assert main(["gpu-phase", "status"]) == 0
    assert capsys.readouterr().out == "GPU sharing is off (GPU_PHASE_ENABLED is not set)\n"


@pytest.mark.parametrize("command", ["audio-now", "release"])
def test_requests_refuse_when_off_and_change_nothing(
    cli_env: list[Settings],
    session_factory: sessionmaker[Session],
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    _set(session_factory, GpuPhase.LLM, operator_request=None)
    cli_env[0] = phase_settings(gpu_phase_enabled=False)
    assert main(["gpu-phase", command]) == 2
    assert capsys.readouterr().out == (
        "error: GPU sharing is off (GPU_PHASE_ENABLED is not set); there is no phase to change\n"
    )
    with session_factory() as session:
        snapshot = read_phase(session)
        assert snapshot is not None and snapshot.operator_request is None


def test_status_prints_phase_lease_error_request_and_counts(
    cli_env: list[Settings],
    session_factory: sessionmaker[Session],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set(
        session_factory,
        GpuPhase.LLM,
        last_error="broker refused (409)",
        failures=2,
        retry_after=NOW,
        operator_request=OperatorRequest.AUDIO,
        lease_expires_at=None,
    )
    with session_factory() as session:
        seed_run(session, None)
        seed_run(session, Stage.TRANSCRIBE)
        seed_run(session, Stage.PREPARE, status="running")
        seed_run(session, Stage.FINALIZE, status="running")
        seed_run(session, Stage.FINALIZE)
        session.commit()
    assert main(["gpu-phase", "status"]) == 0
    out = capsys.readouterr().out
    assert "phase:            llm (the GPU is serving the language model)\n" in out
    # phase_since keeps the row's own value when the phase does not change.
    assert re.search(r"^since:            \d{4}-\d\d-\d\d \d\d:\d\d:\d\d UTC$", out, re.M)
    assert f"retry after:      {NOW.strftime('%Y-%m-%d %H:%M:%S')} UTC\n" in out
    assert "lease expires:    -\n" in out
    assert "last error:       broker refused (409)\n" in out
    assert "failures:         2\n" in out
    assert "pending request:  audio\n" in out
    assert "waiting for GPU:  2 queued runs\n" in out
    assert "in flight:        1 on the GPU, 1 after it\n" in out


def test_status_without_a_row_says_both_lanes_wait(
    cli_env: list[Settings],
    session_factory: sessionmaker[Session],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with session_factory() as session:
        session.execute(text("DELETE FROM gpu_phase"))
        session.commit()
    assert main(["gpu-phase", "status"]) == 0
    out = capsys.readouterr().out
    assert "phase:            none (no phase recorded yet; both lanes wait" in out
    assert "since:" not in out


def test_runs_without_a_row_say_gpu_sharing_has_not_started(
    session_factory: sessionmaker[Session], clean_resource_cache: None
) -> None:
    client = _client(session_factory, _settings())
    with session_factory() as session:
        session.execute(text("DELETE FROM gpu_phase"))
        session.commit()
    body = client.get("/runs").text
    assert "<strong>Waiting for GPU sharing to start.</strong>" in body
    assert 'data-gpu-sharing="none"' in body


@pytest.mark.parametrize(
    ("command", "phase", "request_value", "message"),
    [
        ("audio-now", GpuPhase.LLM, OperatorRequest.AUDIO, "the GPU switches over the next"),
        ("audio-now", GpuPhase.AUDIO, OperatorRequest.AUDIO, "already doing audio work"),
        ("release", GpuPhase.AUDIO, OperatorRequest.RELEASE, "once audio work in progress"),
        ("release", GpuPhase.ERROR, OperatorRequest.RELEASE, "retries handing the GPU back"),
        ("release", GpuPhase.LLM, OperatorRequest.RELEASE, "already serves the language model"),
    ],
)
def test_requests_are_recorded(
    cli_env: list[Settings],
    session_factory: sessionmaker[Session],
    capsys: pytest.CaptureFixture[str],
    command: str,
    phase: GpuPhase,
    request_value: OperatorRequest,
    message: str,
) -> None:
    _set(session_factory, phase, operator_request=None)
    assert main(["gpu-phase", command]) == 0
    assert message in capsys.readouterr().out
    with session_factory() as session:
        snapshot = read_phase(session)
        assert snapshot is not None
        assert snapshot.operator_request == request_value
        assert snapshot.phase == phase  # the request never moves the phase itself


# ---- diagnostics -------------------------------------------------------------


class _Ping:
    def ping(self) -> bool:
        return True


def _diagnose(engine: Engine, settings: Settings) -> list[CheckResult]:
    with httpx.Client(timeout=2.0) as client:
        return run_diagnostics(
            settings, engine, http_client=client, redis_client=_Ping(), include_hf_token=False
        )


def test_doctor_llm_phase_models_down_is_expected_and_exit_zero(
    engine: Engine, session_factory: sessionmaker[Session]
) -> None:
    _set(session_factory, GpuPhase.LLM)
    results = _diagnose(engine, _settings())
    by_name = {r.name: r for r in results}
    for name in ("transcription", "diarization", "speaker embedding"):
        assert by_name[name].expected_stop
        assert by_name[name].detail == "stopped while the GPU serves the language model"
    assert by_name[GPU_SHARING_CHECK].ok
    assert exit_code(results) == 0


def test_doctor_audio_phase_models_down_is_a_failure(
    engine: Engine, session_factory: sessionmaker[Session]
) -> None:
    _set(session_factory, GpuPhase.AUDIO)
    results = _diagnose(engine, _settings())
    assert not any(r.expected_stop for r in results if r.name == "transcription")
    assert exit_code(results) == 1


def test_doctor_audio_phase_llm_endpoint_down_is_expected(
    engine: Engine, session_factory: sessionmaker[Session]
) -> None:
    _set(session_factory, GpuPhase.AUDIO)
    seed_onboarded(session_factory, llm_enabled=True)
    results = _diagnose(
        engine, _settings(llm_enabled=True, llm_base_url=f"{CLOSED}/v1", llm_api_key="k")
    )
    endpoint = next(r for r in results if r.name == "llm endpoint")
    assert endpoint.expected_stop and not endpoint.hard
    assert "expected while Voxint holds the GPU" in endpoint.detail


def test_doctor_flags_a_phase_task_that_stopped_writing(
    engine: Engine, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        set_phase(session, GpuPhase.LLM, now=datetime.now(UTC) - timedelta(minutes=10))
        session.commit()
    sharing = next(r for r in _diagnose(engine, _settings()) if r.name == GPU_SHARING_CHECK)
    assert not sharing.ok and not sharing.hard
    assert sharing.detail.startswith("the GPU sharing task has not run since ")
    assert "gpu-phase service from compose.gpu-phase.yaml" in sharing.detail


def test_doctor_error_phase_reports_last_error(
    engine: Engine, session_factory: sessionmaker[Session]
) -> None:
    _set(session_factory, GpuPhase.ERROR, last_error="could not return the GPU: timed out")
    results = _diagnose(engine, _settings())
    sharing = next(r for r in results if r.name == GPU_SHARING_CHECK)
    assert not sharing.ok and not sharing.hard
    assert "timed out" in sharing.detail
    assert exit_code(results) == 0  # advisory; the model services are expected stops


# ---- Status page -------------------------------------------------------------


def test_status_page_has_gpu_sharing_row_only_when_on(
    session_factory: sessionmaker[Session], clean_resource_cache: None
) -> None:
    _set(session_factory, GpuPhase.LLM)
    on = _client(session_factory, _settings()).get("/settings/status").text
    assert "GPU sharing" in on
    assert "the GPU is serving the language model since" in on
    for label in ("Transcriber", "Voice separation", "Voice identity"):
        assert re.search(
            r'<span class="cr-dot is-off"></span>\s*'
            rf'<span class="cr-name">{label}</span>\s*'
            r'<span class="cr-state">stopped while the GPU serves the language model</span>',
            on,
        ), label

    off = _client(session_factory, _settings(gpu_phase_enabled=False)).get("/settings/status")
    assert "GPU sharing" not in off.text
    assert "stopped while the GPU serves the language model" not in off.text


def test_status_page_error_phase_needs_attention(
    session_factory: sessionmaker[Session], clean_resource_cache: None
) -> None:
    _set(session_factory, GpuPhase.ERROR, last_error="could not return the GPU: timed out")
    body = _client(session_factory, _settings()).get("/settings/status").text
    assert "Some parts need attention" in body
    assert "could not return the GPU to the language model" in body


# ---- Runs page -----------------------------------------------------------------


def test_runs_waiting_banner_strip_and_summary(
    session_factory: sessionmaker[Session], clean_resource_cache: None
) -> None:
    settings = _runs_settings(llm_enabled=True)
    client = _client(session_factory, settings, llm_enabled=True)
    _set(session_factory, GpuPhase.LLM)
    with session_factory() as session:
        seed_run(session, None)
        seed_run(session, Stage.TRANSCRIBE)
        session.commit()
    _prime_resource_cache(settings, transcription_up=False)

    body = client.get("/runs").text
    assert "<strong>Waiting for the GPU.</strong>" in body
    assert (
        "2 recordings are queued; the GPU is serving the language model "
        "until the next audio window." in body
    )
    assert 'data-gpu-sharing="llm"' in body and 'role="status"' in body
    # The model-service banner is replaced, not stacked.
    assert "Transcription is paused." not in body
    assert "paused: the GPU is serving the language model" in body
    assert "paused: transcriber is down" not in body
    assert "waiting for the GPU · 2 queued" in body

    strip = client.get("/runs/progress-strip").text
    assert "data-gpu-sharing-note>waiting for the GPU<" in strip
    assert "paused: the GPU is serving the language model" in strip


def test_runs_error_banner(
    session_factory: sessionmaker[Session], clean_resource_cache: None
) -> None:
    client = _client(session_factory, _settings())
    _set(session_factory, GpuPhase.ERROR)
    body = client.get("/runs").text
    assert "<strong>Could not return the GPU to the language model.</strong>" in body
    assert "LLM work is paused. Run <code>voxint gpu-phase release</code>" in body
    assert "or see the Status page." in body
    assert 'data-gpu-sharing="error" role="alert"' in body
    strip = client.get("/runs/progress-strip").text
    assert "GPU sharing needs attention" in strip


@pytest.mark.parametrize(
    ("phase", "headline"),
    [
        (GpuPhase.ACQUIRING, "Switching the GPU to audio work."),
        (GpuPhase.RELEASING, "Switching the GPU back."),
    ],
)
def test_runs_switching_banner(
    session_factory: sessionmaker[Session],
    clean_resource_cache: None,
    phase: GpuPhase,
    headline: str,
) -> None:
    client = _client(session_factory, _settings())
    _set(session_factory, phase)
    assert f"<strong>{headline}</strong>" in client.get("/runs").text


def test_runs_audio_phase_and_off_show_no_gpu_copy(
    session_factory: sessionmaker[Session], clean_resource_cache: None
) -> None:
    _set(session_factory, GpuPhase.AUDIO)
    with session_factory() as session:
        seed_run(session, None)
        session.commit()
    for settings in (_runs_settings(), _runs_settings(gpu_phase_enabled=False)):
        _reset_cache_for_tests()
        client = _client(session_factory, settings)
        _prime_resource_cache(settings, transcription_up=False)
        body = client.get("/runs").text
        assert "data-gpu-sharing" not in body
        assert "Waiting for the GPU" not in body
        # A down service outside the expected window is still reported as down.
        assert "Transcription is paused." in body
        assert "paused: transcriber is down" in body
