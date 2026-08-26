"""Settings → "Pipeline models" panel render (configurable pipeline models, B1).

The live-identity collection and its classification are unit-tested in
``tests/unit/test_service_identity.py``; this drives the read-only panel through
the real ``GET /settings/hardware`` route and template against seeded database
rows, with the probe stubbed to fixed views so the render is hermetic (no live
services). The panel moved off the flat ``/settings`` page onto the hardware
sub-page when the settings hub activated (Console 2.0 P6b, #161); it reuses the
same ``settings/_models.html`` partial, so the rendered copy is unchanged.
Pins that each classified state renders its honest copy: a validated default, an
unvalidated override warning, the two fail-closed states (weights mismatch and
unverified), a fixed non-configurable model, and an unavailable service, plus the
"How to change" keys.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

import voxint.api.routers.settings as settings_module
from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.service_identity import ModelVerdict, ServiceIdentityView
from voxint.config import Settings

CREDS = ("reviewer", "s3cret")


@pytest.fixture()
def client(session_factory: sessionmaker[Session]) -> TestClient:
    settings = Settings(voxint_user=CREDS[0], voxint_password=CREDS[1])
    test_client = TestClient(create_app(settings=settings, session_factory=session_factory))
    test_client.auth = CREDS
    seed_onboarded(session_factory)
    return test_client


def _views() -> list[ServiceIdentityView]:
    return [
        ServiceIdentityView(
            role="asr",
            label="Transcription",
            url="http://localhost:8022",
            reachable=True,
            model="large-v2",
            revision="f" * 40,
            engine="faster-whisper",
            configurable=True,
            verdict=ModelVerdict.VALIDATED,
            identity_axis=None,
            detail=None,
            env_keys=("WHISPER_MODEL", "WHISPER_REVISION", "WHISPER_ALLOW_DOWNLOAD"),
        ),
        ServiceIdentityView(
            role="diarizer",
            label="Speaker diarization",
            url="http://localhost:8024",
            reachable=True,
            model="pyannote/speaker-diarization-4.0",
            revision=None,
            engine="pyannote.audio",
            configurable=True,
            verdict=ModelVerdict.UNVALIDATED,
            identity_axis=None,
            detail=None,
            env_keys=("DIARIZER_MODEL_NAME", "DIARIZER_REVISION"),
        ),
        ServiceIdentityView(
            role="embedder",
            label="Speaker embedding",
            url="http://localhost:8021",
            reachable=False,
            model=None,
            revision=None,
            engine=None,
            configurable=False,
            verdict=ModelVerdict.UNVALIDATED,
            identity_axis=None,
            detail="timeout",
            env_keys=(),
        ),
    ]


def _diarizer_view(
    verdict: ModelVerdict, identity_axis: str | None = "weights"
) -> ServiceIdentityView:
    return ServiceIdentityView(
        role="diarizer",
        label="Speaker diarization",
        url="http://localhost:8024",
        reachable=True,
        model="pyannote/speaker-diarization-3.1",
        revision=None,
        engine="pyannote.audio",
        configurable=True,
        verdict=verdict,
        identity_axis=identity_axis,
        detail=None,
        env_keys=("DIARIZER_MODEL_NAME", "DIARIZER_REVISION"),
    )


def test_pipeline_models_panel_renders_each_state(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings_module, "collect_service_identity", lambda _settings: _views())
    body = client.get("/settings/hardware").text

    # The panel and its heading are present.
    assert 'id="pipeline-models"' in body
    assert "Pipeline models" in body

    # Validated default: the model id and the validated confirmation.
    assert "large-v2" in body
    assert "This is the validated model" in body

    # Unvalidated override: the warning copy and the reported override model.
    assert "This is not the validated model" in body
    assert "pyannote/speaker-diarization-4.0" in body

    # Unavailable service: honest unavailable copy with the detail and the address.
    assert "Unavailable (timeout)" in body
    assert "http://localhost:8021" in body

    # How-to-change keys for the two configurable services.
    assert "WHISPER_MODEL" in body
    assert "DIARIZER_MODEL_NAME" in body


def test_pipeline_models_panel_marks_fixed_embedder(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A reachable, non-configurable embedder shows its identity and the fixed note,
    # never a validated/unvalidated badge.
    views = _views()
    reachable_embedder = ServiceIdentityView(
        role="embedder",
        label="Speaker embedding",
        url="http://localhost:8021",
        reachable=True,
        model="nvidia/speakerverification_en_titanet_large",
        revision=None,
        engine="nemo",
        configurable=False,
        verdict=ModelVerdict.VALIDATED,
        identity_axis=None,
        detail=None,
        env_keys=(),
    )
    views[2] = reachable_embedder
    monkeypatch.setattr(settings_module, "collect_service_identity", lambda _settings: views)
    body = client.get("/settings/hardware").text
    assert "nvidia/speakerverification_en_titanet_large" in body
    assert "This model is fixed" in body


def test_pipeline_models_panel_renders_weights_mismatch(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # #125: a validated name whose loaded weights differ renders the fail-closed
    # mismatch copy, not the validated confirmation.
    views = _views()
    views[1] = _diarizer_view(ModelVerdict.MISMATCH)
    monkeypatch.setattr(settings_module, "collect_service_identity", lambda _settings: views)
    body = client.get("/settings/hardware").text
    assert "do not match the validated" in body
    assert "This is the validated model" in body  # asr is still validated
    assert "re-pull or rebuild" in body


def test_pipeline_models_panel_renders_asr_mismatch_with_revision_guidance(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # #125 review: an ASR weights mismatch is caused by an overridden
    # WHISPER_REVISION, which a rebuild does not clear, so the copy must tell the
    # operator to remove WHISPER_REVISION rather than re-pull/rebuild.
    views = _views()
    views[0] = ServiceIdentityView(
        role="asr",
        label="Transcription",
        url="http://localhost:8022",
        reachable=True,
        model="large-v2",
        revision="0" * 40,
        engine="faster-whisper",
        configurable=True,
        verdict=ModelVerdict.MISMATCH,
        identity_axis="weights",
        detail=None,
        env_keys=("WHISPER_MODEL", "WHISPER_REVISION", "WHISPER_ALLOW_DOWNLOAD"),
    )
    monkeypatch.setattr(settings_module, "collect_service_identity", lambda _settings: views)
    body = client.get("/settings/hardware").text
    assert "different version of the weights" in body
    assert "Remove <code>WHISPER_REVISION</code>" in body
    assert "re-pull or rebuild" not in body  # the diarizer-only guidance


def test_pipeline_models_panel_renders_unverified(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # #125: a validated name whose weights cannot be verified (online source)
    # renders the amber unverified copy.
    views = _views()
    views[1] = _diarizer_view(ModelVerdict.UNVERIFIED)
    monkeypatch.setattr(settings_module, "collect_service_identity", lambda _settings: views)
    body = client.get("/settings/hardware").text
    assert "cannot verify the loaded model files" in body


def test_pipeline_models_panel_renders_config_mismatch(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # #129: a validated name + weights but a drifted clustering config renders the
    # config-axis remedy (reset the PYANNOTE_CLUSTERING_* env vars), NOT the
    # weights re-pull/rebuild copy.
    views = _views()
    views[1] = _diarizer_view(ModelVerdict.MISMATCH, identity_axis="config")
    monkeypatch.setattr(settings_module, "collect_service_identity", lambda _settings: views)
    body = client.get("/settings/hardware").text
    assert "clustering configuration does not match" in body
    assert "PYANNOTE_CLUSTERING_THRESHOLD" in body
    assert "re-pull or rebuild" not in body  # the weights-only remedy


def test_pipeline_models_panel_renders_config_unverified(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # #129: a validated name whose clustering config cannot be verified renders the
    # config-axis unverified copy, distinct from the weights-files copy.
    views = _views()
    views[1] = _diarizer_view(ModelVerdict.UNVERIFIED, identity_axis="config")
    monkeypatch.setattr(settings_module, "collect_service_identity", lambda _settings: views)
    body = client.get("/settings/hardware").text
    assert "cannot verify its clustering configuration" in body


def test_flat_page_still_renders_the_panel_when_flag_off(
    session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The panel's home moved to /settings/hardware when the hub activated, but the
    # flag-off flat page keeps rendering it inline. Pin that fallback (#161 P6b).
    settings = Settings(
        voxint_user=CREDS[0], voxint_password=CREDS[1], console_settings_enabled=False
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory)
    monkeypatch.setattr(settings_module, "collect_service_identity", lambda _settings: _views())
    body = client.get("/settings").text
    assert 'id="pipeline-models"' in body
    assert "large-v2" in body


def test_activated_hub_does_not_probe_model_services(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The hub (default on) moved the panel to /settings/hardware, so rendering
    # /settings must NOT probe the three model services (#161 P6b review): a probe
    # here would be wasted work on every hub render and POST re-render. Fail loudly
    # if the shared context resumes probing behind the hub.
    def _boom(_settings: Settings) -> list[ServiceIdentityView]:
        raise AssertionError("hub render must not probe model-service identity")

    monkeypatch.setattr(settings_module, "collect_service_identity", _boom)
    resp = client.get("/settings")
    assert resp.status_code == 200
    assert 'id="pipeline-models"' not in resp.text
