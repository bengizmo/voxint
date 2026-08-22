"""Settings → "Pipeline models" panel render (configurable pipeline models, B1).

The live-identity collection and its classification are unit-tested in
``tests/unit/test_service_identity.py``; this drives the read-only panel through
the real ``GET /settings`` route and template against seeded database rows, with
the probe stubbed to fixed views so the render is hermetic (no live services).
Pins that each classified state renders its honest copy: a validated default, an
unvalidated override warning, the two fail-closed states (weights mismatch and
unverified), a fixed non-configurable model, and an unavailable service, plus the
"How to change" keys.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

import voxint.api.app as app_module
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
            detail="timeout",
            env_keys=(),
        ),
    ]


def _diarizer_view(verdict: ModelVerdict) -> ServiceIdentityView:
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
        detail=None,
        env_keys=("DIARIZER_MODEL_NAME", "DIARIZER_REVISION"),
    )


def test_pipeline_models_panel_renders_each_state(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "collect_service_identity", lambda _settings: _views())
    body = client.get("/settings").text

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
        detail=None,
        env_keys=(),
    )
    views[2] = reachable_embedder
    monkeypatch.setattr(app_module, "collect_service_identity", lambda _settings: views)
    body = client.get("/settings").text
    assert "nvidia/speakerverification_en_titanet_large" in body
    assert "This model is fixed" in body


def test_pipeline_models_panel_renders_weights_mismatch(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # #125: a validated name whose loaded weights differ renders the fail-closed
    # mismatch copy, not the validated confirmation.
    views = _views()
    views[1] = _diarizer_view(ModelVerdict.MISMATCH)
    monkeypatch.setattr(app_module, "collect_service_identity", lambda _settings: views)
    body = client.get("/settings").text
    assert "do not match the validated" in body
    assert "This is the validated model" in body  # asr is still validated
    assert "re-pull or rebuild" in body


def test_pipeline_models_panel_renders_unverified(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # #125: a validated name whose weights cannot be verified (online source)
    # renders the amber unverified copy.
    views = _views()
    views[1] = _diarizer_view(ModelVerdict.UNVERIFIED)
    monkeypatch.setattr(app_module, "collect_service_identity", lambda _settings: views)
    body = client.get("/settings").text
    assert "cannot verify the loaded model files" in body
