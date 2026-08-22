"""Route-level tests against the real FastAPI apps with fake model backends.

TestClient is used without its context manager on purpose: entering it runs
the lifespan, which loads real GPU models. These tests inject fakes instead
and exercise the error mapping the schema tests can't see.
"""

from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tests.contracts.conftest import load_service_main

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture
def media_root(tmp_path: Path) -> Path:
    root = tmp_path / "media"
    root.mkdir()
    (root / "audio.wav").write_bytes(b"RIFF0000WAVE")
    return root


def _client(mod: ModuleType) -> TestClient:
    return TestClient(mod.app, raise_server_exceptions=False)


class TestWhisperRoutes:
    @pytest.fixture
    def mod(self, media_root: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
        mod = load_service_main("whisper")
        monkeypatch.setattr(mod, "MEDIA_ROOT", media_root)
        monkeypatch.setattr(mod, "MAX_PENDING_REQUESTS", 8)
        monkeypatch.setattr(mod, "_pending", 0)
        fake_output = SimpleNamespace(
            language="en",
            duration_seconds=2.0,
            transcript="hello",
            confidence=0.9,
            segments=[
                {
                    "start_seconds": 0.0,
                    "end_seconds": 2.0,
                    "text": "hello",
                    "confidence": 0.9,
                    "suspect": False,
                    "suspect_score": None,
                    "suspect_span": None,
                }
            ],
            words=[],
            suspect_segment_count=0,
        )
        monkeypatch.setattr(
            mod,
            "transcriber",
            SimpleNamespace(
                is_initialized=True,
                model_name="large-v2",
                device="cpu",
                engine="faster-whisper",
                engine_version="test-engine-ver",
                runtime="ctranslate2",
                runtime_version="test-runtime-ver",
                transcribe=lambda *a, **k: fake_output,
                cleanup_memory=lambda: None,
                decode_identity=lambda: {
                    "vad_plan_version": "fw-1.2.1-batched-v1",
                    "vad_params": {"threshold": 0.5, "min_silence_duration_ms": 160},
                    "decode_config_hash": "deadbeef",
                    "model_revision": "test-rev",
                },
            ),
        )
        return mod

    def test_success_shape(self, mod: ModuleType) -> None:
        response = _client(mod).post("/v1/transcribe", json={"path": "audio.wav"})
        assert response.status_code == 200
        body = response.json()
        assert body["transcript"] == "hello"
        assert body["segments"][0]["suspect"] is False

    def test_healthz_ok(self, mod: ModuleType) -> None:
        response = _client(mod).get("/healthz")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model"] == "large-v2"
        assert body["contract_version"] == "v1"
        assert body["engine"] == "faster-whisper"
        assert body["engine_version"] == "test-engine-ver"
        assert body["runtime"] == "ctranslate2"
        assert body["runtime_version"] == "test-runtime-ver"
        # Decode identity (#33 Slice 2b) surfaced once loaded.
        assert body["vad_plan_version"] == "fw-1.2.1-batched-v1"
        assert body["vad_params"]["min_silence_duration_ms"] == 160
        assert body["decode_config_hash"] == "deadbeef"
        assert body["model_revision"] == "test-rev"

    def test_healthz_degraded_503_null_model(
        self, mod: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(mod.transcriber, "is_initialized", False)
        response = _client(mod).get("/healthz")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "degraded"
        assert body["model"] is None
        # Decode identity is null until the model is loaded.
        assert body["decode_config_hash"] is None
        assert body["vad_plan_version"] is None

    def test_model_unavailable(
        self, mod: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(mod.transcriber, "is_initialized", False)
        response = _client(mod).post("/v1/transcribe", json={"path": "audio.wav"})
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["code"] == "model_unavailable"
        assert detail["retryable"] is True

    def test_absolute_path_400(self, mod: ModuleType) -> None:
        response = _client(mod).post("/v1/transcribe", json={"path": "/etc/passwd"})
        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "path_violation"

    def test_missing_file_404(self, mod: ModuleType) -> None:
        response = _client(mod).post("/v1/transcribe", json={"path": "nope.wav"})
        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "file_not_found"

    def test_unknown_field_422(self, mod: ModuleType) -> None:
        response = _client(mod).post(
            "/v1/transcribe", json={"path": "audio.wav", "beam_size": 5}
        )
        assert response.status_code == 422

    def test_saturated_503(self, mod: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "MAX_PENDING_REQUESTS", 0)
        response = _client(mod).post("/v1/transcribe", json={"path": "audio.wav"})
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["code"] == "saturated"
        assert detail["retryable"] is True

    def test_decode_error_maps_to_invalid_media(
        self, mod: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*a: object, **k: object) -> None:
            raise mod.DecodeError("Could not decode audio: not audio")

        monkeypatch.setattr(mod.transcriber, "transcribe", boom)
        response = _client(mod).post("/v1/transcribe", json={"path": "audio.wav"})
        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "invalid_media"

    def test_inference_failure_500_not_retryable(
        self, mod: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*a: object, **k: object) -> None:
            raise RuntimeError("CUDA error")

        monkeypatch.setattr(mod.transcriber, "transcribe", boom)
        response = _client(mod).post("/v1/transcribe", json={"path": "audio.wav"})
        assert response.status_code == 500
        detail = response.json()["detail"]
        assert detail["code"] == "inference_failed"
        assert detail["retryable"] is False

    def test_pending_counter_released_after_failure(
        self, mod: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*a: object, **k: object) -> None:
            raise RuntimeError("CUDA error")

        monkeypatch.setattr(mod.transcriber, "transcribe", boom)
        client = _client(mod)
        for _ in range(3):
            client.post("/v1/transcribe", json={"path": "audio.wav"})
        assert mod._pending == 0


class TestPyannoteRoutes:
    @pytest.fixture
    def mod(self, media_root: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
        mod = load_service_main("pyannote")
        monkeypatch.setattr(mod, "MEDIA_ROOT", media_root)
        monkeypatch.setattr(mod, "_pending", 0)
        result = {
            "duration_seconds": 9.0,
            "num_speakers": 1,
            "turns": [
                {
                    "start_seconds": 0.0,
                    "end_seconds": 9.0,
                    "label": "SPEAKER_00",
                    "overlap": False,
                    "overlap_seconds": 0.0,
                }
            ],
            "speakers": [{"label": "SPEAKER_00", "total_seconds": 9.0, "num_turns": 1}],
        }
        monkeypatch.setattr(
            mod,
            "diarizer",
            SimpleNamespace(
                model_loaded=True,
                model_name="pyannote/speaker-diarization-3.1",
                device_name="cpu",
                engine="pyannote.audio",
                engine_version="test-engine-ver",
                runtime="torch",
                runtime_version="test-runtime-ver",
                model_revision=None,
                checkpoint_fingerprint="a" * 64,
                diarize=lambda *a, **k: result,
            ),
        )
        return mod

    def test_success_shape(self, mod: ModuleType) -> None:
        response = _client(mod).post("/v1/diarize", json={"path": "audio.wav"})
        assert response.status_code == 200
        assert response.json()["turns"][0]["label"] == "SPEAKER_00"

    def test_cross_field_validation_422(self, mod: ModuleType) -> None:
        response = _client(mod).post(
            "/v1/diarize",
            json={"path": "audio.wav", "min_speakers": 5, "max_speakers": 2},
        )
        assert response.status_code == 422

    def test_decode_error_maps_to_invalid_media(
        self, mod: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*a: object, **k: object) -> None:
            raise mod.DecodeError("Could not decode audio: not audio")

        monkeypatch.setattr(mod.diarizer, "diarize", boom)
        response = _client(mod).post("/v1/diarize", json={"path": "audio.wav"})
        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "invalid_media"

    def test_healthz_ok(self, mod: ModuleType) -> None:
        response = _client(mod).get("/healthz")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model"] == "pyannote/speaker-diarization-3.1"
        assert body["engine"] == "pyannote.audio"
        assert body["model_revision"] is None
        assert body["model_loaded"] is True
        # #125: the loaded-checkpoint fingerprint is on the identity contract.
        assert body["checkpoint_fingerprint"] == "a" * 64

    def test_healthz_degraded(self, mod: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod.diarizer, "model_loaded", False)
        response = _client(mod).get("/healthz")
        assert response.status_code == 503
        assert response.json()["model"] is None
        # Degraded: the fingerprint is not read from the (possibly unset) diarizer.
        assert response.json()["checkpoint_fingerprint"] is None


class TestTitanetRoutes:
    @pytest.fixture
    def mod(self, media_root: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
        mod = load_service_main("titanet")
        monkeypatch.setattr(mod, "MEDIA_ROOT", media_root)
        monkeypatch.setattr(mod, "_pending", 0)

        def fake_embed(path: str, windows: list[tuple[float, float]]) -> list[object]:
            outcomes: list[object] = []
            for start, end in windows:
                if end - start < 1.0:
                    outcomes.append(
                        SimpleNamespace(embedding=None, snr_db=None, skip_reason="too_short")
                    )
                else:
                    outcomes.append(
                        SimpleNamespace(
                            embedding=[0.01] * 192, snr_db=20.0, skip_reason=None
                        )
                    )
            return outcomes

        monkeypatch.setattr(
            mod,
            "embedder",
            SimpleNamespace(
                model_loaded=True,
                model_name="nvidia/speakerverification_en_titanet_large",
                device_name="cpu",
                engine="nemo",
                engine_version="test-engine-ver",
                runtime="torch",
                runtime_version="test-runtime-ver",
                embed_windows=fake_embed,
                cleanup_memory=lambda: None,
            ),
        )
        return mod

    def test_positional_alignment_with_skips(self, mod: ModuleType) -> None:
        response = _client(mod).post(
            "/v1/embed",
            json={
                "path": "audio.wav",
                "windows": [
                    {"start_seconds": 0.0, "end_seconds": 5.0},
                    {"start_seconds": 5.0, "end_seconds": 5.5},
                ],
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["embedding_space"] == "titanet-large-v1"
        assert len(body["results"]) == 2
        assert body["results"][0]["skip_reason"] is None
        assert body["results"][1]["skip_reason"] == "too_short"
        assert body["results"][1]["embedding"] is None

    def test_empty_windows_422(self, mod: ModuleType) -> None:
        response = _client(mod).post("/v1/embed", json={"path": "audio.wav", "windows": []})
        assert response.status_code == 422

    def test_traversal_400(self, mod: ModuleType) -> None:
        response = _client(mod).post(
            "/v1/embed",
            json={
                "path": "../audio.wav",
                "windows": [{"start_seconds": 0.0, "end_seconds": 5.0}],
            },
        )
        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "path_violation"
