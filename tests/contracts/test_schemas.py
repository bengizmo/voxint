"""Schema-level contract tests: fixtures parse, malformed variants reject."""

from types import ModuleType
from typing import Any

import pytest
from pydantic import ValidationError

from tests.contracts.conftest import load_fixture


def _fill_embedding(response: dict[str, Any]) -> dict[str, Any]:
    for result in response["results"]:
        if result["embedding"] == "__FILL_192__":
            result["embedding"] = [0.01] * 192
    return response


class TestWhisperSchemas:
    def test_fixture_roundtrip(self, whisper_schemas: ModuleType) -> None:
        fixture = load_fixture("whisper_transcribe.json")
        request = whisper_schemas.TranscribeRequest.model_validate(fixture["request"])
        assert request.path == "items/1234/audio.16khz.wav"
        response = whisper_schemas.TranscribeResponse.model_validate(fixture["response"])
        assert response.suspect_segment_count == 1
        assert response.segments[1].suspect is True

    def test_unknown_request_field_rejected(self, whisper_schemas: ModuleType) -> None:
        with pytest.raises(ValidationError):
            whisper_schemas.TranscribeRequest.model_validate(
                {"path": "a.wav", "beam_size": 5}
            )

    def test_language_null_means_autodetect(self, whisper_schemas: ModuleType) -> None:
        request = whisper_schemas.TranscribeRequest.model_validate(
            {"path": "a.wav", "language": None}
        )
        assert request.language is None

    def test_oversized_initial_prompt_rejected(self, whisper_schemas: ModuleType) -> None:
        with pytest.raises(ValidationError):
            whisper_schemas.TranscribeRequest.model_validate(
                {"path": "a.wav", "initial_prompt": "x" * 2001}
            )


class TestPyannoteSchemas:
    def test_fixture_roundtrip(self, pyannote_schemas: ModuleType) -> None:
        fixture = load_fixture("pyannote_diarize.json")
        pyannote_schemas.DiarizeRequest.model_validate(fixture["request"])
        response = pyannote_schemas.DiarizeResponse.model_validate(fixture["response"])
        assert response.num_speakers == 2
        assert response.turns[1].overlap_seconds == pytest.approx(0.4)

    def test_speaker_bounds_cross_field(self, pyannote_schemas: ModuleType) -> None:
        with pytest.raises(ValidationError):
            pyannote_schemas.DiarizeRequest.model_validate(
                {"path": "a.wav", "min_speakers": 5, "max_speakers": 2}
            )

    def test_unknown_request_field_rejected(self, pyannote_schemas: ModuleType) -> None:
        with pytest.raises(ValidationError):
            pyannote_schemas.DiarizeRequest.model_validate(
                {"path": "a.wav", "recording_setup": "single_mic"}
            )

    @pytest.mark.parametrize("value", [0, 21])
    def test_speaker_range_bounds(self, pyannote_schemas: ModuleType, value: int) -> None:
        with pytest.raises(ValidationError):
            pyannote_schemas.DiarizeRequest.model_validate(
                {"path": "a.wav", "max_speakers": value}
            )


class TestTitanetSchemas:
    def test_fixture_roundtrip(self, titanet_schemas: ModuleType) -> None:
        fixture = load_fixture("titanet_embed.json")
        request = titanet_schemas.EmbedRequest.model_validate(fixture["request"])
        assert len(request.windows) == 3
        response = titanet_schemas.EmbedResponse.model_validate(
            _fill_embedding(fixture["response"])
        )
        assert response.embedding_space == "titanet-large-v1"
        assert len(response.results) == len(request.windows)
        assert response.results[1].embedding is None
        assert response.results[1].snr_db is None  # too_short → SNR not measured

    def test_empty_windows_rejected(self, titanet_schemas: ModuleType) -> None:
        with pytest.raises(ValidationError):
            titanet_schemas.EmbedRequest.model_validate({"path": "a.wav", "windows": []})

    def test_oversized_window_list_rejected(self, titanet_schemas: ModuleType) -> None:
        windows = [{"start_seconds": float(i), "end_seconds": i + 1.0} for i in range(513)]
        with pytest.raises(ValidationError):
            titanet_schemas.EmbedRequest.model_validate({"path": "a.wav", "windows": windows})

    @pytest.mark.parametrize(
        "window",
        [
            {"start_seconds": -1.0, "end_seconds": 2.0},
            {"start_seconds": 3.0, "end_seconds": 3.0},
            {"start_seconds": 4.0, "end_seconds": 2.0},
            {"start_seconds": float("inf"), "end_seconds": 2.0},
        ],
    )
    def test_bad_window_bounds_rejected(
        self, titanet_schemas: ModuleType, window: dict[str, float]
    ) -> None:
        with pytest.raises(ValidationError):
            titanet_schemas.EmbedRequest.model_validate({"path": "a.wav", "windows": [window]})

    def test_result_invariant_embedding_xor_skip(self, titanet_schemas: ModuleType) -> None:
        # both set
        with pytest.raises(ValidationError):
            titanet_schemas.WindowResult.model_validate(
                {"embedding": [0.1] * 192, "skip_reason": "low_snr"}
            )
        # neither set
        with pytest.raises(ValidationError):
            titanet_schemas.WindowResult.model_validate({"embedding": None, "skip_reason": None})

    def test_result_dimension_enforced(self, titanet_schemas: ModuleType) -> None:
        with pytest.raises(ValidationError):
            titanet_schemas.WindowResult.model_validate(
                {"embedding": [0.1] * 191, "skip_reason": None}
            )
