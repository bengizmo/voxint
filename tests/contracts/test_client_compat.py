"""Service responses must remain convertible to the client result types.

These adapters mirror what the P2/P3 HTTP clients will do; if a schema rename
or reorder breaks the mapping, this file fails before any GPU is involved.
"""

from types import ModuleType

from tests.contracts.conftest import load_fixture
from voxint.clients.base import (
    DiarizationResult,
    DiarizationTurn,
    EmbeddingEntry,
    EmbeddingResult,
    TranscriptionResult,
    TranscriptionSegment,
)


def test_whisper_response_to_transcription_result(whisper_schemas: ModuleType) -> None:
    response = whisper_schemas.TranscribeResponse.model_validate(
        load_fixture("whisper_transcribe.json")["response"]
    )
    result = TranscriptionResult(
        segments=tuple(
            TranscriptionSegment(
                start_seconds=s.start_seconds,
                end_seconds=s.end_seconds,
                text=s.text,
                suspect=s.suspect,
            )
            for s in response.segments
        ),
        language=response.language,
    )
    assert result.language == "en"
    assert result.segments[1].suspect is True
    assert result.segments[0].text == "hello and welcome"


def test_pyannote_response_to_diarization_result(pyannote_schemas: ModuleType) -> None:
    response = pyannote_schemas.DiarizeResponse.model_validate(
        load_fixture("pyannote_diarize.json")["response"]
    )
    result = DiarizationResult(
        turns=tuple(
            DiarizationTurn(
                start_seconds=t.start_seconds, end_seconds=t.end_seconds, label=t.label
            )
            for t in response.turns
        )
    )
    assert [t.label for t in result.turns] == ["SPEAKER_00", "SPEAKER_01"]


def test_titanet_response_to_embedding_result(titanet_schemas: ModuleType) -> None:
    fixture = load_fixture("titanet_embed.json")
    for result_row in fixture["response"]["results"]:
        if result_row["embedding"] == "__FILL_192__":
            result_row["embedding"] = [0.01] * 192
    response = titanet_schemas.EmbedResponse.model_validate(fixture["response"])

    result = EmbeddingResult(
        embedding_space=response.embedding_space,
        entries=tuple(
            EmbeddingEntry(
                embedding=tuple(r.embedding) if r.embedding is not None else None,
                snr_db=r.snr_db,
                skip_reason=r.skip_reason,
            )
            for r in response.results
        ),
    )
    # Positional alignment: one entry per requested window, same order.
    assert len(result.entries) == len(fixture["request"]["windows"])
    assert result.entries[0].embedding is not None
    assert len(result.entries[0].embedding) == 192
    assert result.entries[1].skip_reason == "too_short"
    assert result.entries[2].skip_reason == "low_snr"
    assert result.embedding_space == "titanet-large-v1"
