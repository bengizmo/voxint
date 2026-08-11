"""HTTP client behavior against httpx.MockTransport — no services involved."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from voxint.clients.asr import HttpASRClient
from voxint.clients.diarize import HttpDiarizerClient
from voxint.clients.embed import MAX_WINDOWS_PER_REQUEST, HttpEmbedderClient
from voxint.clients.errors import ProtocolError, ServiceError

MEDIA_ROOT = Path("/data/media")
AUDIO = MEDIA_ROOT / "runs/abc/normalized.wav"


def make_client(
    cls: type, handler: Any, media_root: Path = MEDIA_ROOT
) -> Any:
    http = httpx.Client(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )
    return cls("http://test", media_root, timeout_seconds=5.0, client=http)


def structured_error(status: int, code: str, *, retryable: bool) -> httpx.Response:
    return httpx.Response(
        status, json={"detail": {"code": code, "message": "boom", "retryable": retryable}}
    )


# ---------------------------------------------------------------- path handling


def test_relative_path_sent_posix() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"language": "en", "segments": []})

    client = make_client(HttpASRClient, handler)
    client.transcribe(AUDIO)
    assert seen["path"] == "runs/abc/normalized.wav"


def test_path_outside_media_root_is_non_retryable() -> None:
    client = make_client(HttpASRClient, lambda r: httpx.Response(500))
    with pytest.raises(ServiceError) as exc_info:
        client.transcribe(Path("/etc/passwd"))
    assert exc_info.value.code == "path_violation"
    assert exc_info.value.retryable is False


def test_relative_input_path_rejected() -> None:
    client = make_client(HttpASRClient, lambda r: httpx.Response(500))
    with pytest.raises(ServiceError) as exc_info:
        client.transcribe(Path("runs/abc.wav"))
    assert exc_info.value.code == "path_violation"


def test_dotdot_escape_rejected() -> None:
    client = make_client(HttpASRClient, lambda r: httpx.Response(500))
    with pytest.raises(ServiceError) as exc_info:
        client.transcribe(MEDIA_ROOT / ".." / "secret.wav")
    assert exc_info.value.code == "path_violation"


# ---------------------------------------------------------------- error mapping


def test_structured_error_respects_retryable_flag() -> None:
    client = make_client(
        HttpASRClient, lambda r: structured_error(503, "saturated", retryable=True)
    )
    with pytest.raises(ServiceError) as exc_info:
        client.transcribe(AUDIO)
    assert exc_info.value.code == "saturated"
    assert exc_info.value.retryable is True
    assert exc_info.value.status_code == 503


def test_structured_500_not_retryable() -> None:
    client = make_client(
        HttpASRClient, lambda r: structured_error(500, "inference_failed", retryable=False)
    )
    with pytest.raises(ServiceError) as exc_info:
        client.transcribe(AUDIO)
    assert exc_info.value.retryable is False


def test_non_conforming_5xx_presumed_retryable() -> None:
    client = make_client(
        HttpASRClient, lambda r: httpx.Response(502, text="<html>bad gateway</html>")
    )
    with pytest.raises(ServiceError) as exc_info:
        client.transcribe(AUDIO)
    assert exc_info.value.code == "http_error"
    assert exc_info.value.retryable is True


def test_non_conforming_4xx_not_retryable() -> None:
    client = make_client(HttpASRClient, lambda r: httpx.Response(404, text="nope"))
    with pytest.raises(ServiceError) as exc_info:
        client.transcribe(AUDIO)
    assert exc_info.value.retryable is False


def test_fastapi_422_not_retryable() -> None:
    client = make_client(
        HttpASRClient,
        lambda r: httpx.Response(422, json={"detail": [{"loc": ["body"], "msg": "bad"}]}),
    )
    with pytest.raises(ServiceError) as exc_info:
        client.transcribe(AUDIO)
    assert exc_info.value.code == "validation_error"
    assert exc_info.value.retryable is False


def test_transport_failure_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = make_client(HttpASRClient, handler)
    with pytest.raises(ServiceError) as exc_info:
        client.transcribe(AUDIO)
    assert exc_info.value.code == "transport_error"
    assert exc_info.value.retryable is True


def test_non_json_2xx_is_protocol_violation() -> None:
    client = make_client(HttpASRClient, lambda r: httpx.Response(200, text="ok"))
    with pytest.raises(ServiceError) as exc_info:
        client.transcribe(AUDIO)
    assert exc_info.value.code == "protocol_violation"
    assert exc_info.value.retryable is False


# ---------------------------------------------------------------- ASR mapping


def test_transcribe_maps_segments_and_tolerates_additive_fields() -> None:
    body = {
        "language": "en",
        "duration_seconds": 9.0,
        "transcript": "hello",
        "confidence": 0.9,
        "future_field": {"anything": 1},
        "segments": [
            {
                "start_seconds": 0.0,
                "end_seconds": 4.0,
                "text": "hello",
                "confidence": 0.95,
                "suspect": False,
            },
            {"start_seconds": 4.0, "end_seconds": 5.0, "text": "mm", "suspect": True},
        ],
        "words": [],
    }
    client = make_client(HttpASRClient, lambda r: httpx.Response(200, json=body))
    result = client.transcribe(AUDIO)
    assert result.language == "en"
    assert len(result.segments) == 2
    assert result.segments[0].text == "hello"
    assert result.segments[1].suspect is True


def test_transcribe_malformed_body_raises_protocol_error() -> None:
    client = make_client(
        HttpASRClient, lambda r: httpx.Response(200, json={"segments": [{"oops": 1}]})
    )
    with pytest.raises(ProtocolError):
        client.transcribe(AUDIO)


# ---------------------------------------------------------------- diarizer mapping


def test_diarize_maps_turns() -> None:
    body = {
        "duration_seconds": 9.0,
        "num_speakers": 2,
        "turns": [
            {"start_seconds": 0.0, "end_seconds": 4.0, "label": "SPEAKER_00"},
            {
                "start_seconds": 4.0,
                "end_seconds": 9.0,
                "label": "SPEAKER_01",
                "overlap": True,
                "overlap_seconds": 0.5,
            },
        ],
        "speakers": [],
    }
    client = make_client(HttpDiarizerClient, lambda r: httpx.Response(200, json=body))
    result = client.diarize(AUDIO)
    assert [t.label for t in result.turns] == ["SPEAKER_00", "SPEAKER_01"]
    assert result.turns[1].end_seconds == 9.0


# ---------------------------------------------------------------- embedder


def embed_response(count: int, space: str = "titanet-large-v1") -> dict[str, Any]:
    return {
        "embedding_space": space,
        "results": [
            {"embedding": [0.01] * 192, "snr_db": 20.0, "skip_reason": None}
            for _ in range(count)
        ],
    }


def test_embed_single_batch() -> None:
    client = make_client(
        HttpEmbedderClient, lambda r: httpx.Response(200, json=embed_response(2))
    )
    result = client.embed(AUDIO, ((0.0, 1.0), (1.0, 2.5)))
    assert result.embedding_space == "titanet-large-v1"
    assert len(result.entries) == 2
    assert result.entries[0].embedding is not None
    assert len(result.entries[0].embedding) == 192


def test_embed_skip_entries_mapped() -> None:
    body = {
        "embedding_space": "titanet-large-v1",
        "results": [
            {"embedding": None, "snr_db": None, "skip_reason": "too_short"},
            {"embedding": None, "snr_db": 1.2, "skip_reason": "low_snr"},
        ],
    }
    client = make_client(HttpEmbedderClient, lambda r: httpx.Response(200, json=body))
    result = client.embed(AUDIO, ((0.0, 0.5), (0.5, 1.0)))
    assert result.entries[0].skip_reason == "too_short"
    assert result.entries[0].embedding is None
    assert result.entries[1].skip_reason == "low_snr"
    assert result.entries[1].snr_db == 1.2


def test_embed_batches_over_512_windows() -> None:
    batch_sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        n = len(json.loads(request.content)["windows"])
        batch_sizes.append(n)
        return httpx.Response(200, json=embed_response(n))

    client = make_client(HttpEmbedderClient, handler)
    windows = tuple((float(i), float(i) + 1.0) for i in range(MAX_WINDOWS_PER_REQUEST + 3))
    result = client.embed(AUDIO, windows)
    assert batch_sizes == [MAX_WINDOWS_PER_REQUEST, 3]
    assert len(result.entries) == MAX_WINDOWS_PER_REQUEST + 3


def test_embed_zero_windows_is_caller_bug() -> None:
    client = make_client(HttpEmbedderClient, lambda r: httpx.Response(500))
    with pytest.raises(ValueError):
        client.embed(AUDIO, ())


def test_embed_count_mismatch_is_protocol_error() -> None:
    client = make_client(
        HttpEmbedderClient, lambda r: httpx.Response(200, json=embed_response(1))
    )
    with pytest.raises(ProtocolError):
        client.embed(AUDIO, ((0.0, 1.0), (1.0, 2.0)))


def test_embed_wrong_dimension_is_protocol_error() -> None:
    body = {
        "embedding_space": "titanet-large-v1",
        "results": [{"embedding": [0.01] * 191, "snr_db": 20.0, "skip_reason": None}],
    }
    client = make_client(HttpEmbedderClient, lambda r: httpx.Response(200, json=body))
    with pytest.raises(ProtocolError):
        client.embed(AUDIO, ((0.0, 1.0),))


def test_embed_both_embedding_and_skip_is_protocol_error() -> None:
    body = {
        "embedding_space": "titanet-large-v1",
        "results": [{"embedding": [0.01] * 192, "snr_db": 20.0, "skip_reason": "low_snr"}],
    }
    client = make_client(HttpEmbedderClient, lambda r: httpx.Response(200, json=body))
    with pytest.raises(ProtocolError):
        client.embed(AUDIO, ((0.0, 1.0),))


def test_embed_space_change_across_batches_is_protocol_error() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        n = len(json.loads(request.content)["windows"])
        return httpx.Response(200, json=embed_response(n, space=f"space-{calls['n']}"))

    client = make_client(HttpEmbedderClient, handler)
    windows = tuple((float(i), float(i) + 1.0) for i in range(MAX_WINDOWS_PER_REQUEST + 1))
    with pytest.raises(ProtocolError):
        client.embed(AUDIO, windows)


# ---------------------------------------------------------------- lifecycle


def test_close_leaves_shared_client_open() -> None:
    http = httpx.Client(
        base_url="http://test",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
    )
    client = HttpASRClient("http://test", MEDIA_ROOT, timeout_seconds=5.0, client=http)
    client.close()
    assert not http.is_closed


def test_owned_client_closed_via_context_manager() -> None:
    with HttpASRClient("http://test", MEDIA_ROOT, timeout_seconds=5.0) as client:
        pass
    assert client._client.is_closed
