"""HTTP client behavior against httpx.MockTransport — no services involved."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from voxint.clients.asr import HttpASRClient
from voxint.clients.diarize import HttpDiarizerClient
from voxint.clients.embed import MAX_WINDOWS_PER_REQUEST, HttpEmbedderClient
from voxint.clients.errors import ProtocolError, ServiceError, error_from_transport

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


def test_initial_prompt_sent_when_present() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"language": "en", "segments": []})

    client = make_client(HttpASRClient, handler)
    client.transcribe(AUDIO, initial_prompt="Foo, Bar")
    assert seen["initial_prompt"] == "Foo, Bar"


def test_initial_prompt_omitted_when_empty() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"language": "en", "segments": []})

    client = make_client(HttpASRClient, handler)
    client.transcribe(AUDIO, initial_prompt=None)
    assert "initial_prompt" not in seen
    # an empty string is treated the same as absent — no wasted field
    client.transcribe(AUDIO, initial_prompt="")
    assert "initial_prompt" not in seen


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


def test_connect_failure_names_service_host() -> None:
    # A dead/unresolvable service should read as "the service is down", naming
    # the host, not as a bare resolver error (issue #23).
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno -2] Name or service not known", request=request)

    client = make_client(HttpASRClient, handler)
    with pytest.raises(ServiceError) as exc_info:
        client.transcribe(AUDIO)
    assert exc_info.value.retryable is True
    assert "could not connect to 'test'" in exc_info.value.message
    assert "likely down" in exc_info.value.message


def test_connect_failure_without_request_still_maps() -> None:
    # httpx.ConnectError.request raises RuntimeError when never attached; the
    # hint must degrade gracefully rather than mask the transport error.
    err = error_from_transport(httpx.ConnectError("refused"))
    assert err.code == "transport_error"
    assert err.retryable is True
    assert "could not connect to the service:" in err.message


def test_read_timeout_keeps_plain_message() -> None:
    # Only connect failures get the "likely down" hint — a read timeout on a
    # reachable service must not claim the service is down.
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out", request=request)

    client = make_client(HttpASRClient, handler)
    with pytest.raises(ServiceError) as exc_info:
        client.transcribe(AUDIO)
    assert exc_info.value.code == "transport_error"
    assert exc_info.value.retryable is True
    assert "likely down" not in exc_info.value.message


def test_connect_timeout_keeps_plain_message() -> None:
    # Deliberate taxonomy decision (issue #23 review): ConnectTimeout is a
    # sibling of ConnectError, not a subclass, and stays PLAIN — a timeout on
    # a reachable host usually means overload or slow startup, not a dead
    # container. This pins the choice so a future widening is conscious.
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("connect timed out", request=request)

    client = make_client(HttpASRClient, handler)
    with pytest.raises(ServiceError) as exc_info:
        client.transcribe(AUDIO)
    assert exc_info.value.code == "transport_error"
    assert exc_info.value.retryable is True
    assert "likely down" not in exc_info.value.message


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
    assert result.segments[0].confidence == 0.95  # captured (issue #53)
    assert result.segments[1].suspect is True
    assert result.segments[1].confidence is None  # absent -> None, never fabricated


@pytest.mark.parametrize("bad", [1.5, -0.1, "0.9", True])
def test_transcribe_invalid_confidence_raises_protocol_error(bad: object) -> None:
    body = {
        "language": "en",
        "segments": [
            {"start_seconds": 0.0, "end_seconds": 1.0, "text": "hi", "confidence": bad}
        ],
    }
    client = make_client(HttpASRClient, lambda r: httpx.Response(200, json=body))
    with pytest.raises(ProtocolError):
        client.transcribe(AUDIO)


def test_transcribe_null_confidence_maps_to_none() -> None:
    body = {
        "language": "en",
        "segments": [
            {"start_seconds": 0.0, "end_seconds": 1.0, "text": "hi", "confidence": None}
        ],
    }
    client = make_client(HttpASRClient, lambda r: httpx.Response(200, json=body))
    assert client.transcribe(AUDIO).segments[0].confidence is None


def test_transcribe_malformed_body_raises_protocol_error() -> None:
    client = make_client(
        HttpASRClient, lambda r: httpx.Response(200, json={"segments": [{"oops": 1}]})
    )
    with pytest.raises(ProtocolError):
        client.transcribe(AUDIO)


def test_transcribe_maps_words() -> None:
    body = {
        "language": "en",
        "segments": [{"start_seconds": 0.0, "end_seconds": 2.0, "text": "hi there"}],
        "words": [
            {"start_seconds": 0.0, "end_seconds": 0.4, "word": "hi", "confidence": 0.99},
            {
                "start_seconds": 0.4,
                "end_seconds": 2.0,
                "word": "there",
                "confidence": None,
            },
        ],
    }
    client = make_client(HttpASRClient, lambda r: httpx.Response(200, json=body))
    result = client.transcribe(AUDIO)
    assert len(result.words) == 2
    assert result.words[0].word == "hi"
    assert result.words[0].confidence == 0.99
    assert result.words[1].confidence is None  # absent/null -> None, never fabricated


def test_transcribe_absent_words_key_is_empty_not_error() -> None:
    # A service or fake predating #59 omits the key entirely; that is "no word
    # timing", not a protocol violation.
    body = {
        "language": "en",
        "segments": [{"start_seconds": 0.0, "end_seconds": 1.0, "text": "hi"}],
    }
    client = make_client(HttpASRClient, lambda r: httpx.Response(200, json=body))
    assert client.transcribe(AUDIO).words == ()


def test_transcribe_explicit_null_words_is_protocol_error() -> None:
    # Present-but-null is NOT back-compat (an omitted key is): the v1 contract
    # types words as list[Word], so a null is a malformed current response and
    # must be loud, not silently treated like "no words".
    body = {
        "language": "en",
        "segments": [{"start_seconds": 0.0, "end_seconds": 1.0, "text": "hi"}],
        "words": None,
    }
    client = make_client(HttpASRClient, lambda r: httpx.Response(200, json=body))
    with pytest.raises(ProtocolError):
        client.transcribe(AUDIO)


@pytest.mark.parametrize(
    "words",
    [
        [{"start_seconds": 0.0, "end_seconds": 1.0, "confidence": 0.9}],  # no word text
        [{"start_seconds": 0.0, "end_seconds": 1.0, "word": 5}],  # word not a string
        [{"start_seconds": 0.0, "end_seconds": 1.0, "word": ""}],  # empty word
        [{"start_seconds": 1.0, "end_seconds": 0.0, "word": "x"}],  # reversed interval
        [{"start_seconds": "0", "end_seconds": 1.0, "word": "x"}],  # non-numeric bound
        [{"start_seconds": 0.0, "end_seconds": 1.0, "word": "x", "confidence": 2.0}],
        [5],  # word entry is not an object
        "not-a-list",
    ],
)
def test_transcribe_malformed_words_raise_protocol_error(words: object) -> None:
    body = {
        "language": "en",
        "segments": [{"start_seconds": 0.0, "end_seconds": 1.0, "text": "hi"}],
        "words": words,
    }
    client = make_client(HttpASRClient, lambda r: httpx.Response(200, json=body))
    with pytest.raises(ProtocolError):
        client.transcribe(AUDIO)


def test_transcribe_request_opts_into_autodetect() -> None:
    """The client sends an explicit ``language: null`` (#124): the contract's
    documented auto-detect path, chosen over changing the v1 omitted-field
    default (which is contract for other callers of a self-hosted service)."""
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"language": "es", "segments": []})

    client = make_client(HttpASRClient, handler)
    client.transcribe(AUDIO)
    assert "language" in captured[0]
    assert captured[0]["language"] is None


def test_transcribe_maps_language_probability() -> None:
    body = {
        "language": "es",
        "language_probability": 0.92,
        "segments": [{"start_seconds": 0.0, "end_seconds": 1.0, "text": "hola"}],
    }
    client = make_client(HttpASRClient, lambda r: httpx.Response(200, json=body))
    result = client.transcribe(AUDIO)
    assert result.language == "es"
    assert result.language_probability == 0.92


@pytest.mark.parametrize("shape", ["absent", "null"])
def test_transcribe_missing_language_probability_maps_to_none(shape: str) -> None:
    # An absent key is an older service predating #124 (back-compat); a null is
    # the documented "no detection ran" value. Both map to None, never a
    # fabricated score.
    body: dict[str, object] = {
        "language": "en",
        "segments": [{"start_seconds": 0.0, "end_seconds": 1.0, "text": "hi"}],
    }
    if shape == "null":
        body["language_probability"] = None
    client = make_client(HttpASRClient, lambda r: httpx.Response(200, json=body))
    assert client.transcribe(AUDIO).language_probability is None


@pytest.mark.parametrize("bad", [1.5, -0.1, "0.9", True, [0.9]])
def test_transcribe_invalid_language_probability_raises_protocol_error(
    bad: object,
) -> None:
    body = {
        "language": "en",
        "language_probability": bad,
        "segments": [{"start_seconds": 0.0, "end_seconds": 1.0, "text": "hi"}],
    }
    client = make_client(HttpASRClient, lambda r: httpx.Response(200, json=body))
    with pytest.raises(ProtocolError):
        client.transcribe(AUDIO)


@pytest.mark.parametrize("language", [None, ""])
def test_transcribe_probability_without_language_is_protocol_error(
    language: str | None,
) -> None:
    # A score describes a detected language; a null/empty language beside a
    # non-null score is contradictory provenance and must be loud.
    body = {
        "language": language,
        "language_probability": 0.9,
        "segments": [{"start_seconds": 0.0, "end_seconds": 1.0, "text": "hi"}],
    }
    client = make_client(HttpASRClient, lambda r: httpx.Response(200, json=body))
    with pytest.raises(ProtocolError):
        client.transcribe(AUDIO)


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_transcribe_nonfinite_language_probability_raises_protocol_error(
    bad: str,
) -> None:
    # Python's json.loads accepts these nonstandard literals, so a misbehaving
    # service CAN deliver them; sent as raw text because a strict serializer
    # (httpx's mock included) refuses to emit them.
    raw = (
        f'{{"language": "en", "language_probability": {bad},'
        ' "segments": [{"start_seconds": 0.0, "end_seconds": 1.0, "text": "hi"}]}'
    )
    client = make_client(
        HttpASRClient,
        lambda r: httpx.Response(
            200, content=raw.encode(), headers={"content-type": "application/json"}
        ),
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
    # The service's speaker count is surfaced back (was previously discarded).
    assert result.num_speakers == 2


def _diarize_body() -> dict[str, Any]:
    return {"duration_seconds": 1.0, "num_speakers": 1, "turns": [], "speakers": []}


def _capture_diarize_request(seen: dict[str, Any]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=_diarize_body())

    return handler


def test_diarize_sends_no_bounds_by_default() -> None:
    seen: dict[str, Any] = {}
    client = make_client(HttpDiarizerClient, _capture_diarize_request(seen))
    client.diarize(AUDIO)
    # Backward-compatible wire shape: with no hint, only the path is sent and the
    # service applies its own default ceiling.
    assert set(seen) == {"path"}


def test_diarize_sends_max_speakers_bound() -> None:
    seen: dict[str, Any] = {}
    client = make_client(HttpDiarizerClient, _capture_diarize_request(seen))
    client.diarize(AUDIO, max_speakers=4)
    assert seen["max_speakers"] == 4
    assert "min_speakers" not in seen  # a bound leaves min at the service default


def test_diarize_pins_bounds_for_exact_count() -> None:
    seen: dict[str, Any] = {}
    client = make_client(HttpDiarizerClient, _capture_diarize_request(seen))
    client.diarize(AUDIO, num_speakers=2)
    # An exact count is expressed as min == max == N (no service-side field).
    assert seen["min_speakers"] == 2
    assert seen["max_speakers"] == 2


def test_diarize_exact_count_wins_over_bound() -> None:
    seen: dict[str, Any] = {}
    client = make_client(HttpDiarizerClient, _capture_diarize_request(seen))
    client.diarize(AUDIO, max_speakers=8, num_speakers=3)
    assert seen["min_speakers"] == 3
    assert seen["max_speakers"] == 3


def test_diarize_rejects_non_integer_num_speakers() -> None:
    body = {**_diarize_body(), "num_speakers": "two"}
    client = make_client(HttpDiarizerClient, lambda r: httpx.Response(200, json=body))
    with pytest.raises(ProtocolError):
        client.diarize(AUDIO)


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
