"""Per-attempt model-identity probe against httpx.MockTransport — no services.

Exercises every normalized per-role outcome (ready with identity, missing
identity fields, degraded 503, not-ready 200, malformed body, timeout, transport
error), the stage->services mapping (TRANSCRIBE=asr, DIARIZE_EMBED=diarizer+embedder,
a non-model stage -> None), and that the probe never raises into the caller.
"""

import httpx

from voxint.config import Settings
from voxint.db.models import Stage
from voxint.pipeline.model_identity import (
    CHECKPOINT_FINGERPRINT_FIELD,
    DIARIZATION_CONFIG_HASH_FIELD,
    IDENTITY_SCHEMA_VERSION,
    observe_stage_model_identity,
    probe_identity_one,
    stage_has_model_identity,
)

# Default service ports (config.py); the probe hits "{url}/healthz" so the handler
# dispatches on the request's port to give each service its own response.
_ASR_PORT = 8022
_DIARIZER_PORT = 8024
_EMBEDDER_PORT = 8021


def _settings() -> Settings:
    return Settings(voxint_user="u", voxint_password="p")


def _client(by_port: dict[int, object]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/healthz"
        outcome = by_port[request.url.port or 0]
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, httpx.Response)
        return outcome

    return httpx.Client(transport=httpx.MockTransport(handler))


def _whisper_ready() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "status": "ok",
            "service": "whisper",
            "model": "large-v2",
            "model_revision": "a" * 40,
            "engine": "ct2-legacy",
            "decode_config_hash": "deadbeef",
            "model_loaded": True,
        },
    )


def _pyannote_ready() -> httpx.Response:
    # No decode_config_hash / model_revision on this service today.
    return httpx.Response(
        200,
        json={
            "status": "ok",
            "service": "pyannote",
            "model": "pyannote/speaker-diarization-3.1",
            "engine": "pyannote.audio",
            "model_loaded": True,
        },
    )


def _titanet_ready() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "status": "ok",
            "service": "titanet",
            "model": "titanet-large-v1",
            "engine": "onnxruntime",
            "model_loaded": True,
        },
    )


def test_stage_has_model_identity_mapping() -> None:
    assert stage_has_model_identity(Stage.TRANSCRIBE)
    assert stage_has_model_identity(Stage.DIARIZE_EMBED)
    assert not stage_has_model_identity(Stage.PREPARE)
    assert not stage_has_model_identity(Stage.ACQUIRE)
    assert not stage_has_model_identity(Stage.FINALIZE)


def test_non_model_stage_returns_none() -> None:
    client = _client({})  # never probed
    assert observe_stage_model_identity(_settings(), Stage.PREPARE, client=client) is None


def test_transcribe_captures_asr_identity() -> None:
    client = _client({_ASR_PORT: _whisper_ready()})
    result = observe_stage_model_identity(_settings(), Stage.TRANSCRIBE, client=client)
    assert result is not None
    assert result["v"] == IDENTITY_SCHEMA_VERSION
    assert result["observed_before_attempt"] is True
    assert result["asr"] == {
        "reachable": True,
        "model": "large-v2",
        "revision": "a" * 40,
        "engine": "ct2-legacy",
        "decode_config_hash": "deadbeef",
    }


def test_diarize_embed_captures_both_roles() -> None:
    client = _client(
        {_DIARIZER_PORT: _pyannote_ready(), _EMBEDDER_PORT: _titanet_ready()}
    )
    result = observe_stage_model_identity(_settings(), Stage.DIARIZE_EMBED, client=client)
    assert result is not None
    assert set(result) == {"v", "observed_before_attempt", "diarizer", "embedder"}
    # Missing identity fields (no revision / decode_config_hash) record as null,
    # keeping the per-role shape stable across services.
    assert result["diarizer"] == {
        "reachable": True,
        "model": "pyannote/speaker-diarization-3.1",
        "revision": None,
        "engine": "pyannote.audio",
        "decode_config_hash": None,
    }
    assert result["embedder"]["model"] == "titanet-large-v1"
    assert result["embedder"]["reachable"] is True


def test_unreachable_service_records_marker() -> None:
    client = _client({_ASR_PORT: httpx.ConnectError("refused")})
    result = observe_stage_model_identity(_settings(), Stage.TRANSCRIBE, client=client)
    assert result is not None
    assert result["asr"] == {"reachable": False, "detail": "unreachable"}


def test_timeout_records_marker() -> None:
    client = _client({_ASR_PORT: httpx.ReadTimeout("slow")})
    result = observe_stage_model_identity(_settings(), Stage.TRANSCRIBE, client=client)
    assert result is not None
    assert result["asr"] == {"reachable": False, "detail": "timeout"}


def test_degraded_503_records_not_loaded() -> None:
    client = _client({_ASR_PORT: httpx.Response(503, json={"status": "degraded"})})
    result = observe_stage_model_identity(_settings(), Stage.TRANSCRIBE, client=client)
    assert result is not None
    assert result["asr"]["reachable"] is False
    assert "model not loaded" in result["asr"]["detail"]


def test_not_ready_200_records_marker() -> None:
    client = _client(
        {_ASR_PORT: httpx.Response(200, json={"status": "starting", "model_loaded": False})}
    )
    result = observe_stage_model_identity(_settings(), Stage.TRANSCRIBE, client=client)
    assert result is not None
    assert result["asr"] == {"reachable": False, "detail": "not ready"}


def test_malformed_body_records_marker() -> None:
    client = _client({_ASR_PORT: httpx.Response(200, text="not json")})
    result = observe_stage_model_identity(_settings(), Stage.TRANSCRIBE, client=client)
    assert result is not None
    assert result["asr"] == {"reachable": False, "detail": "invalid response"}


def _probe(body: dict[str, object]) -> dict[str, object]:
    client = _client({_DIARIZER_PORT: httpx.Response(200, json=body)})
    return probe_identity_one(client, f"http://localhost:{_DIARIZER_PORT}")


_READY_BASE: dict[str, object] = {
    "status": "ok",
    "service": "pyannote",
    "model": "pyannote/speaker-diarization-3.1",
    "engine": "pyannote.audio",
    "model_loaded": True,
}


def test_identity_hash_fields_carry_present_hex() -> None:
    # #125/#129: both special-cased hash fields carry their reported value.
    payload = _probe(
        {
            **_READY_BASE,
            CHECKPOINT_FINGERPRINT_FIELD: "a" * 64,
            DIARIZATION_CONFIG_HASH_FIELD: "b" * 64,
        }
    )
    assert payload[CHECKPOINT_FINGERPRINT_FIELD] == "a" * 64
    assert payload[DIARIZATION_CONFIG_HASH_FIELD] == "b" * 64


def test_identity_hash_fields_present_but_null_carry_none() -> None:
    # PRESENT-but-null must be preserved as None (fail-closed signal), distinct
    # from the key being absent below — a non-string is normalised to null too.
    payload = _probe(
        {**_READY_BASE, CHECKPOINT_FINGERPRINT_FIELD: None, DIARIZATION_CONFIG_HASH_FIELD: 123}
    )
    assert payload[CHECKPOINT_FINGERPRINT_FIELD] is None
    assert payload[DIARIZATION_CONFIG_HASH_FIELD] is None


def test_identity_hash_fields_absent_are_omitted() -> None:
    # An older service omits the keys entirely; the probe must not invent them, so
    # the console can tell "old service, classify by name" from "new service, null".
    payload = _probe(dict(_READY_BASE))
    assert CHECKPOINT_FINGERPRINT_FIELD not in payload
    assert DIARIZATION_CONFIG_HASH_FIELD not in payload


def test_partial_role_failure_isolated() -> None:
    # One role down must not suppress the other role's recorded identity.
    client = _client(
        {_DIARIZER_PORT: httpx.ConnectError("down"), _EMBEDDER_PORT: _titanet_ready()}
    )
    result = observe_stage_model_identity(_settings(), Stage.DIARIZE_EMBED, client=client)
    assert result is not None
    assert result["diarizer"]["reachable"] is False
    assert result["embedder"]["reachable"] is True
