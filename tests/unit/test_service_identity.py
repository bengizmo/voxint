"""Live-identity collection for the settings "Pipeline models" panel.

Drives ``collect_service_identity`` against an ``httpx.MockTransport`` (no real
services): the validated-default classification per service, the whisper alias,
the unvalidated-override warning, the fixed (non-configurable) embedder, the
unreachable and degraded states, and that a probe-machinery failure degrades to
"unavailable" rather than raising into the settings page.
"""

import httpx
import pytest

from voxint.api.service_identity import (
    _VALIDATED_ASR_REVISION,
    _VALIDATED_DIARIZER_CHECKPOINT,
    ModelVerdict,
    ServiceIdentityView,
    collect_service_identity,
)
from voxint.config import Settings

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


def _ready(service: str, model: str, engine: str, **extra: object) -> httpx.Response:
    body: dict[str, object] = {
        "status": "ok",
        "service": service,
        "model": model,
        "engine": engine,
        "model_loaded": True,
    }
    body.update(extra)
    return httpx.Response(200, json=body)


def _whisper_default() -> httpx.Response:
    # The shipped whisper reports the baked large-v2 snapshot as its revision;
    # that exact revision is what reads as validated (#125).
    return _ready(
        "whisper", "large-v2", "faster-whisper", model_revision=_VALIDATED_ASR_REVISION
    )


def _pyannote_default() -> httpx.Response:
    # The vendored pyannote reports the weight-checkpoint fingerprint of the two
    # baked .bin files; that exact digest is what reads as validated (#125).
    return _ready(
        "pyannote",
        "pyannote/speaker-diarization-3.1",
        "pyannote.audio",
        checkpoint_fingerprint=_VALIDATED_DIARIZER_CHECKPOINT,
    )


def _titanet_default() -> httpx.Response:
    return _ready("titanet", "nvidia/speakerverification_en_titanet_large", "nemo")


def _by_role(views: list[ServiceIdentityView]) -> dict[str, ServiceIdentityView]:
    return {v.role: v for v in views}


def test_all_defaults_classify_validated() -> None:
    client = _client(
        {
            _ASR_PORT: _whisper_default(),
            _DIARIZER_PORT: _pyannote_default(),
            _EMBEDDER_PORT: _titanet_default(),
        }
    )
    views = collect_service_identity(_settings(), client=client)
    # Returned in stage order: transcription, diarization, embedding.
    assert [v.role for v in views] == ["asr", "diarizer", "embedder"]
    by_role = _by_role(views)

    asr = by_role["asr"]
    assert asr.label == "Transcription"
    assert asr.reachable and asr.verdict == ModelVerdict.VALIDATED and asr.configurable
    assert asr.model == "large-v2"
    assert asr.engine == "faster-whisper"
    assert asr.revision == _VALIDATED_ASR_REVISION
    assert asr.detail is None
    assert asr.env_keys == ("WHISPER_MODEL", "WHISPER_REVISION", "WHISPER_ALLOW_DOWNLOAD")

    diarizer = by_role["diarizer"]
    assert (
        diarizer.reachable
        and diarizer.verdict == ModelVerdict.VALIDATED
        and diarizer.configurable
    )
    assert diarizer.model == "pyannote/speaker-diarization-3.1"
    assert diarizer.revision is None
    assert diarizer.env_keys == ("DIARIZER_MODEL_NAME", "DIARIZER_REVISION")

    embedder = by_role["embedder"]
    assert embedder.reachable and embedder.verdict == ModelVerdict.VALIDATED
    assert embedder.configurable is False
    assert embedder.model == "nvidia/speakerverification_en_titanet_large"
    assert embedder.env_keys == ()


def test_whisper_fully_qualified_alias_is_validated() -> None:
    # The fully-qualified repo id names the same baked large-v2 weights and does
    # not pass the download gate, so it must not read as an unvalidated override.
    client = _client(
        {
            _ASR_PORT: _ready(
                "whisper",
                "Systran/faster-whisper-large-v2",
                "faster-whisper",
                model_revision=_VALIDATED_ASR_REVISION,
            ),
            _DIARIZER_PORT: _pyannote_default(),
            _EMBEDDER_PORT: _titanet_default(),
        }
    )
    asr = _by_role(collect_service_identity(_settings(), client=client))["asr"]
    assert asr.reachable and asr.verdict == ModelVerdict.VALIDATED


def test_whisper_override_is_unvalidated() -> None:
    client = _client(
        {
            _ASR_PORT: _ready("whisper", "large-v3", "faster-whisper"),
            _DIARIZER_PORT: _pyannote_default(),
            _EMBEDDER_PORT: _titanet_default(),
        }
    )
    asr = _by_role(collect_service_identity(_settings(), client=client))["asr"]
    assert asr.reachable is True
    assert asr.configurable is True
    assert asr.verdict == ModelVerdict.UNVALIDATED
    assert asr.model == "large-v3"
    assert asr.detail is None  # reachable: an override, not an unavailability


def test_whisper_validated_name_with_wrong_revision_is_mismatch() -> None:
    # #125: a validated NAME (large-v2) reporting a revision other than the baked
    # snapshot loaded different weights under the validated name — fail closed.
    client = _client(
        {
            _ASR_PORT: _ready(
                "whisper", "large-v2", "faster-whisper", model_revision="a" * 40
            ),
            _DIARIZER_PORT: _pyannote_default(),
            _EMBEDDER_PORT: _titanet_default(),
        }
    )
    asr = _by_role(collect_service_identity(_settings(), client=client))["asr"]
    assert asr.reachable is True
    assert asr.verdict == ModelVerdict.MISMATCH


def test_whisper_validated_name_without_revision_is_unverified() -> None:
    # A validated name reporting no revision cannot be confirmed as the baked
    # weights — fail closed to unverified, not validated.
    client = _client(
        {
            _ASR_PORT: _ready("whisper", "large-v2", "faster-whisper"),
            _DIARIZER_PORT: _pyannote_default(),
            _EMBEDDER_PORT: _titanet_default(),
        }
    )
    asr = _by_role(collect_service_identity(_settings(), client=client))["asr"]
    assert asr.reachable is True
    assert asr.verdict == ModelVerdict.UNVERIFIED


def test_diarizer_override_is_unvalidated() -> None:
    client = _client(
        {
            _ASR_PORT: _whisper_default(),
            _DIARIZER_PORT: _ready(
                "pyannote", "pyannote/speaker-diarization-4.0", "pyannote.audio"
            ),
            _EMBEDDER_PORT: _titanet_default(),
        }
    )
    diarizer = _by_role(collect_service_identity(_settings(), client=client))["diarizer"]
    assert diarizer.reachable is True
    assert diarizer.verdict == ModelVerdict.UNVALIDATED


def test_diarizer_validated_name_with_wrong_fingerprint_is_mismatch() -> None:
    # #125: the vendored name with a checkpoint fingerprint that is not the
    # validated one means swapped/re-fetched weights — fail closed (red).
    client = _client(
        {
            _ASR_PORT: _whisper_default(),
            _DIARIZER_PORT: _ready(
                "pyannote",
                "pyannote/speaker-diarization-3.1",
                "pyannote.audio",
                checkpoint_fingerprint="0" * 64,
            ),
            _EMBEDDER_PORT: _titanet_default(),
        }
    )
    diarizer = _by_role(collect_service_identity(_settings(), client=client))["diarizer"]
    assert diarizer.reachable is True
    assert diarizer.verdict == ModelVerdict.MISMATCH


def test_diarizer_validated_name_with_null_fingerprint_is_unverified() -> None:
    # A validated name reporting a null fingerprint (an HF source whose files are
    # not hashed) cannot be verified — fail closed to unverified (amber).
    client = _client(
        {
            _ASR_PORT: _whisper_default(),
            _DIARIZER_PORT: _ready(
                "pyannote",
                "pyannote/speaker-diarization-3.1",
                "pyannote.audio",
                checkpoint_fingerprint=None,
            ),
            _EMBEDDER_PORT: _titanet_default(),
        }
    )
    diarizer = _by_role(collect_service_identity(_settings(), client=client))["diarizer"]
    assert diarizer.reachable is True
    assert diarizer.verdict == ModelVerdict.UNVERIFIED


def test_diarizer_validated_name_without_fingerprint_field_is_validated() -> None:
    # Rollout compatibility: an older pyannote that predates the fingerprint field
    # omits the key entirely. Absent (not null) means no signal to fail on, so the
    # validated name is trusted as before — default installs do not flip to amber
    # mid-upgrade.
    client = _client(
        {
            _ASR_PORT: _whisper_default(),
            _DIARIZER_PORT: _ready(
                "pyannote", "pyannote/speaker-diarization-3.1", "pyannote.audio"
            ),
            _EMBEDDER_PORT: _titanet_default(),
        }
    )
    diarizer = _by_role(collect_service_identity(_settings(), client=client))["diarizer"]
    assert diarizer.reachable is True
    assert diarizer.verdict == ModelVerdict.VALIDATED


def test_embedder_never_unvalidated_even_with_unexpected_model() -> None:
    # titanet is a DB invariant, not operator-configurable, so an unexpected model
    # is still shown but never carries an unvalidated warning.
    client = _client(
        {
            _ASR_PORT: _whisper_default(),
            _DIARIZER_PORT: _pyannote_default(),
            _EMBEDDER_PORT: _ready("titanet", "something-else", "nemo"),
        }
    )
    embedder = _by_role(collect_service_identity(_settings(), client=client))["embedder"]
    assert embedder.reachable is True
    assert embedder.configurable is False
    assert embedder.verdict == ModelVerdict.VALIDATED
    assert embedder.model == "something-else"


def test_reachable_without_model_name_is_unvalidated_for_configurable() -> None:
    # A reachable service that reports no model name cannot be confirmed validated.
    body_no_model = httpx.Response(
        200,
        json={
            "status": "ok",
            "service": "whisper",
            "engine": "faster-whisper",
            "model_loaded": True,
        },
    )
    client = _client(
        {
            _ASR_PORT: body_no_model,
            _DIARIZER_PORT: _pyannote_default(),
            _EMBEDDER_PORT: _titanet_default(),
        }
    )
    asr = _by_role(collect_service_identity(_settings(), client=client))["asr"]
    assert asr.reachable is True
    assert asr.model is None
    assert asr.verdict == ModelVerdict.UNVALIDATED


def test_unreachable_service_records_unavailable() -> None:
    client = _client(
        {
            _ASR_PORT: httpx.ConnectError("refused"),
            _DIARIZER_PORT: _pyannote_default(),
            _EMBEDDER_PORT: _titanet_default(),
        }
    )
    asr = _by_role(collect_service_identity(_settings(), client=client))["asr"]
    assert asr.reachable is False
    assert asr.model is None and asr.engine is None and asr.revision is None
    assert asr.detail == "unreachable"
    assert asr.url == _settings().asr_url  # the probed address is surfaced


def test_degraded_503_records_detail() -> None:
    client = _client(
        {
            _ASR_PORT: httpx.Response(503, json={"status": "degraded", "model_loaded": False}),
            _DIARIZER_PORT: _pyannote_default(),
            _EMBEDDER_PORT: _titanet_default(),
        }
    )
    asr = _by_role(collect_service_identity(_settings(), client=client))["asr"]
    assert asr.reachable is False
    assert asr.detail == "degraded (model not loaded)"


def test_probe_machinery_failure_degrades_all(monkeypatch: pytest.MonkeyPatch) -> None:
    # A non-httpx failure escaping the per-service probe must never break Settings;
    # every service degrades to an "unavailable" record instead.
    import voxint.api.service_identity as si

    def _boom(_client: httpx.Client, _url: str) -> dict[str, object]:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(si, "probe_identity_one", _boom)
    client = _client({})  # handler never reached
    views = collect_service_identity(_settings(), client=client)
    assert len(views) == 3
    assert all(not v.reachable for v in views)
    assert all(v.detail == "probe failed" for v in views)


def test_default_client_is_constructed_and_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    # With no injected client, a short-timeout client is built here and closed;
    # patch the constructor to a MockTransport so no real network call happens.
    mock = _client(
        {
            _ASR_PORT: _whisper_default(),
            _DIARIZER_PORT: _pyannote_default(),
            _EMBEDDER_PORT: _titanet_default(),
        }
    )
    closed: list[bool] = []
    original_close = mock.close

    def _tracked_close() -> None:
        closed.append(True)
        original_close()

    monkeypatch.setattr(mock, "close", _tracked_close)
    monkeypatch.setattr("voxint.api.service_identity.httpx.Client", lambda **_kwargs: mock)
    views = collect_service_identity(_settings())
    assert [v.role for v in views] == ["asr", "diarizer", "embedder"]
    assert closed == [True]  # the self-owned client was closed
