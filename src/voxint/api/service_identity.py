"""Live model identity of the pipeline services, for the settings panel.

The Settings > "Pipeline models" panel shows the operator which transcription,
diarization, and speaker-embedding model each service is running right now, read
from its ``/healthz`` identity, and whether that identity is the *validated*
default or an unvalidated override. Model selection is deployment-owned
(env/compose/installer), so this panel is read-only: it reports what the running
containers say and points at the ``.env`` keys that change it, never mutating
anything itself.

This is provenance display, not a gate. Every service is probed live on each
render, concurrently, under a short dedicated timeout, and any probe failure
records an "unavailable" state rather than raising: an unreachable service must
never break the settings page. There is deliberately **no cache** here (unlike
``voxint.api.resource_status``): an operator who just changed ``.env`` and
restarted a container reloads this page to confirm the new identity, and a stale
cache would lie to them.

"Validated" means the reported identity matches a validated default, not proof
that no override env var was set (an override can point at the same model id), so
the panel copy says "validated identity". Only ``large-v2`` and pyannote
``speaker-diarization-3.1`` are validated; overrides are a supported mechanism
with unvalidated numerics (v3/turbo hallucinate). titanet is a database invariant
(its embedding space is fixed), so the speaker-embedding model is never
operator-configurable and never carries an "unvalidated" warning.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import httpx

from voxint.config import Settings
from voxint.pipeline.model_identity import probe_identity_one

# Validated transcription model ids. Mirrors
# ``services/whisper/app/whisper_startup.py::DEFAULT_MODELS`` (kept in sync by
# hand; that container's code is not importable from the console). Both name the
# baked large-v2 weights — the short id the default reports and its
# fully-qualified repo id — and neither passes the whisper download gate.
_VALIDATED_ASR_MODELS = frozenset({"large-v2", "Systran/faster-whisper-large-v2"})

# The validated diarization pipeline id. The vendored default reports this
# canonical id on ``/healthz``; pyannote 4.x and other pipelines are unvalidated.
_VALIDATED_DIARIZER_MODELS = frozenset({"pyannote/speaker-diarization-3.1"})


@dataclass(frozen=True)
class _ServiceSpec:
    """Static description of one probed service: how to reach it, whether the
    operator can change its model, and which model ids count as validated."""

    role: str  # identity key, matches the /healthz probe roles
    label: str  # operator-facing service name
    url_attr: str  # Settings attribute holding the base URL
    configurable: bool  # False => a fixed model (titanet DB invariant)
    validated_models: frozenset[str]  # ids that read as the validated default
    env_keys: tuple[str, ...]  # the .env keys that change this service's model


# The three services, in stage order. Only ASR and diarization are operator-
# configurable; the embedder (titanet) is a DB invariant with no override path,
# so it carries no validated-model set and no env keys.
_SERVICE_SPECS: tuple[_ServiceSpec, ...] = (
    _ServiceSpec(
        role="asr",
        label="Transcription",
        url_attr="asr_url",
        configurable=True,
        validated_models=_VALIDATED_ASR_MODELS,
        env_keys=("WHISPER_MODEL", "WHISPER_REVISION", "WHISPER_ALLOW_DOWNLOAD"),
    ),
    _ServiceSpec(
        role="diarizer",
        label="Speaker diarization",
        url_attr="diarizer_url",
        configurable=True,
        validated_models=_VALIDATED_DIARIZER_MODELS,
        env_keys=("DIARIZER_MODEL_NAME", "DIARIZER_REVISION"),
    ),
    _ServiceSpec(
        role="embedder",
        label="Speaker embedding",
        url_attr="embedder_url",
        configurable=False,
        validated_models=frozenset(),
        env_keys=(),
    ),
)


@dataclass(frozen=True)
class ServiceIdentityView:
    """One service's live identity, classified for the settings panel.

    ``reachable`` False means the probe could not read a trustworthy identity
    (down, still loading, or a bad URL); ``detail`` carries the plain-language
    reason and the model fields are ``None``. ``validated`` is only meaningful
    for a ``configurable`` service: it is the reachable-and-matches-a-validated-
    default verdict that drives the unvalidated-override warning. A
    non-configurable service (the embedder) is a fixed model and shows its raw
    identity without a validated/unvalidated badge.
    """

    role: str
    label: str
    url: str
    reachable: bool
    model: str | None
    revision: str | None
    engine: str | None
    configurable: bool
    validated: bool
    detail: str | None
    env_keys: tuple[str, ...]


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _classify(spec: _ServiceSpec, url: str, payload: dict[str, Any]) -> ServiceIdentityView:
    """Turn one probe payload into a classified display record.

    An unreachable payload becomes an "unavailable" record (never a false
    unvalidated warning); a reachable one is validated when the service is
    non-configurable (fixed model) or its reported model id is in the validated
    set.
    """
    if payload.get("reachable") is not True:
        detail = _str_or_none(payload.get("detail")) or "unavailable"
        return ServiceIdentityView(
            role=spec.role,
            label=spec.label,
            url=url,
            reachable=False,
            model=None,
            revision=None,
            engine=None,
            configurable=spec.configurable,
            validated=False,
            detail=detail,
            env_keys=spec.env_keys,
        )
    model = _str_or_none(payload.get("model"))
    validated = (not spec.configurable) or (
        model is not None and model in spec.validated_models
    )
    return ServiceIdentityView(
        role=spec.role,
        label=spec.label,
        url=url,
        reachable=True,
        model=model,
        revision=_str_or_none(payload.get("revision")),
        engine=_str_or_none(payload.get("engine")),
        configurable=spec.configurable,
        validated=validated,
        detail=None,
        env_keys=spec.env_keys,
    )


def collect_service_identity(
    settings: Settings, *, client: httpx.Client | None = None
) -> list[ServiceIdentityView]:
    """Probe every model service's live identity, classified for the panel.

    Best-effort and never raises: the three services are probed concurrently
    under the health-probe timeout, and any failure of a probe (or of the probe
    machinery itself) records an "unavailable" record so the settings page always
    renders. ``client`` injects a transport in tests (the caller then owns it).
    """
    urls = [getattr(settings, spec.url_attr) for spec in _SERVICE_SPECS]
    own_client = client is None
    probe_client = client or httpx.Client(
        timeout=httpx.Timeout(settings.health_probe_timeout_seconds)
    )
    try:
        with ThreadPoolExecutor(max_workers=max(1, len(_SERVICE_SPECS))) as pool:
            payloads = list(pool.map(lambda u: probe_identity_one(probe_client, u), urls))
    except Exception:
        # Defence in depth: the panel is advisory, so any unexpected failure of
        # the probe machinery records "unavailable" rather than breaking Settings.
        payloads = [{"reachable": False, "detail": "probe failed"} for _ in _SERVICE_SPECS]
    finally:
        if own_client:
            probe_client.close()
    return [
        _classify(spec, url, payload)
        for spec, url, payload in zip(_SERVICE_SPECS, urls, payloads, strict=True)
    ]
