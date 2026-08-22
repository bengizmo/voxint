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
from enum import StrEnum
from typing import Any

import httpx

from voxint.config import Settings
from voxint.pipeline.model_identity import (
    CHECKPOINT_FINGERPRINT_FIELD,
    probe_identity_one,
)

# Validated transcription model ids. Mirrors
# ``services/whisper/app/whisper_startup.py::DEFAULT_MODELS`` (kept in sync by
# hand; that container's code is not importable from the console). Both name the
# baked large-v2 weights — the short id the default reports and its
# fully-qualified repo id — and neither passes the whisper download gate.
_VALIDATED_ASR_MODELS = frozenset({"large-v2", "Systran/faster-whisper-large-v2"})

# The validated diarization pipeline id. The vendored default reports this
# canonical id on ``/healthz``; pyannote 4.x and other pipelines are unvalidated.
_VALIDATED_DIARIZER_MODELS = frozenset({"pyannote/speaker-diarization-3.1"})

# Exact-identity anchors (#125): a validated NAME is not proof of validated
# WEIGHTS. These pin the exact identity each configurable service must also
# report to read as validated, so a re-fetched or swapped build under the
# validated name fails closed instead of showing green.
#
# The baked large-v2 snapshot: mirrors the whisper Dockerfiles'
# ``WHISPER_HF_REVISION`` ARG (contract-tested to match). An overridden whisper
# revision under the validated name reads as a weights mismatch.
_VALIDATED_ASR_REVISION = "f0fe81560cb8b68660e564f55dd99207059c092e"

# The vendored pyannote checkpoint fingerprint: sha256 over the two loaded
# ``.bin`` files, composed as documented in ``docs/gpu-contracts.md``. Derived
# from ``services/pyannote/models/provenance.json`` and contract-tested to match,
# so a weights refresh to ``pyannote-models-v2`` moves both together.
_VALIDATED_DIARIZER_CHECKPOINT = (
    "aa94a2d96a8f1eb5eb8fb80b863c6616417ff1e5c9a8dab91ce42914f836a0d2"
)


class ModelVerdict(StrEnum):
    """How a reachable configurable service's live identity compares to the
    validated one. ``VALIDATED`` is the reported name matching a validated id
    *and* the exact identity (weights) matching; ``MISMATCH`` is a validated
    name whose weights demonstrably differ (fail closed, the tampered/wrong-build
    case); ``UNVERIFIED`` is a validated name whose weights cannot be verified on
    this deployment (fail closed, e.g. an online/HF source); ``UNVALIDATED`` is a
    different model id (opted out). A non-configurable service is always
    ``VALIDATED`` — the template shows its fixed-model copy instead."""

    VALIDATED = "validated"
    MISMATCH = "mismatch"
    UNVERIFIED = "unverified"
    UNVALIDATED = "unvalidated"


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
    # Exact-identity anchor for a validated NAME (#125). At most one is set.
    # ``validated_checkpoint``: the pyannote weight fingerprint, compared against
    # the ``checkpoint_fingerprint`` the service reports (absent => an older
    # service, classify by name only). ``validated_revision``: the whisper baked
    # snapshot, compared against the reported ``revision``.
    validated_checkpoint: str | None
    validated_revision: str | None


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
        validated_checkpoint=None,
        validated_revision=_VALIDATED_ASR_REVISION,
    ),
    _ServiceSpec(
        role="diarizer",
        label="Speaker diarization",
        url_attr="diarizer_url",
        configurable=True,
        validated_models=_VALIDATED_DIARIZER_MODELS,
        env_keys=("DIARIZER_MODEL_NAME", "DIARIZER_REVISION"),
        validated_checkpoint=_VALIDATED_DIARIZER_CHECKPOINT,
        validated_revision=None,
    ),
    _ServiceSpec(
        role="embedder",
        label="Speaker embedding",
        url_attr="embedder_url",
        configurable=False,
        validated_models=frozenset(),
        env_keys=(),
        validated_checkpoint=None,
        validated_revision=None,
    ),
)


@dataclass(frozen=True)
class ServiceIdentityView:
    """One service's live identity, classified for the settings panel.

    ``reachable`` False means the probe could not read a trustworthy identity
    (down, still loading, or a bad URL); ``detail`` carries the plain-language
    reason and the model fields are ``None``. ``verdict`` is only meaningful for
    a reachable ``configurable`` service: it drives the panel's validated /
    weights-mismatch / unverified / unvalidated states (see ``ModelVerdict``). A
    non-configurable service (the embedder) is a fixed model and shows its raw
    identity without a badge; its ``verdict`` is ``VALIDATED`` but the template
    renders the fixed-model copy instead.
    """

    role: str
    label: str
    url: str
    reachable: bool
    model: str | None
    revision: str | None
    engine: str | None
    configurable: bool
    verdict: ModelVerdict
    detail: str | None
    env_keys: tuple[str, ...]


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _exact_identity_verdict(
    spec: _ServiceSpec, payload: dict[str, Any], revision: str | None
) -> ModelVerdict:
    """Classify a service whose reported NAME is a validated one by its exact
    identity (#125). A validated name is necessary but not sufficient: the
    weights must also match, or the panel fails closed.

    Uses whichever exactness anchor the spec carries. Pyannote reports a weight
    ``checkpoint_fingerprint`` whose key is absent on an older service (classify
    by name only, ``VALIDATED``), null when the source is unverifiable
    (``UNVERIFIED``), or a hex string to compare (``VALIDATED`` / ``MISMATCH``).
    Whisper compares the reported ``revision`` against the baked snapshot: a
    different revision under the validated name is a weights ``MISMATCH``; a
    missing one is ``UNVERIFIED``.
    """
    if spec.validated_checkpoint is not None:
        if CHECKPOINT_FINGERPRINT_FIELD not in payload:
            # Older service that predates the fingerprint: no signal to fail on,
            # so trust the validated name exactly as before.
            return ModelVerdict.VALIDATED
        reported = _str_or_none(payload.get(CHECKPOINT_FINGERPRINT_FIELD))
        if reported is None:
            return ModelVerdict.UNVERIFIED
        return (
            ModelVerdict.VALIDATED
            if reported == spec.validated_checkpoint
            else ModelVerdict.MISMATCH
        )
    if spec.validated_revision is not None:
        if revision is None:
            return ModelVerdict.UNVERIFIED
        return (
            ModelVerdict.VALIDATED
            if revision == spec.validated_revision
            else ModelVerdict.MISMATCH
        )
    # No exactness anchor for this service: the validated name is the verdict.
    return ModelVerdict.VALIDATED


def _classify(spec: _ServiceSpec, url: str, payload: dict[str, Any]) -> ServiceIdentityView:
    """Turn one probe payload into a classified display record.

    An unreachable payload becomes an "unavailable" record (never a false
    unvalidated warning). A reachable non-configurable service is always
    ``VALIDATED`` (fixed model). A reachable configurable service is
    ``UNVALIDATED`` when its reported id is not a validated one, otherwise its
    exact identity decides validated / mismatch / unverified.
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
            verdict=ModelVerdict.UNVALIDATED,
            detail=detail,
            env_keys=spec.env_keys,
        )
    model = _str_or_none(payload.get("model"))
    revision = _str_or_none(payload.get("revision"))
    if not spec.configurable:
        verdict = ModelVerdict.VALIDATED
    elif model is None or model not in spec.validated_models:
        verdict = ModelVerdict.UNVALIDATED
    else:
        verdict = _exact_identity_verdict(spec, payload, revision)
    return ServiceIdentityView(
        role=spec.role,
        label=spec.label,
        url=url,
        reachable=True,
        model=model,
        revision=revision,
        engine=_str_or_none(payload.get("engine")),
        configurable=spec.configurable,
        verdict=verdict,
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
