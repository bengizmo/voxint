"""Per-attempt model-identity observation, stamped onto each StageRun.

Every stage that calls a model service records, on its own ``StageRun`` attempt,
which model actually answered — read from the service's ``/healthz`` identity
fields immediately before the stage body runs. This is provenance, not a gate:

- **Best-effort, never blocking a run.** An unreachable, slow, or not-yet-loaded
  service records a ``reachable: false`` marker and the stage proceeds unchanged.
  A dedicated short timeout is used (the inference clients' hours-long read
  timeout would make this hang), and the two DIARIZE_EMBED services are probed
  concurrently, so the added latency is bounded, not additive.
- **Per-attempt, so it is attempt-safe.** The observation lives on the StageRun
  row, which *is* the execution claim, written in the same transaction that
  completes the claim. A failed or lease-expired attempt can never overwrite a
  later successful attempt's recorded identity; run detail renders the latest
  completed attempt per stage.
- **Observed before the attempt, not response-carried.** The ``/v1`` wire
  contracts do not echo model identity, so this is the identity the service
  reported just before the call, not proof of the exact build that produced the
  output. The UI copy says so (``observed_before_attempt``).

The shape written under ``StageRun.metrics["model_identity"]`` is::

    {"v": 1, "observed_before_attempt": true,
     "asr": {"reachable": true, "model": "large-v2", "revision": "<sha|null>",
             "engine": "ct2-legacy", "decode_config_hash": "<hex|null>"},
     ...}

with one entry per role the stage exercises. An unreachable role is
``{"reachable": false, "detail": "timeout"}``. A role whose service reports a
weight-checkpoint fingerprint (pyannote, #125) additionally carries
``"checkpoint_fingerprint": "<hex|null>"``; the key is omitted entirely for a
service that does not report it.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from voxint.config import Settings
from voxint.db.models import Stage

METRICS_KEY = "model_identity"
IDENTITY_SCHEMA_VERSION = 1

# The pyannote weight-checkpoint fingerprint (#125): a digest over the actually-
# loaded ``.bin`` checkpoint files, so a service reporting the validated NAME can
# be checked against the validated WEIGHTS. Handled outside ``_IDENTITY_FIELDS``
# because, unlike the always-present string fields, the console must distinguish
# the key being ABSENT (an older service that predates the field — classify by
# name as before) from PRESENT-but-null (a new service on an unverifiable source
# — fail closed). The other fields collapse both to null; this one must not.
CHECKPOINT_FINGERPRINT_FIELD = "checkpoint_fingerprint"

# Which model services each stage exercises, as (role, Settings URL attribute)
# pairs in call order. A stage absent from this map calls no model service and is
# never stamped. ``embedder`` is recorded for DIARIZE_EMBED for completeness even
# though the embedder is not operator-configurable (titanet is a DB invariant).
_STAGE_MODEL_SERVICES: dict[Stage, tuple[tuple[str, str], ...]] = {
    Stage.TRANSCRIBE: (("asr", "asr_url"),),
    Stage.DIARIZE_EMBED: (("diarizer", "diarizer_url"), ("embedder", "embedder_url")),
}

# The ``/healthz`` identity fields captured, mapped to our stable output keys.
# Absent fields (e.g. pyannote/titanet carry no decode_config_hash) record null,
# keeping the per-role shape stable across services.
_IDENTITY_FIELDS: tuple[tuple[str, str], ...] = (
    ("model", "model"),
    ("revision", "model_revision"),
    ("engine", "engine"),
    ("decode_config_hash", "decode_config_hash"),
)


def stage_has_model_identity(stage: Stage) -> bool:
    """True if this stage calls a model service whose identity we stamp."""
    return stage in _STAGE_MODEL_SERVICES


def probe_identity_one(client: httpx.Client, base_url: str) -> dict[str, Any]:
    """Read one service's ``/healthz`` identity. Never raises.

    Returns a role payload: ``reachable: true`` with the identity fields on a
    ready ``200``; otherwise ``reachable: false`` with a plain-language detail.
    """
    url = f"{base_url.rstrip('/')}/healthz"
    try:
        response = client.get(url)
    except httpx.TimeoutException:
        return {"reachable": False, "detail": "timeout"}
    except httpx.InvalidURL:
        return {"reachable": False, "detail": "invalid url"}
    except httpx.HTTPError:
        return {"reachable": False, "detail": "unreachable"}

    if response.status_code == 503:
        return {"reachable": False, "detail": "degraded (model not loaded)"}
    if not response.is_success:
        return {"reachable": False, "detail": f"HTTP {response.status_code}"}
    try:
        body: object = response.json()
    except ValueError:
        return {"reachable": False, "detail": "invalid response"}
    if not isinstance(body, dict):
        return {"reachable": False, "detail": "invalid response"}
    if body.get("model_loaded") is not True or body.get("status") != "ok":
        # A parseable-but-not-ready 200 (e.g. status="starting"): reachable, but no
        # trustworthy identity to record yet.
        return {"reachable": False, "detail": "not ready"}

    payload: dict[str, Any] = {"reachable": True}
    for out_key, health_key in _IDENTITY_FIELDS:
        value = body.get(health_key)
        payload[out_key] = value if isinstance(value, str) else None
    # Only carry the checkpoint fingerprint when the service actually reports the
    # key, so a consumer can tell "old service, field absent" from "new service,
    # value null". Present-but-non-string is normalised to null (unverifiable).
    if CHECKPOINT_FINGERPRINT_FIELD in body:
        raw = body[CHECKPOINT_FINGERPRINT_FIELD]
        payload[CHECKPOINT_FINGERPRINT_FIELD] = raw if isinstance(raw, str) else None
    return payload


def observe_stage_model_identity(
    settings: Settings, stage: Stage, *, client: httpx.Client | None = None
) -> dict[str, Any] | None:
    """Observe the model identity for ``stage``'s services. Never raises.

    Returns ``None`` for a stage that calls no model service (nothing to stamp).
    Otherwise returns the ``model_identity`` object to store on the attempt's
    ``StageRun.metrics``. Roles are probed concurrently under a short timeout;
    ``client`` injects a transport in tests (the caller then owns it).
    """
    targets = _STAGE_MODEL_SERVICES.get(stage)
    if not targets:
        return None

    own_client = client is None
    probe_client = client or httpx.Client(
        timeout=httpx.Timeout(settings.health_probe_timeout_seconds)
    )
    try:
        roles = [role for role, _ in targets]
        urls = [getattr(settings, url_attr) for _, url_attr in targets]
        with ThreadPoolExecutor(max_workers=max(1, len(targets))) as pool:
            payloads = list(pool.map(lambda u: probe_identity_one(probe_client, u), urls))
    except Exception:
        # Defence in depth: the probe is advisory, so any unexpected failure of the
        # probe machinery itself must never propagate into the stage.
        return {"v": IDENTITY_SCHEMA_VERSION, "observed_before_attempt": True}
    finally:
        if own_client:
            probe_client.close()

    result: dict[str, Any] = {"v": IDENTITY_SCHEMA_VERSION, "observed_before_attempt": True}
    for role, payload in zip(roles, payloads, strict=True):
        result[role] = payload
    return result
