"""Best-effort readiness probe of the model services (first-run wizard step 5).

The wizard shows the operator, in plain language, whether transcription
(whisper), diarization (pyannote), and speaker embedding (titanet) are reachable,
so a later skipped stage is not a surprise. This is advisory only: a probe
failure NEVER raises into the request, and it never gates a pipeline run.

Each service exposes ``GET /healthz`` (see ``docs/gpu-contracts.md``): ``200`` with
``status="ok"`` when the model is loaded, otherwise ``503`` with
``status="degraded"`` and ``model: null``. We use a short, dedicated timeout — the
inference clients' hours-long read timeout would make the "check services" step
appear to hang — and normalize every outcome (2xx, 503, timeout, transport error,
malformed body) into a small struct with a monotonic round-trip latency.

Voxint's own unauthenticated ``/healthz`` stays pure application liveness; these
downstream results are surfaced through the authenticated setup flow, never folded
into that endpoint.
"""

import time
from dataclasses import dataclass

import httpx

from voxint.config import Settings

# The three services the pipeline calls, in stage order, each paired with a
# human label for the wizard and the ``Settings`` attribute holding its base URL.
_SERVICES: tuple[tuple[str, str], ...] = (
    ("transcription", "asr_url"),
    ("diarization", "diarizer_url"),
    ("speaker embedding", "embedder_url"),
)


@dataclass(frozen=True)
class ServiceHealth:
    """One service's probe outcome, normalized for plain-language display."""

    name: str  # human label, e.g. "transcription"
    url: str  # the base URL probed
    up: bool  # True iff a 2xx readiness response with the model loaded
    detail: str  # stable, plain-language outcome ("ready", "timeout", …)
    latency_ms: float | None  # round-trip for a completed attempt; None if none
    device: str | None = None  # the service's reported compute device, when it says
    embedding_space: str | None = None  # additive titanet voice-vector space id
    # Additive, optional hardware telemetry (docs/gpu-contracts.md ``resources``
    # block). The raw parsed object when the service reports one, else None
    # (tolerated exactly as ``device`` is). Captured on the degraded 503 path
    # too, since telemetry is most useful when a service is struggling.
    resources: dict[str, object] | None = None


def _service_targets(settings: Settings) -> list[tuple[str, str]]:
    """(human label, base URL) for each model service, in stage order."""
    return [(name, getattr(settings, url_attr)) for name, url_attr in _SERVICES]


def probe_services(
    settings: Settings, *, client: httpx.Client | None = None
) -> list[ServiceHealth]:
    """Probe each model service's ``/healthz``. Best-effort; never raises.

    Pass ``client`` to inject a transport in tests (the caller then owns it);
    otherwise a short-timeout client is created and closed here. Probes run
    sequentially — three localhost services under a few-second timeout each is a
    bounded worst case, and the wizard step is not latency-critical. The
    concurrent, cached aggregation for the live resource view lives in
    ``voxint.api.resource_status``.
    """
    own_client = client is None
    probe_client = client or httpx.Client(
        timeout=httpx.Timeout(settings.health_probe_timeout_seconds)
    )
    try:
        return [
            _probe_one(probe_client, name, base_url)
            for name, base_url in _service_targets(settings)
        ]
    finally:
        if own_client:
            probe_client.close()


def _extract_resources(body: object) -> dict[str, object] | None:
    """The additive ``resources`` block, if the body carries one as an object."""
    if isinstance(body, dict):
        resources = body.get("resources")
        if isinstance(resources, dict):
            return resources
    return None


def _read_body(response: httpx.Response) -> object | None:
    try:
        data: object = response.json()
    except ValueError:
        return None
    return data


def _probe_one(client: httpx.Client, name: str, base_url: str) -> ServiceHealth:
    def outcome(
        *,
        up: bool,
        detail: str,
        latency_ms: float | None,
        device: str | None = None,
        embedding_space: str | None = None,
        resources: dict[str, object] | None = None,
    ) -> ServiceHealth:
        return ServiceHealth(
            name=name,
            url=base_url,
            up=up,
            detail=detail,
            latency_ms=latency_ms,
            device=device,
            embedding_space=embedding_space,
            resources=resources,
        )

    url = f"{base_url.rstrip('/')}/healthz"
    start = time.monotonic()
    try:
        response = client.get(url)
    except httpx.TimeoutException:
        # No completed round-trip → no latency to report.
        return outcome(up=False, detail="timeout", latency_ms=None)
    except httpx.InvalidURL:
        # A malformed configured URL is a config error, not a transport failure —
        # and InvalidURL is NOT an httpx.HTTPError, so it must be caught explicitly
        # to keep this probe's "never raises into the request" contract.
        return outcome(up=False, detail="invalid url", latency_ms=None)
    except httpx.HTTPError:
        return outcome(up=False, detail="unreachable", latency_ms=None)
    latency_ms = (time.monotonic() - start) * 1000.0
    if response.status_code == 503:
        # Reachable, but the model is not loaded (the contract's degraded state) —
        # distinct from an unreachable service so the wizard can say so. Parse the
        # body for telemetry: a struggling service is exactly when the operator
        # most wants its resource numbers.
        resources = _extract_resources(_read_body(response))
        return outcome(
            up=False,
            detail="degraded (model not loaded)",
            latency_ms=latency_ms,
            resources=resources,
        )
    if not response.is_success:
        return outcome(up=False, detail=f"HTTP {response.status_code}", latency_ms=latency_ms)
    body = _read_body(response)
    if body is None:
        return outcome(up=False, detail="invalid response", latency_ms=latency_ms)
    # Enforce the documented readiness shape (docs/gpu-contracts.md) rather than
    # trusting any 2xx — some other server answering 200 on that port must not read
    # as "ready". Additive fields are tolerated: only these two keys are inspected.
    if not isinstance(body, dict):
        return outcome(up=False, detail="invalid response", latency_ms=latency_ms)
    resources = _extract_resources(body)
    if body.get("model_loaded") is False:
        return outcome(
            up=False, detail="model not loaded", latency_ms=latency_ms, resources=resources
        )
    if body.get("status") == "ok" and body.get("model_loaded") is True:
        # ``device`` is an additive, optional field (docs/gpu-contracts.md); surface
        # it only when the service reports a string so `doctor` can show cpu/cuda/rocm.
        raw_device = body.get("device")
        device = raw_device if isinstance(raw_device, str) else None
        raw_space = body.get("embedding_space")
        embedding_space = raw_space if isinstance(raw_space, str) and raw_space else None
        return outcome(
            up=True,
            detail="ready",
            latency_ms=latency_ms,
            device=device,
            embedding_space=embedding_space,
            resources=resources,
        )
    # A parseable-but-not-ready 2xx (e.g. status="starting"): keep any telemetry
    # the body carried, exactly as the degraded paths above do.
    return outcome(up=False, detail="not ready", latency_ms=latency_ms, resources=resources)


def probe_embedding_service(
    settings: Settings, *, client: httpx.Client | None = None
) -> ServiceHealth:
    """Probe only titanet, using the same readiness parser as the setup flow."""
    own_client = client is None
    probe_client = client or httpx.Client(
        timeout=httpx.Timeout(settings.health_probe_timeout_seconds)
    )
    try:
        return _probe_one(probe_client, "speaker embedding", settings.embedder_url)
    finally:
        if own_client:
            probe_client.close()
