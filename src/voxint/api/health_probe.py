"""Best-effort readiness probe of the GPU model services (first-run wizard step 5).

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


def probe_services(
    settings: Settings, *, client: httpx.Client | None = None
) -> list[ServiceHealth]:
    """Probe each GPU service's ``/healthz``. Best-effort; never raises.

    Pass ``client`` to inject a transport in tests (the caller then owns it);
    otherwise a short-timeout client is created and closed here. Probes run
    sequentially — three localhost services under a few-second timeout each is a
    bounded worst case, and the wizard step is not latency-critical.
    """
    own_client = client is None
    probe_client = client or httpx.Client(
        timeout=httpx.Timeout(settings.health_probe_timeout_seconds)
    )
    try:
        return [
            _probe_one(probe_client, name, getattr(settings, url_attr))
            for name, url_attr in _SERVICES
        ]
    finally:
        if own_client:
            probe_client.close()


def _probe_one(client: httpx.Client, name: str, base_url: str) -> ServiceHealth:
    def outcome(*, up: bool, detail: str, latency_ms: float | None) -> ServiceHealth:
        return ServiceHealth(name=name, url=base_url, up=up, detail=detail, latency_ms=latency_ms)

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
        # distinct from an unreachable service so the wizard can say so.
        return outcome(up=False, detail="degraded (model not loaded)", latency_ms=latency_ms)
    if not response.is_success:
        return outcome(up=False, detail=f"HTTP {response.status_code}", latency_ms=latency_ms)
    try:
        body = response.json()
    except ValueError:
        return outcome(up=False, detail="invalid response", latency_ms=latency_ms)
    # Enforce the documented readiness shape (docs/gpu-contracts.md) rather than
    # trusting any 2xx — some other server answering 200 on that port must not read
    # as "ready". Additive fields are tolerated: only these two keys are inspected.
    if not isinstance(body, dict):
        return outcome(up=False, detail="invalid response", latency_ms=latency_ms)
    if body.get("model_loaded") is False:
        return outcome(up=False, detail="model not loaded", latency_ms=latency_ms)
    if body.get("status") == "ok" and body.get("model_loaded") is True:
        return outcome(up=True, detail="ready", latency_ms=latency_ms)
    return outcome(up=False, detail="not ready", latency_ms=latency_ms)
