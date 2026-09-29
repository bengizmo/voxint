"""Which model-service identities the real-pipeline lane expects (``VOXINT_E2E_LANE``).

The lane asserts each service's ``/healthz`` identity so a silent device
fallback fails the gate instead of passing it. Which devices are correct
depends on the compose overlay the services run under, so the operator names
it; there is no default, because guessing would let a fallback through.

Kept free of app imports so ``tests/unit`` can cover the resolver without the
E2E prerequisites.
"""

from __future__ import annotations

# Lane -> /healthz ``device`` for (whisper, pyannote, titanet), matching the
# shipped overlays: compose.gpu.yaml, compose.rocm.yaml, compose.cpu.yaml.
LANE_DEVICES: dict[str, tuple[str, str, str]] = {
    "cuda": ("cuda", "cuda", "cuda"),
    "rocm": ("rocm", "cpu", "cpu"),
    "cpu": ("cpu", "cpu", "cpu"),
}


def expected_services(lane: str | None) -> dict[str, dict[str, str]]:
    """The identity each service must report at ``/healthz`` on ``lane``.

    Keyed by the ``Settings`` URL field that addresses the service. Raises
    ``ValueError`` naming the valid lanes when ``lane`` is unset, empty or
    unknown; the caller turns that into a test failure, never a skip.
    """
    key = (lane or "").strip().lower()
    if key not in LANE_DEVICES:
        valid = ", ".join(LANE_DEVICES)
        raise ValueError(
            f"VOXINT_E2E_LANE must name the compose overlay the model services run "
            f"under, one of: {valid} (got {lane!r})."
        )
    whisper, pyannote, titanet = LANE_DEVICES[key]
    return {
        "asr_url": {"service": "whisper", "device": whisper, "model": "large-v2"},
        "diarizer_url": {
            "service": "pyannote",
            "device": pyannote,
            "model": "pyannote/speaker-diarization-3.1",
        },
        "embedder_url": {"service": "titanet", "device": titanet},
    }
