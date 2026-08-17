"""Shared front-layer VAD plan: reuse faster-whisper 1.2.1's own VAD + packing.

The shared-windows front (the ``ct2`` engine, Slice 2b of #33) owns VAD,
packing, and packed->source time restoration so that every window-decoding
backend — ``ct2`` now, an mlx backend later — consumes *identical* packed
windows. This module builds that plan by reusing faster-whisper's OWN
primitives (``get_speech_timestamps`` + ``collect_chunks``) with the EXACT
parameters ``BatchedInferencePipeline.transcribe`` uses internally, so the
packed windows are byte-identical to what the shipped ``ct2-legacy`` path
computes inside the pipeline. Restoration is delegated to faster-whisper's own
``restore_speech_timestamps`` against the *global* speech-chunk list (never a
hand-rolled per-window offset map, which gets boundary words wrong).

Numerics contract (pinned to faster-whisper 1.2.1 — see
``BatchedInferencePipeline.transcribe`` lines 394-424 and ``vad.py``):

* ``VadOptions`` is constructed EXPLICITLY. The bare ``VadOptions()`` default
  has ``min_silence_duration_ms=2000`` and ``max_speech_duration_s=inf``; the
  pipeline overrides those to ``160`` and ``chunk_length`` (30 s). Every other
  field keeps its ``VadOptions`` default (``threshold=0.5``,
  ``neg_threshold=None``, ``min_speech_duration_ms=0``, ``speech_pad_ms=400``).
* ``collect_chunks`` is called with ``max_duration=CHUNK_LENGTH_S`` (30), the
  ``feature_extractor.chunk_length``, NOT the ``vad.py`` default of ``inf`` —
  otherwise multi-interval speech would not split at the 30 s boundary the
  decoder expects.
* ``duration_seconds`` is the ORIGINAL file duration, never the packed /
  post-VAD length.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.backends.ct2 import SpeechWindow

# The whisper feature extractor's chunk length (30 s @ 16 kHz). The
# BatchedInferencePipeline passes this as BOTH VadOptions.max_speech_duration_s
# and collect_chunks(max_duration=...). Pinned to faster-whisper 1.2.1.
CHUNK_LENGTH_S = 30
SAMPLING_RATE = 16000

# Bumped whenever the VAD parameters or packing behavior change in a way that
# would move the packed windows; surfaced on /healthz so a decode_config_hash
# consumer can tell two deployments apart.
VAD_PLAN_VERSION = "fw-1.2.1-batched-v1"

# The effective VAD parameters, as a plain dict for /healthz and hashing. Keep
# in lockstep with build_vad_plan's VadOptions construction below.
VAD_PARAMS: dict[str, Any] = {
    "threshold": 0.5,
    "neg_threshold": None,
    "min_speech_duration_ms": 0,
    "max_speech_duration_s": CHUNK_LENGTH_S,
    "min_silence_duration_ms": 160,
    "speech_pad_ms": 400,
    "sampling_rate": SAMPLING_RATE,
    "window_size_samples": 512,
}


@dataclass(frozen=True)
class VadPlan:
    """The shared front's VAD decision for one request.

    ``windows`` are the packed decode windows handed to the backend;
    ``speech_chunks`` is the GLOBAL integer-sample speech-chunk list the front
    feeds to ``restore_speech_timestamps`` to lift window-relative (packed)
    segment times back to the original timeline. ``has_speech`` is False when
    VAD found nothing — the front short-circuits to an empty transcript at the
    original duration rather than decoding faster-whisper's one empty chunk.
    """

    windows: list[SpeechWindow]
    speech_chunks: list[dict[str, int]]
    duration_seconds: float
    has_speech: bool = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "has_speech", bool(self.speech_chunks))


def build_vad_plan(audio: Any) -> VadPlan:
    """Build the packed-window VAD plan for a decoded 16 kHz mono float32 array.

    Reuses faster-whisper 1.2.1's ``get_speech_timestamps`` + ``collect_chunks``
    with the pipeline's exact parameters so the packed windows match the
    ``ct2-legacy`` path byte-for-byte. When VAD finds no speech, returns a plan
    with ``has_speech=False`` and no windows (the caller emits an empty result
    at the original duration).
    """
    from faster_whisper.vad import VadOptions, collect_chunks, get_speech_timestamps

    duration_seconds = audio.shape[0] / SAMPLING_RATE

    # EXPLICIT construction — see the module docstring's numerics contract.
    vad_options = VadOptions(
        threshold=0.5,
        neg_threshold=None,
        min_speech_duration_ms=0,
        max_speech_duration_s=CHUNK_LENGTH_S,
        min_silence_duration_ms=160,
        speech_pad_ms=400,
    )
    speech_chunks = get_speech_timestamps(audio, vad_options)

    if not speech_chunks:
        return VadPlan(
            windows=[], speech_chunks=[], duration_seconds=duration_seconds
        )

    audio_chunks, chunks_metadata = collect_chunks(
        audio,
        speech_chunks,
        sampling_rate=SAMPLING_RATE,
        max_duration=CHUNK_LENGTH_S,
    )
    windows = [
        SpeechWindow(audio=chunk, metadata=meta)
        for chunk, meta in zip(audio_chunks, chunks_metadata, strict=True)
    ]
    return VadPlan(
        windows=windows,
        speech_chunks=speech_chunks,
        duration_seconds=duration_seconds,
    )
