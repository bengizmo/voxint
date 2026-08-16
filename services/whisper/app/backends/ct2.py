"""``ct2`` backend: shared-VAD CT2 decode. Types now; decode in Slice 2b.

This module defines the ``shared_windows`` boundary types (``SpeechWindow``,
``RawResult``) and a fail-closed stub so the abstraction is fixed before the
risky, numerics-touching 2b work fills it in. The shared front (in
``transcription.py``, 2b) will own VAD, packing, the piecewise packed->source
time map, confidence and assembly; this backend will only decode packed
windows through batched CT2 with the internal VAD disabled.

Selecting ``WHISPER_ENGINE=ct2`` today resolves to this stub and fails closed
at load — it never silently degrades to ``ct2-legacy``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from app.backends import TranscribeOptions


@dataclass(frozen=True)
class SpeechWindow:
    """One packed decode window produced by the shared front-layer VAD.

    ``collect_chunks`` concatenates *non-contiguous* speech intervals into a
    single packed decode window, so a single offset is wrong: ``sample_map``
    carries the piecewise packed-sample -> source-sample correspondence
    (integer samples) the front uses to restore absolute source time. Fields
    are defined for 2b; the shape is intentionally minimal until the VADPlan
    prototype is validated against faster-whisper 1.2.1.
    """

    # Packed 16 kHz PCM for this window (mono float32), speech intervals only.
    audio: Any
    # Piecewise (packed_start_sample, source_start_sample, length_samples)
    # spans mapping packed offsets back to the original file's timeline.
    sample_map: tuple[tuple[int, int, int], ...] = ()


@dataclass
class RawResult:
    """Window-relative decode output crossing the backend -> front boundary.

    ``segments`` are faster-whisper Segment objects (window-relative
    timestamps); the front restores absolute source time via the window's
    ``sample_map`` and assembles the ``TranscriptionOutput`` (2b).
    """

    segments: list[Any] = field(default_factory=list)
    language: str | None = None
    duration: float = 0.0


class Ct2Backend:
    """Shared-window CT2 backend — stub until Slice 2b (fails closed at load)."""

    kind: Literal["shared_windows"] = "shared_windows"

    def __init__(
        self,
        model_name: str = "large-v2",
        device: str = "cuda",
        compute_type: str = "int8",
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self._requested_device = device

        self.model: Any = None
        self.is_initialized = False
        self.engine = "faster-whisper"
        self.engine_version: str | None = None
        self.runtime: str | None = "ctranslate2"
        self.runtime_version: str | None = None

    def verify_device(self) -> None:  # pragma: no cover - unreachable until 2b
        raise NotImplementedError(
            "WHISPER_ENGINE=ct2 (shared-VAD) is not implemented until Slice 2b"
        )

    def load_model(self) -> None:
        # Fail closed, loudly, at startup — never a silent fallback to legacy.
        raise NotImplementedError(
            "WHISPER_ENGINE=ct2 (shared-VAD) is not implemented until Slice 2b; "
            "use the default WHISPER_ENGINE=ct2-legacy"
        )

    def decode_windows(
        self, windows: list[SpeechWindow], options: TranscribeOptions
    ) -> list[RawResult]:  # pragma: no cover - unreachable until 2b
        raise NotImplementedError(
            "WHISPER_ENGINE=ct2 (shared-VAD) decode lands in Slice 2b"
        )

    def cleanup_memory(self) -> None:  # pragma: no cover - never initialized in 2a
        return
