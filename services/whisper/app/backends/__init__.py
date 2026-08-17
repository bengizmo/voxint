"""Whisper engine registry: fail-closed ``WHISPER_ENGINE`` selection.

The whisper service can run its decode through more than one engine while
keeping a single public surface (``WhisperTranscriber`` +
``/v1/transcribe`` + ``/healthz``). This package holds the backend
strategies; ``transcription.py`` keeps the engine-agnostic front layer and
the facade.

Two *typed strategies*, not one protocol and not an ``owns_vad`` flag — the
whole-file legacy engine and the shared-window engine assemble results at
different layers, so they expose different methods:

* ``legacy_file`` (``kind == "legacy_file"``): owns the whole-file decode AND
  the result-assembly loop. ``transcribe_file(path, options) ->
  TranscriptionOutput`` and the front passes that output straight through.
  ``ct2-legacy`` is the only implementation and is byte-faithful to the
  pre-seam shipped code (Slice 2a of #33).
* ``shared_windows`` (``kind == "shared_windows"``): decodes pre-VAD'd,
  packed windows (``decode_windows``) or runs the non-VAD sequential path
  (``transcribe_raw``); the shared front owns VAD, packing, time remap,
  confidence and assembly. ``ct2`` is the implementation (see
  ``app.backends.ct2``, Slice 2b of #33).

Selection is fail-closed: an unknown ``WHISPER_ENGINE`` raises ``ValueError``,
never a silent fallback (numerics doctrine — model outputs are contract). The
factory mirrors titanet's ``create_embedder``
(``services/titanet/app/embedding.py``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from app.backends.ct2 import RawResult, SpeechWindow
    from app.transcription import TranscriptionOutput, WhisperTranscriber

# Default engine: the byte-faithful shipped path. Every deployment that does
# not set WHISPER_ENGINE gets exactly the pre-seam numerics.
DEFAULT_ENGINE = "ct2-legacy"

# Engines the registry knows how to construct. ``ct2`` resolves to the
# shared-window backend (batched VAD decode + raw sequential decode); an
# unknown value fails closed, never silently degrading to the legacy path.
KNOWN_ENGINES = ("ct2-legacy", "ct2")


@dataclass(frozen=True)
class TranscribeOptions:
    """Decode request crossing the front -> backend boundary.

    Mirrors the public ``WhisperTranscriber.transcribe`` parameters so a
    backend never re-reads env or re-derives request state.
    """

    language: str | None = "en"
    initial_prompt: str | None = None
    vad_filter: bool = True


class LegacyFileBackend(Protocol):
    """Strategy that owns whole-file decode and result assembly."""

    kind: Literal["legacy_file"]
    model_name: str
    device: str
    is_initialized: bool
    engine: str
    engine_version: str | None
    runtime: str | None
    runtime_version: str | None

    def load_model(self) -> None: ...
    def verify_device(self) -> None: ...
    def cleanup_memory(self) -> None: ...
    def transcribe_file(
        self, audio_path: str, options: TranscribeOptions
    ) -> TranscriptionOutput: ...


class SharedWindowsBackend(Protocol):
    """Strategy that decodes pre-VAD'd windows; the front assembles (2b)."""

    kind: Literal["shared_windows"]
    model_name: str
    device: str
    is_initialized: bool
    engine: str
    engine_version: str | None
    runtime: str | None
    runtime_version: str | None

    def load_model(self) -> None: ...
    def verify_device(self) -> None: ...
    def cleanup_memory(self) -> None: ...
    def decode_windows(
        self, windows: list[SpeechWindow], options: TranscribeOptions
    ) -> RawResult: ...
    def transcribe_raw(
        self, audio_path: str, options: TranscribeOptions
    ) -> RawResult: ...


def create_transcriber(
    *, model_name: str, device: str, compute_type: str, batch_size: int
) -> WhisperTranscriber:
    """Build the ``WhisperTranscriber`` facade around the backend selected by
    ``WHISPER_ENGINE`` (default ``ct2-legacy``).

    Fail-closed: an unknown engine raises ``ValueError`` rather than falling
    back to CPU/legacy silently. Mirrors ``create_embedder``.
    """
    engine = os.getenv("WHISPER_ENGINE", DEFAULT_ENGINE).strip()
    backend = _resolve_backend(
        engine,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
    )
    # Lazy: the facade lives in the front-layer module, which imports this
    # package for TranscribeOptions — import it at call time to keep the
    # dependency one-directional.
    from app.transcription import WhisperTranscriber

    return WhisperTranscriber(backend=backend)


def _resolve_backend(
    engine: str, *, model_name: str, device: str, compute_type: str, batch_size: int
) -> Any:
    """Lazy-import and construct the backend for ``engine`` (fail-closed)."""
    if engine == "ct2-legacy":
        from app.backends.ct2_legacy import Ct2LegacyBackend

        return Ct2LegacyBackend(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            batch_size=batch_size,
        )
    if engine == "ct2":
        from app.backends.ct2 import Ct2Backend

        return Ct2Backend(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            batch_size=batch_size,
        )
    raise ValueError(
        f"Unknown WHISPER_ENGINE {engine!r} (expected one of {KNOWN_ENGINES})"
    )
