"""TitaNet embedding core — engine-independent pieces + the engine factory.

Model: NVIDIA NeMo TitaNet-Large, 192-dim speaker embeddings. The per-window
preprocessing chain is part of the ``titanet-large-v1`` embedding-space
definition and lives in ``app.preprocess`` (the normative definition is in
``docs/gpu-contracts.md``) — every engine consumes that module; changing any
step means a new space id, never a silent swap.

Engines (``EMBED_ENGINE`` env var, fail-fast on unknown values):

* ``nemo`` (default) — ``app.engine_nemo.NemoEmbedder``, the CUDA reference
  runtime (NeMo + torch).
* ``onnx`` — ``app.engine_onnx.OnnxEmbedder``, ONNX Runtime on the
  self-exported graph (``tools/export_titanet_onnx.py``) + the reimplemented
  mel front-end (``app.mel``); torch-free.

The window loop itself (decode → slice → skip gates → normalize → model → L2)
is implemented ONCE in :class:`TitanetEmbedderBase`; an engine only supplies
audio decoding and the model call. That keeps the skip semantics and the
preprocessing chain structurally identical across engines instead of relying
on review to catch drift.

This module stays importable without torch/NeMo/onnxruntime (contract-test
requirement); engine modules import their stacks lazily.
"""

import logging
import os
import threading
from dataclasses import dataclass

import numpy as np

from app.preprocess import (
    MIN_WINDOW_SECONDS,
    calculate_snr_db,
    l2_normalize,
    normalize_audio_for_embedding,
    window_sample_bounds,
)

logger = logging.getLogger(__name__)


class DecodeError(ValueError):
    """Input audio could not be decoded (HTTP 400 invalid_media)."""


MODEL_NAME = "nvidia/speakerverification_en_titanet_large"
SAMPLE_RATE = 16000


def resolve_device_name(device_type: str) -> str:
    """Honest /healthz device reporting: torch built for ROCm masquerades as
    CUDA (``torch.cuda.is_available()`` is true, device type is ``cuda``), so
    report ``rocm`` whenever ``torch.version.hip`` is set."""
    if device_type == "cuda":
        import torch

        if getattr(torch.version, "hip", None):
            return "rocm"
    return device_type


@dataclass(frozen=True)
class WindowOutcome:
    embedding: list[float] | None
    snr_db: float | None
    skip_reason: str | None


class TitanetEmbedderBase:
    """Shared window loop; engines implement decode + the model call."""

    engine: str

    def __init__(self) -> None:
        self.model_loaded = False
        self.model_name = MODEL_NAME
        self.device_name = "cpu"
        self.snr_threshold_db = float(os.getenv("TITANET_SNR_THRESHOLD_DB", "5.0"))
        # Model inference is single-flight on every engine: NeMo is not
        # concurrency-safe, and keeping ONNX identical removes a behavioral
        # variable from parity comparisons.
        self._lock = threading.Lock()

        # /healthz identity fields (see docs/gpu-contracts.md); versions
        # resolved at load time so healthz never imports engine packages.
        self.engine_version: str | None = None
        self.runtime: str | None = None
        self.runtime_version: str | None = None

    def load_model(self) -> None:
        raise NotImplementedError

    def _decode(self, audio_path: str) -> np.ndarray:
        """Decode to mono float32 at 16 kHz; raise DecodeError on bad media."""
        raise NotImplementedError

    def _embed_normalized(self, audio_np: np.ndarray) -> np.ndarray:
        """Embed one preprocessed window (float32 @ 16 kHz) → 192-dim vector
        (pre-L2; the base applies the space definition's final L2 step)."""
        raise NotImplementedError

    def cleanup_memory(self) -> None:
        """Post-request memory hook; engines override when their runtime needs it."""

    def embed_windows(
        self, audio_path: str, windows: list[tuple[float, float]]
    ) -> list[WindowOutcome]:
        """One outcome per window, same order — the contract's core guarantee."""
        audio = self._decode(audio_path)

        outcomes: list[WindowOutcome] = []
        with self._lock:
            for start_s, end_s in windows:
                # Skip precedence (contract): too_short (no SNR measured) → low_snr.
                start_sample, end_sample = window_sample_bounds(
                    start_s, end_s, SAMPLE_RATE, len(audio)
                )
                segment = audio[start_sample:end_sample]
                if len(segment) < int(MIN_WINDOW_SECONDS * SAMPLE_RATE):
                    outcomes.append(
                        WindowOutcome(embedding=None, snr_db=None, skip_reason="too_short")
                    )
                    continue

                snr = calculate_snr_db(segment)
                if snr < self.snr_threshold_db:
                    outcomes.append(
                        WindowOutcome(embedding=None, snr_db=round(snr, 2), skip_reason="low_snr")
                    )
                    continue

                normalized = normalize_audio_for_embedding(segment, SAMPLE_RATE)
                embedding_np = self._embed_normalized(normalized.astype(np.float32))
                if embedding_np.ndim > 1:
                    embedding_np = embedding_np.mean(axis=0)
                embedding_np = l2_normalize(embedding_np)

                outcomes.append(
                    WindowOutcome(
                        embedding=[float(x) for x in embedding_np],
                        snr_db=round(snr, 2),
                        skip_reason=None,
                    )
                )
        return outcomes


def create_embedder() -> TitanetEmbedderBase:
    """Instantiate the engine selected by ``EMBED_ENGINE`` (fail-fast)."""
    engine = os.getenv("EMBED_ENGINE", "nemo")
    if engine == "nemo":
        from app.engine_nemo import NemoEmbedder

        return NemoEmbedder()
    if engine == "onnx":
        from app.engine_onnx import OnnxEmbedder

        return OnnxEmbedder()
    raise ValueError(f"Unknown EMBED_ENGINE {engine!r} (expected 'nemo' or 'onnx')")
