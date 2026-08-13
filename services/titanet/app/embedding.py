"""TitaNet embedding core (NeMo engine).

Model: NVIDIA NeMo TitaNet-Large, 192-dim speaker embeddings. The per-window
preprocessing chain is part of the ``titanet-large-v1`` embedding-space
definition and lives in ``app.preprocess`` (the normative definition is in
``docs/gpu-contracts.md``) — every engine consumes that module; changing any
step means a new space id, never a silent swap.
"""

import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any

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


class TitanetEmbedder:
    """Model load + single-flight, per-window embedding extraction."""

    def __init__(self) -> None:
        self.model: Any = None
        self.model_loaded = False
        self.model_name = MODEL_NAME
        self.device_name = "cpu"
        self.snr_threshold_db = float(os.getenv("TITANET_SNR_THRESHOLD_DB", "5.0"))
        # NeMo model inference is not concurrency-safe; single-flight.
        self._lock = threading.Lock()

        # /healthz identity fields (see docs/gpu-contracts.md); versions
        # resolved at load time so healthz never imports engine packages.
        self.engine = "nemo"
        self.engine_version: str | None = None
        self.runtime: str | None = "torch"
        self.runtime_version: str | None = None

    def load_model(self) -> None:
        import nemo
        import nemo.collections.asr as nemo_asr
        import torch

        self.engine_version = nemo.__version__
        self.runtime_version = torch.__version__
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device_name = resolve_device_name(device.type)
        logger.info("Loading %s on %s", self.model_name, self.device_name)
        start = time.time()
        self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name=self.model_name, map_location=device
        )
        self.model.eval()
        self.model_loaded = True
        logger.info("TitaNet loaded in %.2fs", time.time() - start)

    def embed_windows(
        self, audio_path: str, windows: list[tuple[float, float]]
    ) -> list[WindowOutcome]:
        """One outcome per window, same order — the contract's core guarantee."""
        import torch
        import torchaudio

        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as exc:
            raise DecodeError(f"Could not decode audio: {exc}") from exc
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
            sample_rate = 16000
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        outcomes: list[WindowOutcome] = []
        with self._lock:
            for start_s, end_s in windows:
                # Skip precedence (contract): too_short (no SNR measured) → low_snr.
                start_sample, end_sample = window_sample_bounds(
                    start_s, end_s, sample_rate, waveform.shape[1]
                )
                segment = waveform[:, start_sample:end_sample]
                if segment.shape[1] < int(MIN_WINDOW_SECONDS * sample_rate):
                    outcomes.append(
                        WindowOutcome(embedding=None, snr_db=None, skip_reason="too_short")
                    )
                    continue

                audio_np = segment.squeeze(0).numpy()
                snr = calculate_snr_db(audio_np)
                if snr < self.snr_threshold_db:
                    outcomes.append(
                        WindowOutcome(
                            embedding=None, snr_db=round(snr, 2), skip_reason="low_snr"
                        )
                    )
                    continue

                audio_np = normalize_audio_for_embedding(audio_np, sample_rate)
                segment = torch.from_numpy(audio_np.astype(np.float32)).unsqueeze(0)

                # NeMo's get_embedding wants a file path.
                with tempfile.NamedTemporaryFile(suffix=".wav") as tf:
                    torchaudio.save(tf.name, segment.cpu(), sample_rate)
                    with torch.no_grad():
                        embedding = self.model.get_embedding(tf.name)

                if isinstance(embedding, torch.Tensor):
                    embedding_np = embedding.cpu().numpy().squeeze()
                else:
                    embedding_np = np.array(embedding).squeeze()
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

    def cleanup_memory(self) -> None:
        # Keyed on runtime capability, not the reported device label:
        # torch-HIP exposes the same torch.cuda allocator API while healthz
        # honestly reports "rocm", so a label check would skip cleanup on AMD.
        if not self.model_loaded:
            return
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
