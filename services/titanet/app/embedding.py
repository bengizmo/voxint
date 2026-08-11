"""TitaNet embedding core.

Model: NVIDIA NeMo TitaNet-Large, 192-dim speaker embeddings. The full
preprocessing chain below is part of the ``titanet-large-v1`` embedding-space
definition — changing any step means a new space id, never a silent swap.

Per-window chain: slice → resample 16 kHz mono → stationary spectral-gating
noise reduction → LUFS normalization to -16 LUFS → peak normalization to 0.95
→ TitaNet → L2 normalization. LUFS + noise reduction exist because raw
loudness/noise variance fragments a single speaker's embeddings into multiple
clusters (measured on poor audio: mean cosine similarity 0.30 unnormalized vs
0.64 on clean audio).
"""

import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DecodeError(ValueError):
    """Input audio could not be decoded (HTTP 400 invalid_media)."""


MIN_WINDOW_SECONDS = 1.0
MODEL_NAME = "nvidia/speakerverification_en_titanet_large"


@dataclass(frozen=True)
class WindowOutcome:
    embedding: list[float] | None
    snr_db: float | None
    skip_reason: str | None


def calculate_snr_db(audio: np.ndarray, frame_length: int = 2048) -> float:
    """Estimate SNR: RMS energy over the noise floor (quietest 10% of frames)."""
    rms = float(np.sqrt(np.mean(audio**2)))
    # Silence (or near-silence) has no signal: report 0 dB so the low_snr gate
    # skips it. Without this check a ~zero noise floor made pure silence score
    # as pristine 40 dB audio.
    if rms < 1e-6:
        return 0.0
    frame_energies = [
        float(np.sqrt(np.mean(audio[i : i + frame_length] ** 2)))
        for i in range(0, len(audio) - frame_length, frame_length)
    ]
    if not frame_energies:
        return 20.0
    frame_energies.sort()
    noise_floor = float(np.mean(frame_energies[: max(1, len(frame_energies) // 10)]))
    if noise_floor < 1e-10:
        # Real signal over a digitally-silent floor (e.g. gated/denoised input).
        return 40.0
    snr_db = 20.0 * float(np.log10(rms / noise_floor))
    return max(0.0, min(60.0, snr_db))


def normalize_audio_for_embedding(audio_np: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Noise-reduce + loudness-normalize a window before embedding.

    This chain is part of the titanet-large-v1 space definition, so failures
    are request-fatal: silently skipping a stage would emit vectors with
    different preprocessing semantics under the same space id. (The one
    tolerated no-op: pyloudnorm returning -inf loudness for content it cannot
    meter — the LUFS *target* simply cannot apply to such audio, and that is
    deterministic per input, not a degradation.)
    """
    import noisereduce as nr
    import pyloudnorm as pyln

    audio_np = nr.reduce_noise(y=audio_np, sr=sample_rate, stationary=True, prop_decrease=0.75)

    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(audio_np)
    if np.isfinite(loudness):
        audio_np = pyln.normalize.loudness(audio_np, loudness, -16.0)

    peak = float(np.max(np.abs(audio_np)))
    if peak > 0:
        audio_np = audio_np / peak * 0.95
    return audio_np


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

    def load_model(self) -> None:
        import nemo.collections.asr as nemo_asr
        import torch

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device_name = device.type
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
                start_sample = int(start_s * sample_rate)
                end_sample = min(int(end_s * sample_rate), waveform.shape[1])
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
                embedding_np = embedding_np / (np.linalg.norm(embedding_np) + 1e-8)

                outcomes.append(
                    WindowOutcome(
                        embedding=[float(x) for x in embedding_np],
                        snr_db=round(snr, 2),
                        skip_reason=None,
                    )
                )
        return outcomes

    def cleanup_memory(self) -> None:
        if self.device_name == "cuda":
            import torch

            torch.cuda.empty_cache()
