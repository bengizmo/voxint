"""Reference implementation of ``titanet-large-v1`` window preprocessing.

This module is the code half of the space definition in
``docs/gpu-contracts.md`` (steps 1-6 plus the final L2 normalization): slice →
resample 16 kHz mono → skip gates (too_short / low_snr) → stationary
spectral-gating noise reduction → LUFS normalization to -16 LUFS → peak
normalization to 0.95 → model → L2 normalization. Every engine (NeMo/CUDA,
ONNX Runtime, future backends) must consume THIS module — a parallel
implementation of any step is a new embedding space, never a silent swap.

LUFS + noise reduction exist because raw loudness/noise variance fragments a
single speaker's embeddings into multiple clusters (measured on poor audio:
mean cosine similarity 0.30 unnormalized vs 0.64 on clean audio).

Torch-free on purpose: contract tests import this module standalone, and the
ONNX image ships it without torch.
"""

import numpy as np

MIN_WINDOW_SECONDS = 1.0


def window_sample_bounds(
    start_seconds: float, end_seconds: float, sample_rate: int, total_samples: int
) -> tuple[int, int]:
    """Sample-precise window slice (space definition step 1).

    Truncating int conversion — not rounding — at ``start_seconds x sr``, end
    clamped to the media length. Changing this math changes which samples every
    engine embeds, so it lives here rather than inline in any engine.
    """
    start_sample = int(start_seconds * sample_rate)
    end_sample = min(int(end_seconds * sample_rate), total_samples)
    return start_sample, end_sample


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


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Final L2 normalization of the embedding vector (space definition step 8)."""
    return vector / (np.linalg.norm(vector) + 1e-8)
