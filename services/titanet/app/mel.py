"""Torch-free reimplementation of TitaNet's mel-spectrogram front-end.

NeMo exports the acoustic model only (NeMo issue #7245): the
``AudioToMelSpectrogramPreprocessor`` lives OUTSIDE the ONNX graph, so the
ONNX engine must reproduce it exactly. This module transcribes NeMo 1.22.0's
``FilterbankFeatures`` eval-mode forward pass (read from the pinned service
image, provenance in ``tests/parity/fixtures/onnx/``) into NumPy:

    pad-reflect(n_fft/2) → preemphasis 0.97 → STFT (hann sym 400, hop 160,
    n_fft 512) → |X|² → slaney mel (80 bins, 0 to 8 kHz) → log(x + 2⁻²⁴) →
    per-feature mean/std normalize over valid frames → EXACTLY valid frames out

Behavioral notes, all verified against the installed 1.22.0 source:

* **Dither is training-only** (``if self.training and self.dither > 0``), so
  eval-mode extraction is deterministic and dither=1e-5 from the checkpoint
  config is intentionally NOT applied here.
* ``get_seq_len`` uses the center-padded formula ``floor(L / hop) + 1``.
* Per-feature normalization uses the *unbiased* std over the valid frames
  plus CONSTANT=1e-5, matching ``normalize_batch``.
* **Output is NEVER padded to NeMo's ``pad_to: 16``.** NeMo pads for GPU
  efficiency and masks convolution activations past ``length`` — but the
  exported graph LOST those masked convolutions (``model.export()`` replaces
  them with regular convs), so padded frames would leak into conv receptive
  fields. Measured: 0.988 cosine on a 1 s window with padding vs ≥ 0.999999
  without. See ``mel_spectrogram``'s docstring and the normative finding in
  ``docs/gpu-contracts.md``.

The measured-equivalence gate for this module is the mel level of
``tests/parity/test_titanet_onnx.py`` (vs ``references/mel/``). Numerics run
in float64 and are cast to float32 at the end; the gate is tolerance-based
(measured, then ratcheted), not bit-exact.

Constants are pinned from the checkpoint dump
(``tests/parity/fixtures/onnx/preprocessor-config.json``); a contract test
asserts they stay in sync. Changing any of them changes the embedding space.
"""

import numpy as np

SAMPLE_RATE = 16000
WIN_LENGTH = 400  # window_size 0.025 s
HOP_LENGTH = 160  # window_stride 0.01 s
N_FFT = 512
N_MELS = 80  # "features"
PREEMPH = 0.97
LOG_ZERO_GUARD = 2.0**-24
NORMALIZE_CONSTANT = 1e-5  # NeMo features.CONSTANT, added to the std
# NeMo's pad_to value — documented for completeness, intentionally NOT applied
# to this module's output (see mel_spectrogram docstring: the exported graph
# has no conv masking, so padded frames must never reach it).
NEMO_PAD_TO = 16
LOWFREQ = 0.0
HIGHFREQ = SAMPLE_RATE / 2.0
MAG_POWER = 2.0

_mel_filterbank: np.ndarray | None = None
_window: np.ndarray | None = None


def _filterbank() -> np.ndarray:
    global _mel_filterbank
    if _mel_filterbank is None:
        import librosa

        # Exactly NeMo's construction: slaney-normalized, non-HTK mel scale.
        _mel_filterbank = librosa.filters.mel(
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            n_mels=N_MELS,
            fmin=LOWFREQ,
            fmax=HIGHFREQ,
            norm="slaney",
        ).astype(np.float64)
    return _mel_filterbank


def _hann_window() -> np.ndarray:
    global _window
    if _window is None:
        # torch.hann_window(WIN_LENGTH, periodic=False) == np.hanning(WIN_LENGTH):
        # the symmetric window, later centered inside the n_fft frame like
        # torch.stft does for win_length < n_fft.
        _window = np.hanning(WIN_LENGTH).astype(np.float64)
    return _window


def num_valid_frames(num_samples: int) -> int:
    """NeMo ``get_seq_len`` for center=True: floor(L / hop) + 1.

    (Literally ``floor((L + 2*(n_fft//2) - n_fft) / hop) + 1``, which reduces
    to ``floor(L / hop) + 1`` — kept reduced for clarity.)
    """
    return num_samples // HOP_LENGTH + 1


def mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """Eval-mode NeMo mel features for one mono 16 kHz window.

    Input: float array, shape ``[num_samples]``. Output: float32
    ``[N_MELS, num_valid_frames(len(audio))]`` — exactly the valid frames,
    deliberately NOT padded to NeMo's ``pad_to: 16``.

    NeMo pads the frame axis for GPU efficiency and then MASKS convolution
    activations past ``length`` — but ``model.export()`` replaces those masked
    convolutions with regular ones ("Turned off 25 masked convolutions"), so
    the exported graph would leak zero-padding into every conv receptive field
    near the window's end. Feeding exact-length features reproduces the masked
    behavior (measured: padded-to-16 input drifted to 0.988 cosine on a 1 s
    window vs the CUDA reference; exact-length input restores ≥ 0.999999).
    Pair with ``num_valid_frames(len(audio))`` as the graph's ``length`` input.
    """
    if audio.ndim != 1:
        raise ValueError(f"expected mono audio, got shape {audio.shape}")
    if len(audio) < N_FFT:
        # Guards reflect-padding and the unbiased per-feature std (seq_len==1
        # would yield NaN). Unreachable through the service path — the
        # too_short gate rejects windows < 1 s (16000 samples) — but this
        # module must fail loud, not emit NaN vectors, if reused elsewhere.
        raise ValueError(f"mel front-end requires >= {N_FFT} samples, got {len(audio)}")
    x = audio.astype(np.float64)
    seq_len = num_valid_frames(len(x))

    # Preemphasis, first sample preserved.
    x = np.concatenate(([x[0]], x[1:] - PREEMPH * x[:-1]))

    # torch.stft(center=True): reflect-pad n_fft//2 on both sides, then frame.
    pad = N_FFT // 2
    x = np.pad(x, (pad, pad), mode="reflect")
    num_frames = 1 + (len(x) - N_FFT) // HOP_LENGTH

    # Symmetric hann of win_length, centered in the n_fft frame (torch pads the
    # window with zeros on both sides when win_length < n_fft).
    window = np.zeros(N_FFT, dtype=np.float64)
    left = (N_FFT - WIN_LENGTH) // 2
    window[left : left + WIN_LENGTH] = _hann_window()

    frame_starts = np.arange(num_frames) * HOP_LENGTH
    frames = x[frame_starts[:, None] + np.arange(N_FFT)] * window
    spectrum = np.fft.rfft(frames, n=N_FFT, axis=1)

    # Magnitude → power → mel → log(x + guard).
    power = np.abs(spectrum).T ** MAG_POWER  # [n_fft//2+1, num_frames]
    mel = _filterbank() @ power
    mel = np.log(mel + LOG_ZERO_GUARD)

    # Per-feature normalization over the valid frames (unbiased std + 1e-5).
    valid = mel[:, :seq_len]
    mean = valid.mean(axis=1, keepdims=True)
    std = valid.std(axis=1, ddof=1, keepdims=True) + NORMALIZE_CONSTANT
    mel = (mel - mean) / std

    # Exact-length output (no pad_to-16, no masking needed): with center=True
    # framing, num_frames == seq_len for every input, and the graph must never
    # see padded frames (see docstring).
    return mel[:, :seq_len].astype(np.float32)
