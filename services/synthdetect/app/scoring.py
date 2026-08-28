"""Synthdetect inference engine -- model load, windowing, forward pass, pooling.

Implements the w2v2-AASIST scoring pipeline: read PCM from a 16kHz mono WAV,
tile each requested interval into 64,600-sample model-width windows (with
repeat-padding for short clips), run the fairseq AASIST forward pass, and
mean-pool raw logits per interval.

The windowing policy is part of the inference space identity. Changes to
window size, stride, padding mode, or pooling require a new inference_space
string.

Score convention: ``-bona_fide_logit`` (higher = more synthetic). The raw
score is the negative of the model's column-1 output (which is the bona-fide
logit in the upstream SSL_Anti-spoofing convention).
"""

from __future__ import annotations

import hashlib
import logging
import os
import wave
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

MODEL_WINDOW_SAMPLES = 64_600
SAMPLE_RATE = 16_000
MIN_SCORABLE_SAMPLES = 8_000

WEIGHTS_DIR = Path(os.getenv("SYNTHDETECT_WEIGHTS_DIR", "/app/weights"))
AASIST_CHECKPOINT = os.getenv("SYNTHDETECT_AASIST_CHECKPOINT", "finetuned_aasist.pth")
XLSR_CHECKPOINT = os.getenv("SYNTHDETECT_XLSR_CHECKPOINT", "xlsr2_300m.pt")

AASIST_CHECKPOINT_SHA = os.getenv(
    "SYNTHDETECT_AASIST_SHA",
    "e178446b640b8e9f9cf6dd359428b2243f49e24e613e1ae952cd706216b8111e",
)
XLSR_CHECKPOINT_SHA = os.getenv(
    "SYNTHDETECT_XLSR_SHA",
    "b08927597f2c9eb2ebd7dcc3ac78ee4b5f6021cbac4b3a6c5a9deec445d80ed9",
)

DEVICE = os.getenv("SYNTHDETECT_DEVICE", "cuda:0")


class DecodeError(Exception):
    """Audio file cannot be read or does not meet the 16kHz mono S16LE contract."""


@dataclass(frozen=True)
class IntervalOutcome:
    raw_score: float | None
    window_count: int
    skip_reason: str | None


def _read_pcm_payload(path: str) -> np.ndarray:
    """Read a 16kHz mono S16LE WAV and return float32 samples in [-1, 1]."""
    try:
        with wave.open(path, "rb") as wf:
            if wf.getnchannels() != 1:
                raise DecodeError(f"expected mono, got {wf.getnchannels()} channels")
            if wf.getframerate() != SAMPLE_RATE:
                raise DecodeError(
                    f"expected {SAMPLE_RATE}Hz, got {wf.getframerate()}Hz"
                )
            if wf.getsampwidth() != 2:
                raise DecodeError(
                    f"expected 16-bit (2 bytes), got {wf.getsampwidth()} bytes"
                )
            raw = wf.readframes(wf.getnframes())
    except wave.Error as exc:
        raise DecodeError(f"cannot read WAV: {exc}") from exc

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


def _tile_windows(
    n_samples: int,
    start_sample: int,
    end_sample: int,
) -> list[tuple[int, int]]:
    """Tile an interval into non-overlapping MODEL_WINDOW_SAMPLES windows.

    Returns a list of (start, end) sample indices. Short tails below
    MIN_SCORABLE_SAMPLES are dropped when at least one full window exists.
    """
    length = end_sample - start_sample
    if length < MIN_SCORABLE_SAMPLES:
        return []

    windows: list[tuple[int, int]] = []
    pos = start_sample
    while pos + MODEL_WINDOW_SAMPLES <= end_sample:
        windows.append((pos, pos + MODEL_WINDOW_SAMPLES))
        pos += MODEL_WINDOW_SAMPLES

    remaining = end_sample - pos
    if remaining > 0:
        if not windows:
            windows.append((start_sample, end_sample))
        elif remaining >= MIN_SCORABLE_SAMPLES:
            windows.append((pos, end_sample))

    return windows


def _repeat_pad(samples: np.ndarray, target_length: int) -> np.ndarray:
    """Repeat-pad a short clip to ``target_length`` samples.

    Matches the upstream SSL_Anti-spoofing convention: tile the clip until the
    target length is reached, then truncate. Clips already at or above the
    target length are returned as a prefix slice.
    """
    n = len(samples)
    if n >= target_length:
        return samples[:target_length]
    repeats = (target_length + n - 1) // n
    return np.tile(samples, repeats)[:target_length]


@contextmanager
def _pushd(directory: Path):
    """Temporarily change the working directory."""
    prev = os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(prev)


def _verify_weight_file(path: Path, expected_sha: str) -> None:
    """Verify a weight file's sha256 matches the expected hash."""
    if not expected_sha:
        logger.warning("No sha256 configured for %s; skipping verification", path.name)
        return
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected_sha:
        raise RuntimeError(
            f"Weight file {path.name} sha256 mismatch: "
            f"expected {expected_sha[:16]}..., got {actual[:16]}..."
        )


class SynthdetectScorer:
    """Lazy-loaded AASIST scorer with service-side windowing."""

    def __init__(self) -> None:
        self._model = None
        self._device_name = DEVICE
        self._model_name: str | None = None
        self._engine_version: str | None = None
        self._runtime_version: str | None = None

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> str | None:
        return self._model_name

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def engine(self) -> str:
        return "fairseq-aasist"

    @property
    def engine_version(self) -> str | None:
        return self._engine_version

    @property
    def runtime(self) -> str | None:
        return "cuda" if "cuda" in self._device_name else "cpu"

    @property
    def runtime_version(self) -> str | None:
        return self._runtime_version

    def load_model(self) -> None:
        """Load the AASIST model; called once at startup in the lifespan."""
        import torch

        self._configure_determinism()

        aasist_path = WEIGHTS_DIR / AASIST_CHECKPOINT
        xlsr_path = WEIGHTS_DIR / XLSR_CHECKPOINT

        if not aasist_path.exists():
            raise FileNotFoundError(f"AASIST checkpoint not found: {aasist_path}")
        if not xlsr_path.exists():
            raise FileNotFoundError(f"XLS-R checkpoint not found: {xlsr_path}")

        _verify_weight_file(aasist_path, AASIST_CHECKPOINT_SHA)
        _verify_weight_file(xlsr_path, XLSR_CHECKPOINT_SHA)

        from app.vendor.ssl_antispoofing_model import Model as AASISTModel

        device = torch.device(self._device_name)

        with _pushd(WEIGHTS_DIR):
            model = AASISTModel(None, device)

        state_dict = torch.load(str(aasist_path), map_location="cpu", weights_only=False)

        key_prefix = None
        sample_key = next(iter(state_dict.keys()), "")
        if sample_key.startswith("module."):
            key_prefix = "module."

        if key_prefix:
            state_dict = {
                k[len(key_prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(key_prefix)
            }

        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        model.eval()

        for m in model.modules():
            if m.training:
                raise RuntimeError(
                    f"Module {type(m).__name__} still in train mode after eval()"
                )

        self._model = model
        self._model_name = "w2v2-aasist-df-m2-s0e11"

        self._engine_version = "fairseq"
        try:
            import fairseq
            self._engine_version = f"fairseq-{fairseq.__version__}"
        except (ImportError, AttributeError):
            pass

        try:
            self._runtime_version = torch.version.cuda
        except AttributeError:
            self._runtime_version = None

        logger.info(
            "Synthdetect model loaded on %s (engine=%s, runtime=%s)",
            self._device_name,
            self._engine_version,
            self._runtime_version,
        )

    def _configure_determinism(self) -> None:
        import torch
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    def score_intervals(
        self,
        audio_path: str,
        intervals: list[tuple[float, float]],
    ) -> list[IntervalOutcome]:
        """Score a list of audio intervals, returning one result per interval."""
        import torch

        if self._model is None:
            raise RuntimeError("Model not loaded")

        pcm = _read_pcm_payload(audio_path)
        n_samples = len(pcm)
        results: list[IntervalOutcome] = []

        for start_sec, end_sec in intervals:
            start_sample = round(start_sec * SAMPLE_RATE)
            end_sample = min(round(end_sec * SAMPLE_RATE), n_samples)

            if end_sample <= start_sample:
                results.append(IntervalOutcome(
                    raw_score=None, window_count=0, skip_reason="too_short",
                ))
                continue

            windows = _tile_windows(n_samples, start_sample, end_sample)

            if not windows:
                results.append(IntervalOutcome(
                    raw_score=None, window_count=0, skip_reason="too_short",
                ))
                continue

            window_logits: list[float] = []
            device = torch.device(self._device_name)

            for w_start, w_end in windows:
                segment = pcm[w_start:w_end]
                segment = _repeat_pad(segment, MODEL_WINDOW_SAMPLES)

                tensor = torch.from_numpy(segment).unsqueeze(0).to(device)

                with torch.inference_mode():
                    out = self._model(tensor)
                    logit = -float(out[0, 1])
                    window_logits.append(logit)

            pooled = sum(window_logits) / len(window_logits)

            results.append(IntervalOutcome(
                raw_score=pooled,
                window_count=len(window_logits),
                skip_reason=None,
            ))

        return results

    def cleanup_memory(self) -> None:
        """Best-effort GPU memory cleanup after a scoring batch."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def create_scorer() -> SynthdetectScorer:
    return SynthdetectScorer()
