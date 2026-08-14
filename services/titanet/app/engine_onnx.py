"""ONNX Runtime engine — TitaNet acoustic model on CPU (or any ORT EP).

Torch-free: decodes with soundfile/librosa, computes mel features with
``app.mel`` (the reimplemented NeMo front-end), and runs the self-exported
graph (``tools/export_titanet_onnx.py``; sha256 + provenance in
``tests/parity/fixtures/onnx/provenance.json``). Kept on the same embedding
space id only because the measured-equivalence gate
(``tests/parity/test_titanet_onnx.py``) passes — see docs/gpu-contracts.md.

``TITANET_ONNX_PATH`` points at the .onnx artifact (default
``/app/models/titanet-large.onnx`` in the CPU image).
"""

import logging
import os
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from app.embedding import (
    SAMPLE_RATE,
    DecodeError,
    TitanetEmbedderBase,
)
from app.mel import mel_spectrogram, num_valid_frames

logger = logging.getLogger(__name__)

DEFAULT_ONNX_PATH = "/app/models/titanet-large.onnx"

# Shipped images set no TITANET_ORT_PROVIDERS: the default must stay exactly
# the CPU EP chain the committed parity verdict measured
# (docs/gpu-contracts.md — the verdict binds to requirements.cpu.txt + CPU EP).
DEFAULT_ORT_PROVIDERS = ["CPUExecutionProvider"]

# healthz honesty (docs/gpu-contracts.md): device_name reports where inference
# actually runs. "metal" means the CoreML EP is active (Apple GPU/ANE through
# onnxruntime) — distinct from "mps", which only the pyannote service reports
# (torch Metal). EPs outside this map degrade to their lowercased stem
# ("ROCMExecutionProvider" -> "rocm") so a new provider is honest, if terse.
_PROVIDER_DEVICE_NAMES = {
    "CPUExecutionProvider": "cpu",
    "CoreMLExecutionProvider": "metal",
    "CUDAExecutionProvider": "cuda",
}


def parse_ort_providers(raw: str) -> list[str]:
    """Parse ``TITANET_ORT_PROVIDERS``: comma-separated, priority-ordered EPs.

    Blank entries are dropped; an entirely blank string parses to ``[]`` and
    the caller falls back to ``DEFAULT_ORT_PROVIDERS`` (so a compose-style
    ``${TITANET_ORT_PROVIDERS:-}`` passing "" through behaves like unset).
    """
    return [part.strip() for part in raw.split(",") if part.strip()]


def provider_device_name(provider: str) -> str:
    stem = provider.removesuffix("ExecutionProvider").lower()
    return _PROVIDER_DEVICE_NAMES.get(provider, stem or provider.lower())


def validate_requested_providers(
    requested: Sequence[str], available: Sequence[str]
) -> None:
    """Fail BEFORE session construction when a requested EP is not built in."""
    missing = [p for p in requested if p not in available]
    if missing:
        raise RuntimeError(
            f"TITANET_ORT_PROVIDERS requests {missing}, but this onnxruntime "
            f"build only provides {sorted(available)} — refusing to build a "
            "session that would silently run on a different EP"
        )


def assert_session_honors_providers(
    requested: Sequence[str], actual: Sequence[str]
) -> None:
    """Fail AFTER construction unless the session runs the requested EPs.

    ORT drops an EP it cannot initialize with only a log line and falls back
    down the list, so a session can come up healthy-looking on the wrong
    device. The requested chain must be a prefix of what the session reports
    (prefix, not equality: ORT always appends the CPU EP as final fallback).
    """
    if list(actual[: len(requested)]) != list(requested):
        raise RuntimeError(
            f"onnxruntime session runs {list(actual)} but TITANET_ORT_PROVIDERS "
            f"requested {list(requested)} — ORT silently degraded; failing at "
            "load so healthz cannot report a device that is not in use"
        )


class OnnxEmbedder(TitanetEmbedderBase):
    engine = "onnxruntime"

    def __init__(self) -> None:
        super().__init__()
        self.session: Any = None
        self.onnx_path = os.getenv("TITANET_ONNX_PATH", DEFAULT_ONNX_PATH)
        self.runtime = "onnxruntime"

    def load_model(self) -> None:
        import onnxruntime as ort

        self.engine_version = ort.__version__
        self.runtime_version = ort.__version__
        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(
                f"TITANET_ONNX_PATH not found: {self.onnx_path} "
                "(export with tools/export_titanet_onnx.py)"
            )
        requested = parse_ort_providers(os.getenv("TITANET_ORT_PROVIDERS", ""))
        if not requested:
            requested = list(DEFAULT_ORT_PROVIDERS)
        validate_requested_providers(requested, ort.get_available_providers())
        logger.info(
            "Loading TitaNet ONNX graph from %s (providers=%s)", self.onnx_path, requested
        )
        start = time.time()
        self.session = ort.InferenceSession(self.onnx_path, providers=requested)
        assert_session_honors_providers(requested, self.session.get_providers())
        self.device_name = provider_device_name(requested[0])
        inputs = {i.name for i in self.session.get_inputs()}
        outputs = [o.name for o in self.session.get_outputs()]
        # Bind graph I/O by name, never by declaration order — a re-export
        # with reordered inputs must fail at load, not at first inference.
        if not {"audio_signal", "length"} <= inputs:
            raise RuntimeError(
                f"exported graph inputs {sorted(inputs)} missing audio_signal/length"
            )
        if "embs" not in outputs:
            raise RuntimeError(f"exported graph has no 'embs' output (outputs: {outputs})")
        self.model_loaded = True
        logger.info("TitaNet ONNX loaded in %.2fs (inputs=%s)", time.time() - start, sorted(inputs))

    def _decode(self, audio_path: str) -> np.ndarray:
        import soundfile as sf

        try:
            audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        except Exception as exc:
            raise DecodeError(f"Could not decode audio: {exc}") from exc
        audio = audio.T  # [channels, samples]
        # Same order as the NeMo engine: resample first, then downmix to mono.
        # NOTE (documented deviation, docs/gpu-contracts.md): this fallback
        # resampler (librosa/soxr) is a different kernel than the NeMo
        # engine's torchaudio sinc resampler and is unmeasured by the parity
        # gate — voxint's prepare stage normalizes all media to 16 kHz mono
        # before the services, so conforming deployments never hit it.
        if sample_rate != SAMPLE_RATE or audio.shape[0] > 1:
            logger.warning(
                "non-conforming input (%d Hz, %d ch) — resampling/downmixing in-service",
                sample_rate,
                audio.shape[0],
            )
        if sample_rate != SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        return audio.mean(axis=0).astype(np.float32)

    def _embed_normalized(self, audio_np: np.ndarray) -> np.ndarray:
        mel = mel_spectrogram(audio_np)
        feats = mel[np.newaxis, :, :].astype(np.float32)
        length = np.array([num_valid_frames(len(audio_np))], dtype=np.int64)
        outputs = self.session.run(["embs"], {"audio_signal": feats, "length": length})
        return np.asarray(outputs[0]).squeeze()
