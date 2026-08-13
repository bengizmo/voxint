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
        logger.info("Loading TitaNet ONNX graph from %s", self.onnx_path)
        start = time.time()
        self.session = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
        self.device_name = "cpu"
        inputs = [i.name for i in self.session.get_inputs()]
        outputs = [o.name for o in self.session.get_outputs()]
        if "embs" not in outputs:
            raise RuntimeError(f"exported graph has no 'embs' output (outputs: {outputs})")
        self._input_names = inputs
        self.model_loaded = True
        logger.info("TitaNet ONNX loaded in %.2fs (inputs=%s)", time.time() - start, inputs)

    def _decode(self, audio_path: str) -> np.ndarray:
        import soundfile as sf

        try:
            audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        except Exception as exc:
            raise DecodeError(f"Could not decode audio: {exc}") from exc
        audio = audio.T  # [channels, samples]
        # Same order as the NeMo engine: resample first, then downmix to mono.
        if sample_rate != SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        return audio.mean(axis=0).astype(np.float32)

    def _embed_normalized(self, audio_np: np.ndarray) -> np.ndarray:
        mel = mel_spectrogram(audio_np)
        feats = mel[np.newaxis, :, :].astype(np.float32)
        length = np.array([num_valid_frames(len(audio_np))], dtype=np.int64)
        outputs = self.session.run(
            ["embs"], {self._input_names[0]: feats, self._input_names[1]: length}
        )
        return np.asarray(outputs[0]).squeeze()
