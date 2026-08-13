"""NeMo engine — the CUDA reference runtime for ``titanet-large-v1``.

This is the exact model-invocation path the embedding space was defined on
(NeMo ``get_embedding`` over a temp wav); it must not change behaviorally
without a space-id bump. Torch/NeMo are imported lazily so the module can be
imported (not used) in torch-free environments.
"""

import logging
import tempfile
import time
from typing import Any

import numpy as np

from app.embedding import (
    SAMPLE_RATE,
    DecodeError,
    TitanetEmbedderBase,
    resolve_device_name,
)

logger = logging.getLogger(__name__)


class NemoEmbedder(TitanetEmbedderBase):
    engine = "nemo"

    def __init__(self) -> None:
        super().__init__()
        self.model: Any = None
        self.runtime = "torch"

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

    def _decode(self, audio_path: str) -> np.ndarray:
        import torch
        import torchaudio

        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as exc:
            raise DecodeError(f"Could not decode audio: {exc}") from exc
        if sample_rate != SAMPLE_RATE or waveform.shape[0] > 1:
            logger.warning(
                "non-conforming input (%d Hz, %d ch) — resampling/downmixing in-service",
                sample_rate,
                waveform.shape[0],
            )
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform.squeeze(0).numpy()

    def _embed_normalized(self, audio_np: np.ndarray) -> np.ndarray:
        import torch
        import torchaudio

        segment = torch.from_numpy(audio_np).unsqueeze(0)
        # NeMo's get_embedding wants a file path.
        with tempfile.NamedTemporaryFile(suffix=".wav") as tf:
            torchaudio.save(tf.name, segment.cpu(), SAMPLE_RATE)
            with torch.no_grad():
                embedding = self.model.get_embedding(tf.name)

        if isinstance(embedding, torch.Tensor):
            return embedding.cpu().numpy().squeeze()
        return np.array(embedding).squeeze()

    def cleanup_memory(self) -> None:
        # Keyed on runtime capability, not the reported device label:
        # torch-HIP exposes the same torch.cuda allocator API while healthz
        # honestly reports "rocm", so a label check would skip cleanup on AMD.
        if not self.model_loaded:
            return
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
