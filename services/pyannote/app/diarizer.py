"""Pyannote diarization core.

Model policy: **pyannote/speaker-diarization-3.1** on pyannote.audio **3.1.1**.
The 4.x line (community-1) silently rejects the classic ``clustering.threshold``
/ ``min_cluster_size`` hyperparameters this service tunes, so the 3.1 stack is
a deliberate pin, not a lag. Weights are HF-gated: the image ships none, the
user supplies ``HF_TOKEN`` and must have accepted the conditions of both
``pyannote/speaker-diarization-3.1`` and ``pyannote/segmentation-3.0``.
"""

import logging
import os
import threading
import time
from typing import Any

from app.postprocess import process_turns

logger = logging.getLogger(__name__)


class DecodeError(ValueError):
    """Input audio could not be decoded (HTTP 400 invalid_media)."""


class Diarizer:
    """Pipeline load + single-flight inference + post-processing."""

    def __init__(self) -> None:
        self.model: Any = None
        self.model_loaded = False
        self.model_name = os.getenv("DIARIZER_MODEL_NAME", "pyannote/speaker-diarization-3.1")
        self.hf_token = os.getenv("HF_TOKEN") or None
        self.device_name = "cpu"

        # Env-tunable hyperparameters. Threshold below the pyannote default
        # (~0.70) is deliberate: the default under-clusters quiet recordings
        # into 0-speaker results.
        self.clustering_threshold = float(os.getenv("PYANNOTE_CLUSTERING_THRESHOLD", "0.55"))
        self.clustering_min_size = int(os.getenv("PYANNOTE_CLUSTERING_MIN_SIZE", "10"))
        # Gap merged through in post-processing; prevents natural pauses from
        # fragmenting speakers.
        self.min_duration_off = float(os.getenv("PYANNOTE_MIN_DURATION_OFF", "0.6"))
        self.segmentation_batch_size = int(os.getenv("PYANNOTE_SEGMENTATION_BATCH_SIZE", "8"))
        self.embedding_batch_size = int(os.getenv("PYANNOTE_EMBEDDING_BATCH_SIZE", "12"))
        # Larger than the 0.1 default: fewer, larger chunks sustain GPU load
        # instead of brief bursts.
        self.segmentation_step = float(os.getenv("PYANNOTE_SEGMENTATION_STEP", "0.5"))

        # The pipeline object is not concurrency-safe. A threading.Lock (not
        # asyncio.Lock) because diarize() runs synchronously in a worker
        # thread: client cancellation abandons the thread but cannot release
        # the lock mid-inference, so a follow-up request safely queues instead
        # of running the pipeline concurrently.
        self._lock = threading.Lock()

    def load_model(self) -> None:
        import torch
        from pyannote.audio import Pipeline

        if not self.hf_token:
            logger.warning(
                "HF_TOKEN not set — the gated %s weights will not download", self.model_name
            )

        logger.info("Loading diarization pipeline: %s", self.model_name)
        start = time.time()
        # The auth kwarg name differs between pyannote releases:
        # 3.1.x wants use_auth_token=, 4.x wants token=. Try 4.x first.
        try:
            self.model = Pipeline.from_pretrained(self.model_name, token=self.hf_token)
        except TypeError as exc:
            if "token" not in str(exc):
                raise
            self.model = Pipeline.from_pretrained(
                self.model_name, use_auth_token=self.hf_token
            )
        if self.model is None:
            raise RuntimeError(
                f"Pipeline.from_pretrained returned None for {self.model_name} — "
                "usually an unaccepted HF gate or invalid HF_TOKEN"
            )

        if torch.cuda.is_available():
            self.model.to(torch.device("cuda"))
            self.device_name = "cuda"
            logger.info("Pipeline on GPU: %s", torch.cuda.get_device_name(0))
        else:
            logger.info("Pipeline on CPU")

        # Batch sizes/step are pipeline properties with setters.
        self.model.segmentation_batch_size = self.segmentation_batch_size
        self.model.embedding_batch_size = self.embedding_batch_size
        self.model.segmentation_step = self.segmentation_step

        # Clustering hyperparameters go through instantiate(); parameter names
        # vary across pipeline versions, so fall back from most to least
        # specific rather than failing the boot.
        for params in (
            {
                "clustering": {
                    "threshold": self.clustering_threshold,
                    "min_cluster_size": self.clustering_min_size,
                }
            },
            {"clustering": {"threshold": self.clustering_threshold}},
        ):
            try:
                self.model = self.model.instantiate(params)
                logger.info("Applied clustering hyperparameters: %s", params)
                break
            except Exception as exc:
                logger.debug("instantiate(%s) rejected: %s", params, exc)
        else:
            logger.warning(
                "Pipeline rejected clustering overrides; running with model defaults"
            )

        self.model_loaded = True
        logger.info("Diarization pipeline loaded in %.2fs", time.time() - start)

    def diarize(
        self,
        audio_path: str,
        *,
        min_speakers: int,
        max_speakers: int,
        min_turn_seconds: float,
    ) -> dict[str, Any]:
        """Run diarization + post-processing. Returns the contract response dict.

        Synchronous by design — the caller runs it in a worker thread
        (``run_in_threadpool``); decode, inference, and post-processing all
        happen under the single-flight lock.
        """
        import torchaudio

        with self._lock:
            # Pre-load the waveform so the pipeline never touches file-decoding
            # backends, and so we can report media duration.
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
            except Exception as exc:
                raise DecodeError(f"Could not decode audio: {exc}") from exc
            if sample_rate != 16000:
                logger.warning("Expected 16 kHz input, got %d Hz: %s", sample_rate, audio_path)
            duration = waveform.shape[1] / sample_rate

            annotation = self.model(
                {"waveform": waveform, "sample_rate": sample_rate},
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

        raw_turns = [
            {"start_seconds": turn.start, "end_seconds": turn.end, "label": speaker}
            for turn, _track_id, speaker in annotation.itertracks(yield_label=True)
        ]
        turns, speakers = process_turns(
            raw_turns,
            min_turn_seconds=min_turn_seconds,
            min_duration_off=self.min_duration_off,
        )
        return {
            "duration_seconds": duration,
            "num_speakers": len(speakers),
            "turns": turns,
            "speakers": speakers,
        }
