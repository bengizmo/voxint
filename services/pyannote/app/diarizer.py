"""Pyannote diarization core.

Model policy: **pyannote/speaker-diarization-3.1** on pyannote.audio **3.1.1**.
The 4.x line (community-1) silently rejects the classic ``clustering.threshold``
/ ``min_cluster_size`` hyperparameters this service tunes, so the 3.1 stack is
a deliberate pin, not a lag.

Weights: the image bakes the vendored pipeline (config + sha256-verified
checkpoints from the ``pyannote-models-v1`` asset release, see
``services/pyannote/models/provenance.json``) at ``VOXINT_VENDORED_PIPELINE``
and loads it by default — no Hugging Face account or token involved. Setting
``DIARIZER_MODEL_NAME`` to an HF repo id restores the online path, where
``HF_TOKEN`` is required for gated repos.
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


def resolve_device_name(device_type: str) -> str:
    """Honest /healthz device reporting: torch built for ROCm masquerades as
    CUDA (``torch.cuda.is_available()`` is true, device type is ``cuda``), so
    report ``rocm`` whenever ``torch.version.hip`` is set."""
    if device_type == "cuda":
        import torch

        if getattr(torch.version, "hip", None):
            return "rocm"
    return device_type


def probe_device(device_name: str) -> bool:
    """Run a real tensor op on the device and compare against a CPU reference.

    Availability flags are not enough: the historical MPS failure mode is
    *silent wrong output*, not an exception — a backend that computes garbage
    still reports ``is_available() == True``. A fixed-seed matmul checked
    against the CPU result catches both crash-y and silently-wrong backends
    before the pipeline is moved onto one.
    """
    import torch

    try:
        gen = torch.Generator().manual_seed(0)
        x = torch.randn(64, 64, generator=gen)
        reference = x @ x
        result = (x.to(device_name) @ x.to(device_name)).cpu()
        # Loose on purpose: TF32-style reduced-precision matmul on a healthy
        # accelerator drifts ~1e-3 relative, while the failure mode being
        # screened for (garbage/NaN/zero output) is off by orders of
        # magnitude. A too-tight tolerance would silently demote a healthy
        # GPU to CPU.
        ok = bool(torch.allclose(result, reference, rtol=1e-2, atol=1e-2))
        if not ok:
            logger.warning(
                "Device %s FAILED the tensor-op sanity probe (wrong output)", device_name
            )
        return ok
    except Exception as exc:
        logger.warning("Device %s failed the tensor-op probe: %s", device_name, exc)
        return False


def select_device() -> str:
    """``cuda → mps → cpu`` cascade, each candidate gated by ``probe_device``.

    MPS is inert inside Linux containers (never available) — the branch exists
    for the Apple host-process path, where the same app code runs in a native
    venv. CPU is the unconditional floor and is not probed.
    """
    import torch

    candidates = []
    if torch.cuda.is_available():
        candidates.append("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        candidates.append("mps")
    for name in candidates:
        if probe_device(name):
            return name
    return "cpu"


class Diarizer:
    """Pipeline load + single-flight inference + post-processing."""

    def __init__(self) -> None:
        self.model: Any = None
        self.model_loaded = False
        # Default model source: the vendored pipeline baked into the image
        # (offline, no token). An explicit DIARIZER_MODEL_NAME wins; when the
        # vendored config is absent too (bare-host venv runs), fall back to
        # the HF repo id, which needs HF_TOKEN for its gate.
        vendored = os.getenv("VOXINT_VENDORED_PIPELINE", "/app/vendored/config.yaml")
        explicit = os.getenv("DIARIZER_MODEL_NAME")
        if explicit:
            self.model_name = explicit
        elif os.path.exists(vendored):
            self.model_name = vendored
        else:
            self.model_name = "pyannote/speaker-diarization-3.1"
        self.model_is_local = os.path.exists(self.model_name)
        self.hf_token = os.getenv("HF_TOKEN") or None
        self.device_name = "cpu"

        # /healthz identity fields (see docs/gpu-contracts.md); versions
        # resolved at load time so healthz never imports engine packages.
        self.engine = "pyannote.audio"
        self.engine_version: str | None = None
        self.runtime: str | None = "torch"
        self.runtime_version: str | None = None

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

        if not self.model_is_local and not self.hf_token:
            logger.warning(
                "HF_TOKEN not set — the gated %s weights will not download", self.model_name
            )

        logger.info(
            "Loading diarization pipeline: %s (%s)",
            self.model_name,
            "vendored/local" if self.model_is_local else "hugging face",
        )
        start = time.time()
        import pyannote.audio

        self.engine_version = pyannote.audio.__version__
        self.runtime_version = torch.__version__
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
            hint = (
                "corrupt or incomplete vendored files — rebuild/re-pull the image"
                if self.model_is_local
                else "usually an unaccepted HF gate or invalid HF_TOKEN"
            )
            raise RuntimeError(
                f"Pipeline.from_pretrained returned None for {self.model_name} — {hint}"
            )

        device = select_device()
        if device != "cpu":
            self.model.to(torch.device(device))
        self.device_name = resolve_device_name(device)
        if device == "cuda":
            logger.info("Pipeline on GPU: %s", torch.cuda.get_device_name(0))
        else:
            logger.info("Pipeline on %s", self.device_name)

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
