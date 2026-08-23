"""``ct2-legacy`` backend: the shipped whole-file CT2 path, moved verbatim.

This is the ``legacy_file`` strategy — a **mechanical, byte-faithful** move of
the pre-seam ``WhisperTranscriber`` (model load + the two decode branches + the
result-assembly loop). It reproduces the frozen #33 CT2-CPU baseline oracle
exactly; do not refactor or "clean up" the assembly loop here. Shared
post-processing is deduplicated only in Slice 2b, and only after fixture replay
proves the anchor is stable.

The only additions over the pre-seam code are the ``kind`` descriptor and the
fail-closed ``verify_device`` hook — both no-ops for the shipped cpu/cuda/rocm
paths.
"""

import logging
import os
import threading
import time
from typing import Any, Literal

from app.backends import TranscribeOptions
from app.transcription import (
    DecodeError,
    TranscriptionOutput,
    assemble_transcription_output,
    resolve_device_name,
)

logger = logging.getLogger(__name__)

# CTranslate2 device strings the whisper wheel accepts. Anything else (e.g.
# "mps") has no CT2 backend, so requesting it must fail closed rather than let
# CT2 silently run on CPU. The shipped tiers request only these.
_CT2_SUPPORTED_DEVICES = ("cpu", "cuda", "auto")


class Ct2LegacyBackend:
    """faster-whisper wrapper: model load, single-flight inference, annotation."""

    kind: Literal["legacy_file"] = "legacy_file"

    def __init__(
        self,
        model_name: str = "large-v2",
        device: str = "cuda",
        compute_type: str = "int8",
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        # The device requested at construction, before load-time relabelling
        # (cuda -> rocm). verify_device checks this raw value.
        self._requested_device = device

        self.model: Any = None
        self.pipeline: Any = None
        # The model is not thread-safe; inference is single-flight by design.
        self.model_lock = threading.Lock()
        self.is_initialized = False

        # /healthz identity fields (see docs/gpu-contracts.md): the inference
        # engine and the compute runtime it runs on. Versions are resolved at
        # load time so healthz never imports engine packages itself.
        self.engine = "faster-whisper"
        self.engine_version: str | None = None
        self.runtime: str | None = "ctranslate2"
        self.runtime_version: str | None = None

    def verify_device(self) -> None:
        """Fail closed when the requested device has no CT2 backend.

        CTranslate2 exposes no runtime device introspection (unlike torch), so
        the honest check is that the *requested* device is one CT2 accepts —
        an ``mps`` request, for example, would otherwise let CT2 fall back to
        CPU silently. For the shipped cpu/cuda/rocm strings this is a no-op
        assertion. Precedent: pyannote's ``probe_device``
        (services/pyannote/app/diarizer.py).
        """
        if self._requested_device not in _CT2_SUPPORTED_DEVICES:
            raise RuntimeError(
                f"WHISPER_ENGINE=ct2-legacy cannot run on device "
                f"{self._requested_device!r}: CTranslate2 supports "
                f"{_CT2_SUPPORTED_DEVICES}. Refusing a silent CPU fallback."
            )

    def load_model(self) -> None:
        """Load the Whisper model and wrap it in BatchedInferencePipeline."""
        import ctranslate2
        import faster_whisper
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        self.engine_version = faster_whisper.__version__
        self.runtime_version = ctranslate2.__version__

        with self.model_lock:
            if self.model is not None:
                return
            logger.info("Loading Whisper model %s (%s)...", self.model_name, self.compute_type)
            start = time.time()
            # The image bakes the model under WHISPER_DOWNLOAD_ROOT at build
            # time; without passing it, faster-whisper would look in the HF hub
            # cache and re-download at runtime.
            download_root = os.getenv("WHISPER_DOWNLOAD_ROOT") or None
            # WHISPER_REVISION pins the exact HF snapshot at load time (the
            # metal launcher sets it to the revision it pre-downloaded);
            # unset keeps the images' existing latest-in-cache behavior.
            revision = os.getenv("WHISPER_REVISION") or None
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=4,
                num_workers=1,
                download_root=download_root,
                revision=revision,
            )
            self.pipeline = BatchedInferencePipeline(model=self.model)
            # CT2 got the raw device string above; only the reported name is
            # rewritten to the honest label.
            self.device = resolve_device_name(self.device)
            # Fail closed on an unsupported requested device (no-op for the
            # shipped cpu/cuda/rocm strings) before advertising readiness.
            self.verify_device()
            self.is_initialized = True
            logger.info("Model loaded in %.2fs", time.time() - start)

    def transcribe_file(
        self, audio_path: str, options: TranscribeOptions
    ) -> TranscriptionOutput:
        """Transcribe an audio file (whole-file legacy path).

        With ``vad_filter`` (default) the BatchedInferencePipeline runs Silero
        VAD segmentation + batched encoding. Without it, the raw model is used
        directly — for audio where VAD misclassifies speech as silence (noisy
        rooms, low-volume recordings).
        """
        language = options.language
        initial_prompt = options.initial_prompt
        vad_filter = options.vad_filter

        # Probe the container up front: faster-whisper decodes via PyAV deep
        # inside inference, where a decode failure is indistinguishable from an
        # inference failure. The contract wants undecodable input → 400.
        import av

        try:
            with av.open(audio_path) as container:
                if not container.streams.audio:
                    raise DecodeError("No audio stream in file")
        except DecodeError:
            raise
        except Exception as exc:
            raise DecodeError(f"Could not decode audio: {exc}") from exc

        transcribe_kwargs: dict[str, Any] = {}
        if initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt

        anti_hallucination = {
            "condition_on_previous_text": False,
            "temperature": 0.0,
            "compression_ratio_threshold": 2.4,
            "no_speech_threshold": 0.6,
            "hallucination_silence_threshold": 2.0,
            "log_prob_threshold": -1.0,
        }

        with self.model_lock:
            if vad_filter:
                segments_iter, info = self.pipeline.transcribe(
                    audio_path,
                    language=language,
                    batch_size=self.batch_size,
                    word_timestamps=True,
                    **transcribe_kwargs,
                    **anti_hallucination,
                )
            else:
                logger.info("VAD disabled — raw model transcribe (no Silero filtering)")
                segments_iter, info = self.model.transcribe(
                    audio_path,
                    language=language,
                    word_timestamps=True,
                    **transcribe_kwargs,
                    **anti_hallucination,
                )
            # Consume the generator inside the lock (model is not thread-safe).
            segments = list(segments_iter)
            # Detection score only when detection actually ran: faster-whisper
            # fills info.language_probability with a sentinel 1.0 on the forced
            # and non-multilingual branches, which is not an honest score (#124).
            language_probability = (
                info.language_probability
                if language is None and self.model.model.is_multilingual
                else None
            )

        # Shared assembly (dedup'd into the front layer; byte-identity with the
        # frozen oracle is guarded by test_whisper_ct2_legacy_replay.py).
        return assemble_transcription_output(
            segments,
            language=info.language,
            language_probability=language_probability,
            duration_seconds=info.duration,
        )

    def cleanup_memory(self) -> None:
        # Keyed on runtime capability, not the reported device label (which
        # honestly says "rocm" on torch-HIP while the allocator API is still
        # torch.cuda).
        if not self.is_initialized:
            return
        try:
            import torch
        except ImportError:
            # The -rocm image is torch-free; CT2 manages its own device memory.
            return

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
