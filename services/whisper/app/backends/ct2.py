"""``ct2`` backend: shared-VAD CT2 decode (Slice 2b of #33).

This module defines the ``shared_windows`` boundary types (``SpeechWindow``,
``RawResult``) and the real ``Ct2Backend``. The split is deliberate: the shared
front (``transcription.py``) owns VAD, packing, packed->source time restoration,
confidence, repetition tagging and assembly; this backend does ONLY the raw
batched CT2 forward on the packed windows the front hands it (so a future mlx
backend consumes identical windows).

The batched VAD decode (``decode_windows``) reproduces faster-whisper 1.2.1's
own ``BatchedInferencePipeline.transcribe`` ``vad_filter=True`` branch exactly —
per-window ``feature_extractor(chunk)[..., :-1]`` -> ``pad_or_trim`` ->
``np.stack``, ordered ``forward`` calls of ``batch_size`` windows each with
``last_speech_timestamp`` threading, and ``_batched_segments_generator``'s
``Segment``/``Word`` materialization (global ids, three-decimal segment
rounding). The choice of ``forward`` over the public
``transcribe(clip_timestamps=...)`` path was made by measurement (both are
byte-exact on the comparison fixtures; ``forward`` feeds integer-exact audio and
avoids the seconds->sample float round-trip — see #33 Slice 2b).

The non-VAD path (``transcribe_raw``) is a DIFFERENT algorithm
(``WhisperModel.transcribe`` — the sequential 30 s-seek loop where
``hallucination_silence_threshold=2.0`` and no_speech skipping are LIVE and
``without_timestamps=False``); the front dispatches on ``vad_filter`` and owns
assembly for both modes.

Selecting ``WHISPER_ENGINE=ct2`` runs this backend; it stays a numerics gate
(the self-parity gate proves ``ct2 ≈ ct2-legacy`` to ≤0.5pp pooled WER per vad
mode) and never silently degrades to ``ct2-legacy``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from app.backends import TranscribeOptions
from app.transcription import resolve_device_name

logger = logging.getLogger(__name__)

# CTranslate2 device strings the whisper wheel accepts (mirrors ct2_legacy):
# anything else (e.g. "mps") has no CT2 backend, so requesting it must fail
# closed rather than let CT2 silently run on CPU.
_CT2_SUPPORTED_DEVICES = ("cpu", "cuda", "auto")

SAMPLING_RATE = 16000


@dataclass(frozen=True)
class SpeechWindow:
    """One packed decode window produced by the shared front-layer VAD.

    ``collect_chunks`` concatenates *non-contiguous* speech intervals into a
    single packed decode window. Time restoration is NOT done per window from a
    piecewise offset map — faster-whisper restores against the *global* speech
    chunk list (silence accumulates across the whole file). So each window
    carries faster-whisper's own ``chunks_metadata`` dict (``offset`` on the
    packed timeline, ``duration``, and the source ``segments`` that were packed
    into it); the front restores absolute source time via
    ``restore_speech_timestamps`` against the plan's global chunk list, never a
    hand-rolled per-window map. Validated against faster-whisper 1.2.1 by the
    VADPlan prototype gate.
    """

    # Packed 16 kHz PCM for this window (mono float32), speech intervals only.
    audio: Any
    # faster-whisper's chunks_metadata entry for this packed window:
    # {"offset": packed-timeline seconds, "duration": seconds,
    #  "segments": [source clip dicts in integer samples]}.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawResult:
    """Window-relative decode output crossing the backend -> front boundary.

    ``segments`` are faster-whisper ``Segment`` objects with *packed-timeline*
    timestamps (VAD path) — the front lifts them to absolute source time via
    ``restore_speech_timestamps`` against the plan's global speech-chunk list,
    then assembles the ``TranscriptionOutput``. On the non-VAD path
    (``transcribe_raw``) the segments are already on the original timeline and
    the front assembles them without restoration. ``duration`` is the original
    file duration (raw path only; the VAD path uses the plan's duration).
    """

    segments: list[Any] = field(default_factory=list)
    language: str | None = None
    duration: float = 0.0


class Ct2Backend:
    """Shared-window CT2 backend: batched VAD decode + raw sequential decode."""

    kind: Literal["shared_windows"] = "shared_windows"

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

        # /healthz identity fields (see docs/gpu-contracts.md).
        self.engine = "faster-whisper"
        self.engine_version: str | None = None
        self.runtime: str | None = "ctranslate2"
        self.runtime_version: str | None = None

    def verify_device(self) -> None:
        """Fail closed when the requested device has no CT2 backend.

        Identical policy to ``Ct2LegacyBackend.verify_device``: CTranslate2
        exposes no runtime device introspection, so the honest check is that
        the *requested* device is one CT2 accepts.
        """
        if self._requested_device not in _CT2_SUPPORTED_DEVICES:
            raise RuntimeError(
                f"WHISPER_ENGINE=ct2 cannot run on device "
                f"{self._requested_device!r}: CTranslate2 supports "
                f"{_CT2_SUPPORTED_DEVICES}. Refusing a silent CPU fallback."
            )

    def load_model(self) -> None:
        """Load the Whisper model and wrap it in BatchedInferencePipeline.

        Byte-identical load to ``Ct2LegacyBackend.load_model`` — same
        ``WhisperModel`` construction (cpu_threads, num_workers, download_root,
        revision) and the same ``BatchedInferencePipeline`` wrapper — so the two
        engines share a numerics baseline; only the decode dispatch differs.
        """
        import ctranslate2
        import faster_whisper
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        self.engine_version = faster_whisper.__version__
        self.runtime_version = ctranslate2.__version__

        with self.model_lock:
            if self.model is not None:
                return
            logger.info(
                "Loading Whisper model %s (%s)...", self.model_name, self.compute_type
            )
            start = time.time()
            download_root = os.getenv("WHISPER_DOWNLOAD_ROOT") or None
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
            self.device = resolve_device_name(self.device)
            self.verify_device()
            self.is_initialized = True
            logger.info("Model loaded in %.2fs", time.time() - start)

    def decode_windows(
        self, windows: list[SpeechWindow], options: TranscribeOptions
    ) -> RawResult:
        """Batched CT2 forward over pre-VAD'd packed windows (one transaction).

        Reproduces faster-whisper 1.2.1's ``transcribe`` ``vad_filter=True``
        decode (transcribe.py:463-576) using the front's packed windows: the
        segments come back in *packed* time for the front to restore against the
        plan's global speech-chunk list. Empty ``windows`` (no speech) still
        runs faster-whisper's language detection on the dummy feature so the
        returned language matches the legacy no-speech behavior exactly.
        """
        import numpy as np
        from faster_whisper.audio import pad_or_trim
        from faster_whisper.tokenizer import Tokenizer
        from faster_whisper.transcribe import (
            Segment,
            TranscriptionOptions,
            Word,
            get_suppressed_tokens,
        )

        with self.model_lock:
            # Isolate from any aborted/partially-consumed prior invocation; the
            # pipeline only resets this at normal generator exhaustion.
            self.pipeline.last_speech_timestamp = 0.0
            try:
                raw_features = [
                    self.model.feature_extractor(window.audio)[..., :-1]
                    for window in windows
                ]

                # Language: mirror transcribe.py:471-505. With language=None on a
                # multilingual model, detect on the unpadded features + the
                # dummy -1.5 feature (matches legacy, incl. the no-speech case).
                language = options.language
                if language is None:
                    if not self.model.model.is_multilingual:
                        language = "en"
                    else:
                        language, _prob, _all = self.model.detect_language(
                            features=np.concatenate(
                                [
                                    *raw_features,
                                    np.full(
                                        (self.model.model.n_mels, 1),
                                        -1.5,
                                        dtype="float32",
                                    ),
                                ],
                                axis=1,
                            ),
                            language_detection_segments=1,
                            language_detection_threshold=0.5,
                        )
                elif not self.model.model.is_multilingual and language != "en":
                    language = "en"

                tokenizer = Tokenizer(
                    self.model.hf_tokenizer,
                    self.model.model.is_multilingual,
                    task="transcribe",
                    language=language,
                )

                # Exactly transcribe.py:518-553 — the batched path's pinned
                # silent defaults (without_timestamps=True, max_initial_
                # timestamp=0.0, condition_on_previous_text=False,
                # hallucination_silence_threshold=None). Do NOT unify with the
                # raw path's defaults.
                fw_options = TranscriptionOptions(
                    beam_size=5,
                    best_of=5,
                    patience=1,
                    length_penalty=1,
                    repetition_penalty=1,
                    no_repeat_ngram_size=0,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    compression_ratio_threshold=2.4,
                    temperatures=[0.0],
                    initial_prompt=options.initial_prompt or None,
                    prefix=None,
                    suppress_blank=True,
                    suppress_tokens=get_suppressed_tokens(tokenizer, [-1]),
                    prepend_punctuations="\"'“¿([{-",
                    append_punctuations="\"'.。,，!！?？:：”)]}、",  # noqa: RUF001 (byte-faithful to fw 1.2.1)
                    max_new_tokens=None,
                    hotwords=None,
                    word_timestamps=True,
                    hallucination_silence_threshold=None,
                    condition_on_previous_text=False,
                    clip_timestamps=[],
                    prompt_reset_on_temperature=0.5,
                    multilingual=False,
                    without_timestamps=True,
                    max_initial_timestamp=0.0,
                )

                features = (
                    np.stack([pad_or_trim(feature) for feature in raw_features])
                    if raw_features
                    else []
                )
                metadata = [window.metadata for window in windows]

                # Reproduce _batched_segments_generator (transcribe.py:580-617):
                # ordered batches through forward, global segment ids, three-
                # decimal segment rounding, Word(**word) materialization.
                segments: list[Any] = []
                seg_idx = 0
                for i in range(0, len(features), self.batch_size):
                    results = self.pipeline.forward(
                        features[i : i + self.batch_size],
                        tokenizer,
                        metadata[i : i + self.batch_size],
                        fw_options,
                    )
                    for result in results:
                        for segment in result:
                            seg_idx += 1
                            segments.append(
                                Segment(
                                    seek=segment["seek"],
                                    id=seg_idx,
                                    text=segment["text"],
                                    start=round(segment["start"], 3),
                                    end=round(segment["end"], 3),
                                    words=[Word(**word) for word in segment["words"]],
                                    tokens=segment["tokens"],
                                    avg_logprob=segment["avg_logprob"],
                                    no_speech_prob=segment["no_speech_prob"],
                                    compression_ratio=segment["compression_ratio"],
                                    temperature=fw_options.temperatures[0],
                                )
                            )
                return RawResult(segments=segments, language=language)
            finally:
                self.pipeline.last_speech_timestamp = 0.0

    def transcribe_raw(
        self, audio_path: str, options: TranscribeOptions
    ) -> RawResult:
        """Non-VAD sequential decode (``WhisperModel.transcribe``).

        The ``vad_filter=False`` algorithm — byte-identical to
        ``Ct2LegacyBackend``'s raw branch (ct2_legacy.py:182-190): the raw model
        runs its own 30 s-seek loop where ``hallucination_silence_threshold=2.0``
        and no_speech skipping are LIVE and ``without_timestamps=False``. The
        segments are already on the original timeline; the front assembles them
        without restoration. The undecodable-input probe lives in the front.
        """
        logger.info("VAD disabled — raw model transcribe (no Silero filtering)")

        transcribe_kwargs: dict[str, Any] = {}
        if options.initial_prompt:
            transcribe_kwargs["initial_prompt"] = options.initial_prompt

        anti_hallucination = {
            "condition_on_previous_text": False,
            "temperature": 0.0,
            "compression_ratio_threshold": 2.4,
            "no_speech_threshold": 0.6,
            "hallucination_silence_threshold": 2.0,
            "log_prob_threshold": -1.0,
        }

        with self.model_lock:
            segments_iter, info = self.model.transcribe(
                audio_path,
                language=options.language,
                word_timestamps=True,
                **transcribe_kwargs,
                **anti_hallucination,
            )
            segments = list(segments_iter)

        return RawResult(
            segments=segments, language=info.language, duration=info.duration
        )

    def cleanup_memory(self) -> None:
        if not self.is_initialized:
            return
        try:
            import torch
        except ImportError:
            # The -rocm image is torch-free; CT2 manages its own device memory.
            return

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
