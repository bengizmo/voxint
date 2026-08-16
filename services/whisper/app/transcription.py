"""Whisper transcription core: BatchedInferencePipeline + hallucination soft-tagging.

Model policy: default **large-v2** at int8. large-v3 / large-v3-turbo trade
quiet-audio robustness for speed and produce repetition hallucinations on the
kind of real-world recordings this pipeline exists for. Override via
``WHISPER_MODEL`` at your own risk — the suspect-tagging below is the safety
net either way.

Whisper's quiet-audio failure mode produces runs of repeated tokens
("Class Class Class…") that destroy downstream text analysis. ``detect_repetition``
flags these segments without dropping text — operators decide downstream how to
weight tagged spans.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.backends import TranscribeOptions

if TYPE_CHECKING:
    from app.backends import LegacyFileBackend, SharedWindowsBackend

logger = logging.getLogger(__name__)


class DecodeError(ValueError):
    """Input audio could not be decoded (HTTP 400 invalid_media)."""


def _hip_runtime_loaded() -> bool:
    """True when the process has the HIP runtime library mapped (the CT2 ROCm
    build links it). Read as bytes — non-UTF-8 mapped paths must not raise —
    and swallow OSError: the probe must never fail device detection."""
    try:
        with open("/proc/self/maps", "rb") as maps:
            return b"libamdhip64" in maps.read()
    except OSError:
        return False


def resolve_device_name(device_type: str) -> str:
    """Honest /healthz device reporting: both torch-ROCm and the CTranslate2
    ROCm build masquerade as CUDA (``device="cuda"`` selects the AMD GPU), so
    report ``rocm`` whenever a HIP runtime is actually behind the "cuda"
    label. Two signals, either sufficient: ``torch.version.hip`` when torch is
    present, else the HIP runtime library the loaded CT2 extension links
    (visible in ``/proc/self/maps`` — call this only after the model is
    constructed). The ``-rocm`` image ships no torch at all (the faster-whisper
    1.2.x VAD is onnxruntime-based), so the maps probe is its only signal."""
    if device_type == "cuda":
        try:
            import torch

            if getattr(torch.version, "hip", None):
                return "rocm"
        except (ImportError, OSError):
            # OSError: a broken torch install (missing shared libs) must not
            # take device detection down with it.
            pass
        if _hip_runtime_loaded():
            return "rocm"
    return device_type


# Read once at import — the container env is stable, no hot reload.
# Toggle via WHISPER_REPETITION_TAG=off and restart to disable.
_REPETITION_TAG_ENABLED = os.getenv("WHISPER_REPETITION_TAG", "on").strip().lower() in {
    "on",
    "1",
    "true",
    "yes",
}

_TOKEN_RE = re.compile(r"[\w'\-]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace-tokenization with surrounding punctuation stripped."""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


def detect_repetition(
    text: str,
    *,
    min_repeats: int = 4,
    ratio_cap: float = 0.5,
    max_ngram: int = 5,
    density_min_tokens: int = 20,
) -> tuple[bool, float, str | None]:
    """Flag Whisper hallucination repetition patterns in a transcript segment.

    Returns ``(is_suspect, score, span)`` where:
      * ``is_suspect`` — True if either detection rule fires.
      * ``score`` — highest signal observed in [0, 1]. **The two rules are not
        on the same scale**: rule 1 reports run share of total tokens
        (``run * n / len(tokens)``), rule 2 reports n-gram-share density
        (``top_count / len(ngrams)``). Treat as a severity hint, not a
        comparable confidence.
      * ``span`` — example offending substring, sliced from the original token
        stream and truncated to ~120 chars.

    Rules:
      1. **Run-length**: an n-gram (length 1..``max_ngram``) repeats
         ``min_repeats`` or more times consecutively (non-overlapping). Catches
         the canonical "Class Class Class Class…" hallucination loop.
      2. **Density**: a single n-gram accounts for more than ``ratio_cap`` of
         all n-grams of that length. Only applied when the segment has at
         least ``density_min_tokens`` tokens, so short natural patterns like
         "check check check this" don't trip the rule.

    Empty/short input returns ``(False, 0.0, None)``.
    """
    tokens = _tokenize(text)
    if len(tokens) < min_repeats:
        return False, 0.0, None

    # Both rules: even very long n-grams cap at max_ngram. For rule 1 we need at
    # least min_repeats consecutive non-overlapping copies, so n can't exceed
    # len(tokens) // min_repeats and still fire. Rule 2 reuses the same bound —
    # density at large n is dominated by the same shapes rule 1 catches at
    # smaller n, so the tighter bound costs no real coverage.
    n_max = min(max_ngram, max(1, len(tokens) // min_repeats))

    fired = False
    best_score = 0.0
    best_span: str | None = None

    # Rule 1: walk the n-gram list non-overlappingly. `i` is the start n-gram
    # index (== start token index); we advance by `n` after each comparison so
    # the next candidate starts past the current n-gram, never inside it.
    for n in range(1, n_max + 1):
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            continue
        i = 0
        while i < len(ngrams):
            run = 1
            j = i
            while j + n < len(ngrams) and ngrams[j + n] == ngrams[i]:
                run += 1
                j += n
            if run >= min_repeats:
                fired = True
                share = (run * n) / len(tokens)
                if share > best_score:
                    best_score = share
                    # Slice from the original token stream so the span reflects
                    # actual run length (capped to ~10 reps for readability).
                    span_reps = min(run, 10)
                    best_span = " ".join(tokens[i : i + span_reps * n])
                i = j + n
            else:
                i += 1

    # Rule 2: density on segments long enough that the rule isn't trigger-happy.
    if len(tokens) >= density_min_tokens:
        for n in range(1, n_max + 1):
            ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
            if not ngrams:
                continue
            counts: dict[tuple[str, ...], int] = {}
            for g in ngrams:
                counts[g] = counts.get(g, 0) + 1
            top_gram, top_count = max(counts.items(), key=lambda kv: kv[1])
            density = top_count / len(ngrams)
            if density > ratio_cap:
                fired = True
                if density > best_score:
                    best_score = density
                    best_span = " ".join(top_gram)

    if best_span and len(best_span) > 120:
        best_span = best_span[:117] + "..."
    return fired, round(best_score, 4), best_span


def build_segment_annotation(
    *,
    start_seconds: float,
    end_seconds: float,
    text: str,
    confidence: float | None,
    enabled: bool = _REPETITION_TAG_ENABLED,
) -> tuple[dict[str, Any], bool]:
    """Build a per-segment annotation record, optionally tagging repetition.

    Pure-python helper kept separate from ``transcribe()`` so unit tests can
    exercise the env-flag gating + record shape without a GPU pipeline.
    Returns ``(record, flagged)``.
    """
    record: dict[str, Any] = {
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "text": text,
        "confidence": confidence,
        "suspect": False,
        "suspect_score": None,
        "suspect_span": None,
    }
    flagged = False
    if enabled:
        is_suspect, score, span = detect_repetition(text)
        if is_suspect:
            flagged = True
            record["suspect"] = True
            record["suspect_score"] = score
            record["suspect_span"] = span
    return record, flagged


@dataclass
class TranscriptionOutput:
    """Raw transcription output consumed by the API layer."""

    transcript: str
    language: str
    confidence: float
    duration_seconds: float
    words: list[dict[str, Any]]
    segments: list[dict[str, Any]] = field(default_factory=list)
    suspect_segment_count: int = 0


class WhisperTranscriber:
    """Engine-agnostic facade over a ``WHISPER_ENGINE`` backend.

    ``main.py`` builds one via ``app.backends.create_transcriber`` (which reads
    ``WHISPER_ENGINE``); constructing it directly (no ``backend=``) yields the
    byte-faithful ``ct2-legacy`` engine regardless of env, which is what the
    in-process parity harness wants. The public surface — the ``transcribe``
    signature, the ``/healthz`` identity attributes, and the module-singleton +
    lifespan lifecycle — is unchanged from the pre-seam single-class
    implementation, so callers and ``/healthz`` are byte-compatible.

    Dispatch is by the backend's ``kind`` descriptor: ``legacy_file`` returns a
    fully-assembled ``TranscriptionOutput`` (passed straight through);
    ``shared_windows`` (Slice 2b) will assemble in the shared front layer here.
    """

    def __init__(
        self,
        model_name: str = "large-v2",
        device: str = "cuda",
        compute_type: str = "int8",
        batch_size: int = 16,
        *,
        backend: Any = None,
    ):
        if backend is None:
            # Default engine is the shipped, byte-faithful legacy path,
            # env-independent by design (the factory is the env-driven seam).
            from app.backends.ct2_legacy import Ct2LegacyBackend

            backend = Ct2LegacyBackend(
                model_name=model_name,
                device=device,
                compute_type=compute_type,
                batch_size=batch_size,
            )
        self._backend: LegacyFileBackend | SharedWindowsBackend = backend

    # --- /healthz identity + config, delegated to the selected backend ---
    @property
    def model_name(self) -> str:
        return self._backend.model_name

    @property
    def device(self) -> str:
        return self._backend.device

    @property
    def is_initialized(self) -> bool:
        return self._backend.is_initialized

    @property
    def engine(self) -> str:
        return self._backend.engine

    @property
    def engine_version(self) -> str | None:
        return self._backend.engine_version

    @property
    def runtime(self) -> str | None:
        return self._backend.runtime

    @property
    def runtime_version(self) -> str | None:
        return self._backend.runtime_version

    def load_model(self) -> None:
        self._backend.load_model()

    def cleanup_memory(self) -> None:
        self._backend.cleanup_memory()

    def transcribe(
        self,
        audio_path: str,
        language: str | None = "en",
        initial_prompt: str | None = None,
        vad_filter: bool = True,
    ) -> TranscriptionOutput:
        """Transcribe an audio file, dispatching to the selected backend."""
        options = TranscribeOptions(
            language=language, initial_prompt=initial_prompt, vad_filter=vad_filter
        )
        # Discriminated on the backend's Literal ``kind`` descriptor.
        if self._backend.kind == "legacy_file":
            return self._backend.transcribe_file(audio_path, options)
        if self._backend.kind == "shared_windows":
            raise NotImplementedError(
                "shared_windows front-layer assembly lands in Slice 2b"
            )
        raise ValueError(f"Unknown backend kind {self._backend.kind!r}")
