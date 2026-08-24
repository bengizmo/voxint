#!/usr/bin/env python3
"""GPU inference runner for the synthdetect eval harness (issue #144, M1 S2).

Runs INSIDE the pinned eval container (``services/synthdetect/Dockerfile.eval``,
pinned fairseq) with the GPU. It reads a corpus MANIFEST
(``synthdetect_corpus.py``), scores each clip through the reference detector, and
writes a raw-score JSONL JOURNAL the host scorer (``synthdetect_eval.py``)
consumes unchanged. The runner produces measured evidence, never a verdict: raw
scores only, no thresholding, no calibration.

Design (a 3-model consult, codex planner role):

* **The engine seam is narrow.** Everything that determines corpus identity or
  the scored numbers -- canonical-PCM verification, windowing, repeat-padding,
  batching, pooling, journaling, and resume -- is PURE and unit-tested against a
  recording fake engine. Only the fairseq forward pass lives behind
  :class:`Engine`, lazily imported so this module imports without torch/fairseq
  and its orchestration is covered in CI without a GPU or weights.
* **No resampling in the runner.** Corpus audio is canonicalized ONCE at
  acquisition to ``pcm-s16le-mono-16000-v1`` (16 kHz mono signed-16-bit
  little-endian PCM; no dither, normalization, or trim). The manifest ``sha256``
  is the digest of the PCM ``data`` payload bytes only (no WAV header). The
  runner asserts that exact format, hashes the payload, and fails closed on a
  mismatch, so corpus identity never depends on the container's resampler.
* **Fixed score polarity.** Higher raw score means MORE likely synthetic. The Tak
  SSL_Anti-spoofing checkpoint emits column 1 = bona-fide logit, so the runner
  journals its negation; the choice is pinned in the header and unit-tested.
* **Determinism is recorded, not assumed.** The journal header carries the full
  runtime + flags provenance so a later session can ratify tolerances (or, if the
  spike is bit-exact, anchor on determinism) from measured rerun variance.

Subcommands:

* ``run`` -- score a manifest into a journal (``--resume`` continues one).
* ``verify-sources`` -- compute a dated weight receipt (real shas + sizes) from
  the mounted weights, comparing them to the registry pins. This is the S2
  mechanism; the actual freeze (CANDIDATE -> pinned) is a reviewed diff a
  maintainer commits from the receipt in S2b, never an automated rewrite.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from synthdetect_corpus import ClipEntry, Manifest, load_manifest  # noqa: E402
from synthdetect_sources import (  # noqa: E402
    ModelEntry,
    WindowingPolicy,
    assert_runnable,
    get_model,
)

# Must match synthdetect_eval.JOURNAL_SCHEMA_VERSION -- the runner writes the
# journal the scorer reads. A contract test pins the two together.
JOURNAL_SCHEMA_VERSION = 1

# The one canonicalization the corpus and the runner agree on. Corpus audio is
# stored in this exact form at acquisition; the runner refuses anything else.
CANONICALIZATION_ID = "pcm-s16le-mono-16000-v1"
CANONICAL_SAMPLE_RATE = 16000
CANONICAL_SAMPLE_WIDTH = 2  # signed 16-bit
CANONICAL_CHANNELS = 1

# int16 -> float32 scale (upstream convention: divide by 2**15, not 2**15 - 1).
_INT16_SCALE = 32768.0


class InferError(Exception):
    """A fail-closed runner problem (bad audio, identity mismatch, config)."""


# --------------------------------------------------------------------------- #
# 1. Canonical audio: read + verify (no resampling)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CanonicalAudio:
    """A verified canonical-PCM clip: its int16 samples and payload digest."""

    samples: np.ndarray  # dtype int16, shape (n_samples,)
    pcm_sha256: str
    n_samples: int


def _riff_data_chunk_size(path: Path) -> int:
    """Return the declared byte size of the WAV ``data`` chunk (fail-closed).

    The stdlib ``wave`` reader floors an odd-sized ``data`` chunk to whole frames,
    so an orphan trailing byte (a non-canonical or truncated payload) would slip
    past a frame-count check. Reading the RIFF chunk size directly lets the caller
    reject a payload whose declared size is not a whole number of frames.
    """
    with path.open("rb") as fh:
        riff = fh.read(12)
        if len(riff) < 12 or riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise InferError(f"{path}: not a RIFF/WAVE file")
        while True:
            hdr = fh.read(8)
            if len(hdr) < 8:
                raise InferError(f"{path}: no data chunk found")
            chunk_id = hdr[:4]
            size = int.from_bytes(hdr[4:8], "little")
            if chunk_id == b"data":
                return size
            fh.seek(size + (size & 1), 1)  # skip payload plus its RIFF pad byte


def read_canonical_pcm(path: Path) -> CanonicalAudio:
    """Read a canonical-PCM WAV, failing closed on any non-canonical property.

    The file MUST be mono, 16 kHz, signed-16-bit little-endian PCM. The returned
    ``pcm_sha256`` is the sha256 of the raw ``data``-chunk payload only (the same
    bytes the manifest pins), never the whole file, so a re-muxed WAV header can
    never change corpus identity. Fails closed rather than resampling or
    down-mixing, since either would make the digest depend on this container.
    """
    try:
        with contextlib.closing(wave.open(str(path), "rb")) as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            rate = wav.getframerate()
            comp = wav.getcomptype()
            n_frames = wav.getnframes()
            frames = wav.readframes(n_frames)
    except (wave.Error, OSError, EOFError) as exc:
        raise InferError(f"{path}: not a readable PCM WAV: {exc}") from exc

    if comp != "NONE":
        raise InferError(
            f"{path}: compressed WAV ({comp!r}); canonical audio must be uncompressed PCM"
        )
    if channels != CANONICAL_CHANNELS:
        raise InferError(f"{path}: {channels} channels; canonical audio must be mono")
    if rate != CANONICAL_SAMPLE_RATE:
        raise InferError(f"{path}: {rate} Hz; canonical audio must be {CANONICAL_SAMPLE_RATE} Hz")
    if width != CANONICAL_SAMPLE_WIDTH:
        raise InferError(
            f"{path}: {width * 8}-bit; canonical audio must be signed 16-bit "
            f"({CANONICALIZATION_ID})"
        )
    # Cross-check the declared RIFF data-chunk size: wave silently floors an odd
    # chunk to whole frames, so an orphan trailing byte would otherwise pass. A
    # payload that is not a whole number of frames, or shorter than declared, is a
    # corrupt clip, not something to round.
    block_align = channels * width
    declared = _riff_data_chunk_size(path)
    if declared % block_align != 0:
        raise InferError(
            f"{path}: data chunk is {declared} bytes, not a whole number of "
            f"{block_align}-byte frames (orphan byte)"
        )
    if len(frames) != declared or len(frames) != n_frames * block_align:
        raise InferError(
            f"{path}: data payload is truncated ({len(frames)} bytes read, {declared} declared)"
        )
    samples = np.frombuffer(frames, dtype="<i2")
    return CanonicalAudio(
        samples=samples,
        pcm_sha256=hashlib.sha256(frames).hexdigest(),
        n_samples=int(samples.shape[0]),
    )


def verify_clip_sha(entry: ClipEntry, audio: CanonicalAudio) -> None:
    """Raise unless the clip's canonical-PCM digest matches the manifest (fail closed)."""
    if entry.sha256 != audio.pcm_sha256:
        raise InferError(
            f"clip {entry.clip_id!r}: canonical-PCM sha256 {audio.pcm_sha256} does not match the "
            f"manifest sha256 {entry.sha256} (audio changed, or it was not canonicalized to "
            f"{CANONICALIZATION_ID})"
        )


# --------------------------------------------------------------------------- #
# 2. Windowing (pure): source spans -> fixed-width repeat-padded model batches
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WindowPlan:
    """Per-clip window spans over the sample stream, plus the padding record.

    ``spans`` are (start, end) sample indices into the clip; each is prepared to
    the model's fixed input width by :func:`repeat_pad_to`. ``repeat_padded`` is
    True when any span was shorter than the model width and had to be tiled --
    surfaced so a clip scored entirely from repeat-padding is never mistaken for
    a clip with real coverage.
    """

    spans: tuple[tuple[int, int], ...]
    repeat_padded: bool


def model_input_samples(windowing: WindowingPolicy) -> int:
    """The fixed input width the engine batches to, in samples.

    The AASIST front-end fixes its input length (``upstream_window_samples``);
    the runner supports only fixed-width models in M1. An arbitrary-length model
    (``upstream_window_samples`` None, e.g. AntiDeepfake) is a later session, so
    the runner fails closed rather than guessing a width.
    """
    width = windowing.upstream_window_samples
    if width is None:
        raise InferError(
            "arbitrary-length windowing is not supported by the M1 runner "
            "(the model declares upstream_window_samples=None); it is a later session"
        )
    return width


def plan_windows(n_samples: int, windowing: WindowingPolicy, *, mode: str) -> WindowPlan:
    """Plan window spans over a clip of ``n_samples`` (pure).

    ``upstream`` mode reproduces the gate-1 protocol: a single window that is the
    64,600-sample prefix, repeat-padded when the clip is shorter (the exact Tak
    SSL_Anti-spoofing rule -- NOT zero-padding). ``production`` mode chunks the
    clip into ``production_window_s`` windows at ``production_hop_s``; the final
    partial window is kept and repeat-padded so short tails are still scored.
    """
    if n_samples <= 0:
        raise InferError("cannot window an empty clip")
    width = model_input_samples(windowing)
    if mode == "upstream":
        end = min(n_samples, width)
        return WindowPlan(spans=((0, end),), repeat_padded=n_samples < width)
    if mode == "production":
        win = round(windowing.production_window_s * CANONICAL_SAMPLE_RATE)
        hop = round(windowing.production_hop_s * CANONICAL_SAMPLE_RATE)
        if win <= 0 or hop <= 0:
            raise InferError("production window/hop must be positive")
        spans: list[tuple[int, int]] = []
        start = 0
        while start < n_samples:
            spans.append((start, min(start + win, n_samples)))
            start += hop
        # Any span shorter than the model width (short clips, or the final tail)
        # is repeat-padded when prepared; flag it if so.
        padded = any((e - s) < width for s, e in spans)
        return WindowPlan(spans=tuple(spans), repeat_padded=padded)
    raise InferError(f"unknown windowing mode {mode!r} (expected 'upstream' or 'production')")


def repeat_pad_to(samples: np.ndarray, width: int) -> np.ndarray:
    """Crop-or-repeat ``samples`` to exactly ``width`` (the Tak ``pad`` rule).

    A clip at least ``width`` long is truncated to its prefix; a shorter clip is
    tiled (repeated) and truncated to ``width`` -- deterministic, and identical to
    the upstream evaluation's short-clip handling. Zero-padding would change the
    signal the model sees and is deliberately not used.
    """
    n = int(samples.shape[0])
    if n == 0:
        raise InferError("cannot pad an empty span")
    if n >= width:
        return samples[:width]
    reps = -(-width // n)  # ceil division
    return np.tile(samples, reps)[:width]


def build_batch(audio: CanonicalAudio, plan: WindowPlan, width: int) -> np.ndarray:
    """Build the float32 [n_windows, width] batch the engine scores (pure).

    Each span is sliced from the int16 samples, repeat-padded to ``width``, and
    scaled to float32 in [-1, 1) by dividing by 2**15 (upstream convention). This
    is the ONLY place raw samples become model input, so the fake and real engines
    can never diverge on preprocessing.
    """
    rows = [
        repeat_pad_to(audio.samples[start:end], width).astype(np.float32) / _INT16_SCALE
        for start, end in plan.spans
    ]
    return np.stack(rows, axis=0)


def pool_scores(window_scores: np.ndarray, pooling: str) -> float:
    """Pool per-window logits to one clip score (pure).

    ``logit-mean`` (the pinned policy) averages the per-window logits. The pooling
    rule is part of the inference space -- raw-mean vs logit-mean vs max change the
    decision surface -- so an unknown policy fails closed rather than defaulting.
    """
    if window_scores.ndim != 1 or window_scores.shape[0] == 0:
        raise InferError("cannot pool an empty score vector")
    if not np.all(np.isfinite(window_scores)):
        raise InferError("engine returned a non-finite window score")
    if pooling == "logit-mean":
        return float(np.mean(window_scores))
    raise InferError(f"unknown pooling policy {pooling!r} (expected 'logit-mean')")


# --------------------------------------------------------------------------- #
# 3. The engine seam (real fairseq forward pass is the ONLY GPU/weights-bound part)
# --------------------------------------------------------------------------- #
class Engine(Protocol):
    """Scores prepared float32 window batches; higher = more synthetic.

    Implementations receive an already-preprocessed [n_windows, width] float32
    array (all cropping, repeat-padding, and scaling done in the pure layer) and
    return a finite 1-D score per window in the fixed polarity. Nothing about
    decoding, windowing, or pooling belongs here.
    """

    def score_windows(self, batch: np.ndarray) -> np.ndarray: ...


# --------------------------------------------------------------------------- #
# 4. Journal: header identity, write-ahead append, resume
# --------------------------------------------------------------------------- #
# Header keys that define WHAT was computed; a resume must match every one. The
# volatile keys below are excluded so the identity is stable across cold runs
# (the determinism spike compares headers byte-for-byte modulo these).
_VOLATILE_HEADER_KEYS = frozenset(
    {"created_at", "run_id", "host", "journal_path", "execution_identity_sha256"}
)


def selection_sha256(clip_ids: list[str]) -> str:
    """A digest of the ORDERED selected clip-id set (part of the run identity)."""
    return hashlib.sha256("\n".join(clip_ids).encode()).hexdigest()


def execution_identity_sha256(header: dict[str, Any]) -> str:
    """Canonical digest over the immutable header projection (excludes volatile keys).

    This is the single value a ``--resume`` checks so a resumed journal can never
    mix weights, runtime, flags, windowing, scoring semantics, manifest, or
    selection. Computed over sorted-key canonical JSON of every header field
    except the intrinsically volatile ones (timestamps, run id, host, path, and
    the identity field itself).
    """
    projection = {k: v for k, v in header.items() if k not in _VOLATILE_HEADER_KEYS}
    return hashlib.sha256(
        json.dumps(projection, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build_header(
    *,
    model: ModelEntry,
    manifest_sha256: str,
    split: str | None,
    selected_clip_ids: list[str],
    windowing_mode: str,
    runtime: dict[str, Any],
    flags: dict[str, Any],
    weights: dict[str, Any],
    runner_git: dict[str, Any],
    created_at: str,
    run_id: str,
    host: str,
) -> dict[str, Any]:
    """Assemble the journal header (identity + provenance) the scorer reads.

    Carries the five keys the S1 scorer validates (kind, schema_version,
    inference_space, model_id, manifest_sha256) plus the windowing object (with
    its pooling policy), the fixed scoring semantics, the run selection, and the
    full determinism provenance. ``execution_identity_sha256`` is stamped last
    from the immutable projection.
    """
    w = model.windowing
    width = model_input_samples(w)
    header: dict[str, Any] = {
        "kind": "synthdetect_journal",
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "inference_space": model.inference_space,
        "model_id": model.model_id,
        "manifest_sha256": manifest_sha256,
        "canonicalization_id": CANONICALIZATION_ID,
        "model_repo": model.repo,
        "model_repo_commit": model.commit,
        "windowing": {
            "mode": windowing_mode,
            "pooling": w.pooling,
            "sample_rate_hz": w.sample_rate_hz,
            "model_input_samples": width,
            "upstream_window_samples": w.upstream_window_samples,
            "production_window_s": w.production_window_s,
            "production_hop_s": w.production_hop_s,
            "merge_gap_s": w.merge_gap_s,
            "short_clip_rule": "repeat-pad",
        },
        "scoring": {
            "output_column": 1,
            "output_column_meaning": "bona-fide logit (upstream Tak SSL_Anti-spoofing)",
            "journaled_score": "negated column 1",
            "polarity": "higher-is-more-synthetic",
        },
        "selection": {
            "split": split,
            "n_selected": len(selected_clip_ids),
            "selected_clip_ids_sha256": selection_sha256(selected_clip_ids),
        },
        "weights": weights,
        "runtime": runtime,
        "flags": flags,
        "runner_git": runner_git,
        # Volatile: recorded for the audit trail, excluded from the identity.
        "created_at": created_at,
        "run_id": run_id,
        "host": host,
    }
    header["execution_identity_sha256"] = execution_identity_sha256(header)
    return header


def _validate_resume_record(obj: Any, lineno: int) -> str:
    """Validate one completed result line the way the S1 scorer will, returning its id.

    A resume must not treat any JSON object with a ``clip_id`` as done: a line that
    violates the score/skip XOR or carries a bad ``n_windows`` is one the host
    scorer would later reject, so skipping its clip would strand an unscoreable
    journal. This mirrors ``synthdetect_eval._validate_result``'s essentials.
    """
    if not isinstance(obj, dict):
        raise InferError(f"resume journal line {lineno}: result must be an object")
    clip_id = obj.get("clip_id")
    if not isinstance(clip_id, str) or not clip_id:
        raise InferError(f"resume journal line {lineno}: result missing a clip_id")
    has_score = obj.get("raw_score") is not None
    has_skip = obj.get("skip_reason") is not None
    if has_score == has_skip:
        raise InferError(
            f"resume journal line {lineno} ({clip_id}): exactly one of raw_score / skip_reason"
        )
    if has_score:
        score = obj["raw_score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(
            float(score)
        ):
            raise InferError(f"resume journal line {lineno} ({clip_id}): raw_score must be finite")
    n_windows = obj.get("n_windows", 0)
    if not isinstance(n_windows, int) or isinstance(n_windows, bool) or n_windows < 0:
        raise InferError(f"resume journal line {lineno} ({clip_id}): n_windows must be int >= 0")
    return clip_id


def parse_resume_journal(text: str) -> tuple[dict[str, Any], list[str]]:
    """Parse a whole (already de-torned) journal for resume: header + completed ids.

    Unlike the scorer's ``parse_journal`` (which rejects a header-only journal
    because it cannot be scored), a runner resume MUST accept one: the write-ahead
    order is header-then-results, so an interruption right after the header is a
    valid, resumable state. Every result line is validated against the S1 result
    contract (:func:`_validate_resume_record`), so a completed id is never trusted
    from a record the scorer would reject. Fails closed on a missing/broken header,
    a malformed line, or a duplicate clip. Any torn final line is removed by
    :func:`prepare_journal` BEFORE this runs, so a malformed line here is real
    corruption, not an interrupted flush.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise InferError("resume journal is empty (no header)")
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise InferError(f"resume journal header is not valid JSON: {exc}") from exc
    if not isinstance(header, dict) or header.get("kind") != "synthdetect_journal":
        raise InferError("resume journal first line is not a synthdetect_journal header")
    completed: list[str] = []
    seen: set[str] = set()
    for idx, line in enumerate(lines[1:], start=2):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InferError(f"resume journal line {idx}: malformed JSON: {exc}") from exc
        clip_id = _validate_resume_record(obj, idx)
        if clip_id in seen:
            raise InferError(f"resume journal line {idx}: duplicate clip_id {clip_id!r}")
        seen.add(clip_id)
        completed.append(clip_id)
    return header, completed


def _valid_prefix_len(raw: bytes) -> int:
    """The byte length of ``raw`` up to the last newline-terminated line.

    ``JournalWriter`` writes and fsyncs each line as a whole record ending in a
    newline, so an interruption can only leave a final partial line with NO
    trailing newline. Truncating to the last newline therefore drops exactly the
    torn record and nothing complete. A file that already ends in a newline (or is
    empty) needs no truncation.
    """
    if not raw or raw.endswith(b"\n"):
        return len(raw)
    cut = raw.rfind(b"\n")
    return cut + 1 if cut != -1 else 0


def prepare_journal(
    out: Path, header: dict[str, Any], *, resume: bool,
    selected_ids: frozenset[str] | set[str] | None = None,
) -> frozenset[str]:
    """Open-or-create the journal for this run and return the resume skip set.

    A fresh run creates the file exclusively (so a race never appends to another
    run's journal) and writes the header exactly once. A resume physically
    truncates any torn final line off the existing file, then fully re-validates
    the prior header: it recomputes the stored ``execution_identity_sha256`` from
    the header itself (a tampered header keeping an old digest is rejected) and
    requires it to equal both the stored value and this run's identity. Completed
    ids must all belong to ``selected_ids``. An existing file without ``--resume``
    fails closed rather than appending onto stale content.
    """
    if out.exists():
        if not resume:
            raise InferError(
                f"{out} already exists; pass --resume to continue it or remove it first "
                "(refusing to append to an existing journal)"
            )
        raw = out.read_bytes()
        valid = _valid_prefix_len(raw)
        if valid != len(raw):
            # Preserve the torn bytes for forensics, then truncate so the append
            # never concatenates onto a partial record.
            out.with_suffix(out.suffix + ".torn").write_bytes(raw[valid:])
            with out.open("r+b") as fh:
                fh.truncate(valid)
                fh.flush()
                os.fsync(fh.fileno())
        prior_header, completed = parse_resume_journal(raw[:valid].decode("utf-8"))
        stored = prior_header.get("execution_identity_sha256")
        if not isinstance(stored, str) or execution_identity_sha256(prior_header) != stored:
            raise InferError(
                "cannot resume: the existing journal's header does not match its own recorded "
                "execution identity (the header was edited or corrupted)"
            )
        if stored != header["execution_identity_sha256"]:
            raise InferError(
                "cannot resume: the existing journal's execution identity differs from this run "
                "(weights, runtime, flags, windowing, scoring, manifest, or selection changed)"
            )
        if selected_ids is not None:
            unknown = sorted(set(completed) - set(selected_ids))
            if unknown:
                raise InferError(
                    f"cannot resume: journal has completed ids not in this selection: {unknown}"
                )
        return frozenset(completed)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create: fail closed if it appeared between the exists() check and here.
    with out.open("x", encoding="utf-8") as fh:
        line = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False)
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return frozenset()


class JournalWriter:
    """Append-only write-ahead JSONL journal that flushes every complete line.

    Each result is written and flushed (data ``fsync``'d) before the next clip is
    scored, so an interruption leaves at most the final line torn -- which
    :func:`prepare_journal` truncates before any append. The header is written by
    :func:`prepare_journal` (exclusive create) on a fresh journal; this writer only
    ever appends result lines, so it never emits a second header.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._fh = self._path.open("a", encoding="utf-8")

    def write_line(self, obj: dict[str, Any]) -> None:
        line = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self._fh.write(line + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self._fh.close()

    def __enter__(self) -> JournalWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# 5. Orchestration (pure over an injected engine + writer)
# --------------------------------------------------------------------------- #
@dataclass
class ClipOutcome:
    """The journal result for one clip: a raw score XOR a skip reason."""

    clip_id: str
    raw_score: float | None
    skip_reason: str | None
    n_windows: int

    def as_record(self) -> dict[str, Any]:
        rec: dict[str, Any] = {"clip_id": self.clip_id, "n_windows": self.n_windows}
        if self.skip_reason is None:
            rec["raw_score"] = self.raw_score
        else:
            rec["skip_reason"] = self.skip_reason
        return rec


def score_windows_batched(engine: Engine, batch: np.ndarray, batch_size: int) -> np.ndarray:
    """Score a window matrix in fixed-size chunks, preserving order (pure).

    Sending every window of a long production clip in one engine call can exhaust
    VRAM, and the header would then claim a batch size that never shaped the
    computation. Chunk by ``batch_size`` (order preserved), concatenate the
    scores, and fail closed on a non-positive batch size or a per-chunk shape
    mismatch.
    """
    if batch_size <= 0:
        raise InferError(f"batch_size must be positive, got {batch_size}")
    out: list[np.ndarray] = []
    for start in range(0, batch.shape[0], batch_size):
        chunk = batch[start:start + batch_size]
        scores = np.asarray(engine.score_windows(chunk), dtype=np.float64)
        if scores.shape != (chunk.shape[0],):
            raise InferError(
                f"engine returned {scores.shape} scores for a {chunk.shape[0]}-window chunk"
            )
        out.append(scores)
    return np.concatenate(out) if out else np.empty((0,), dtype=np.float64)


def score_clip(
    entry: ClipEntry,
    corpus_root: Path,
    engine: Engine,
    model: ModelEntry,
    *,
    windowing_mode: str,
    batch_size: int = 8,
) -> ClipOutcome:
    """Score one clip end to end (verify -> window -> engine -> pool), fail-closed.

    Reads the canonical PCM, verifies its payload sha against the manifest, plans
    windows, prepares the fixed-width float32 batch, scores it through the engine
    in ``batch_size`` chunks, and pools to one raw score. Any fail-closed problem
    is re-raised (the caller decides whether a per-clip failure stops the run).
    """
    audio = read_canonical_pcm(corpus_root / entry.rel_path)
    verify_clip_sha(entry, audio)
    width = model_input_samples(model.windowing)
    plan = plan_windows(audio.n_samples, model.windowing, mode=windowing_mode)
    batch = build_batch(audio, plan, width)
    window_scores = score_windows_batched(engine, batch, batch_size)
    if window_scores.shape != (len(plan.spans),):
        raise InferError(
            f"clip {entry.clip_id!r}: engine returned {window_scores.shape} scores for "
            f"{len(plan.spans)} windows"
        )
    raw = pool_scores(window_scores, model.windowing.pooling)
    return ClipOutcome(
        clip_id=entry.clip_id, raw_score=raw, skip_reason=None, n_windows=len(plan.spans)
    )


def select_clips(manifest: Manifest, split: str | None) -> list[ClipEntry]:
    """The clips to score, in manifest order, optionally restricted to one split."""
    clips = [c for c in manifest.clips if split is None or c.split == split]
    if not clips:
        raise InferError(f"no clips to score (split={split!r})")
    return clips


def run_inference(
    *,
    clips: list[ClipEntry],
    corpus_root: Path,
    engine: Engine,
    model: ModelEntry,
    header: dict[str, Any],
    writer: JournalWriter,
    windowing_mode: str,
    batch_size: int = 8,
    already_done: frozenset[str] = frozenset(),
    stop_on_error: bool = True,
) -> dict[str, int]:
    """Score ``clips`` into the journal via ``writer`` (pure over injected deps).

    Skips clips already present in ``already_done`` (the resume set). A clip that
    fails closed is re-raised when ``stop_on_error`` (the default: a bad clip is
    information, not something to paper over); otherwise it is journaled as a skip
    with the error text so a whole batch is not lost to one bad file. Returns a
    small counts summary.
    """
    counts = {"scored": 0, "skipped_error": 0, "resumed": 0}
    for entry in clips:
        if entry.clip_id in already_done:
            counts["resumed"] += 1
            continue
        try:
            outcome = score_clip(
                entry, corpus_root, engine, model,
                windowing_mode=windowing_mode, batch_size=batch_size,
            )
            counts["scored"] += 1
        except InferError:
            if stop_on_error:
                raise
            outcome = ClipOutcome(
                clip_id=entry.clip_id,
                raw_score=None,
                skip_reason=f"infer-error: {sys.exc_info()[1]}",
                n_windows=0,
            )
            counts["skipped_error"] += 1
        writer.write_line(outcome.as_record())
    return counts


# --------------------------------------------------------------------------- #
# 6. Determinism provenance capture (lazy torch; injectable for tests)
# --------------------------------------------------------------------------- #
# The two workspace strings torch accepts for deterministic cuBLAS; any other
# value (or none) means cuBLAS may pick a nondeterministic algorithm.
_ALLOWED_CUBLAS = (":4096:8", ":16:8")
DEFAULT_SEED = 0


def parse_device_index(device: str) -> int:
    """Parse a CUDA device string ('cuda' or 'cuda:N') to its index (fail-closed).

    The runner evaluates on CUDA only; a CPU or MPS device would produce scores in
    a different numeric space than the pinned reference, so anything but ``cuda``
    is rejected rather than silently accepted.
    """
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        suffix = device.split(":", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    raise InferError(f"device must be 'cuda' or 'cuda:N', got {device!r}")


def _cuda_driver_version(torch_mod: Any) -> int | None:
    """Best-effort CUDA driver version (a kernel-affecting fact); None if unknown."""
    try:
        return int(torch_mod._C._cuda_getDriverVersion())
    except (AttributeError, RuntimeError, OSError, ValueError):
        return None


def capture_runtime(torch_mod: Any, *, device_index: int, image_digest: str | None,
                    provenance_sha256: str | None, fairseq_version: str | None) -> dict[str, Any]:
    """Read the runtime identity from a torch module (injectable for tests).

    Load-bearing fields (torch/cuda/cudnn versions, device capability, driver
    version) come from torch for the SELECTED device; the image digest and
    provenance sha are passed in from the container because torch cannot know them.
    """
    cuda = bool(torch_mod.cuda.is_available())
    dev_name = torch_mod.cuda.get_device_name(device_index) if cuda else None
    dev_cap = list(torch_mod.cuda.get_device_capability(device_index)) if cuda else None
    cudnn_v = torch_mod.backends.cudnn.version() if cuda else None
    return {
        "torch": str(torch_mod.__version__),
        "cuda": torch_mod.version.cuda,
        "cudnn": cudnn_v,
        "cuda_driver_version": _cuda_driver_version(torch_mod) if cuda else None,
        "fairseq": fairseq_version,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "device_index": device_index,
        "device_name": dev_name,
        "device_capability": dev_cap,
        "image_digest": image_digest,
        "provenance_sha256": provenance_sha256,
    }


def capture_flags(torch_mod: Any, *, batch_size: int, model_eval: bool,
                  seeds: dict[str, int]) -> dict[str, Any]:
    """Snapshot the determinism-relevant torch flags into the journal header.

    Records the values that decide numeric reproducibility (deterministic
    algorithms, cuDNN determinism/benchmark/TF32, matmul TF32 + precision, the
    cuBLAS workspace, the RNG seeds) plus the batch size and the MEASURED eval-mode
    result. The flags are read, not set, here -- :func:`configure_determinism` sets
    them and :func:`seed_everything` sets the seeds.
    """
    return {
        "deterministic_algorithms": bool(torch_mod.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(
            getattr(torch_mod, "is_deterministic_algorithms_warn_only_enabled", lambda: False)()
        ),
        "cudnn_deterministic": bool(torch_mod.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch_mod.backends.cudnn.benchmark),
        "cudnn_allow_tf32": bool(torch_mod.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch_mod.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": str(torch_mod.get_float32_matmul_precision()),
        "dtype": "float32",
        "autocast_enabled": False,
        "autocast_dtype": None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "seeds": seeds,
        "batch_size": batch_size,
        "model_eval": model_eval,
        "inference_mode": True,
    }


def seed_everything(torch_mod: Any, seed: int = DEFAULT_SEED) -> dict[str, int]:
    """Seed Python, NumPy, and torch (CPU + all CUDA), returning the seed map.

    Seeded before the engine is built so any RNG the model touches at load or
    inference is reproducible across the cold-process runs the determinism spike
    compares. The returned map is recorded in the journal flags.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch_mod.manual_seed(seed)
    torch_mod.cuda.manual_seed_all(seed)
    return {"python": seed, "numpy": seed, "torch": seed, "torch_cuda": seed}


def configure_determinism(torch_mod: Any) -> None:
    """Pin torch to the deterministic, TF32-off configuration (fail-closed on env).

    Requires CUBLAS_WORKSPACE_CONFIG to be one of the exact values torch accepts
    for deterministic cuBLAS BEFORE touching CUDA (a merely non-empty but invalid
    value would let cuBLAS pick a nondeterministic algorithm), then disables TF32,
    enables deterministic algorithms without warn-only, and turns cuDNN benchmark
    off. The chosen values are what the journal flags then record.
    """
    cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if cublas not in _ALLOWED_CUBLAS:
        raise InferError(
            f"CUBLAS_WORKSPACE_CONFIG={cublas!r} is not one of {_ALLOWED_CUBLAS}; deterministic "
            "cuBLAS needs an exact value (the eval image bakes ':4096:8'). Refusing to run "
            "non-deterministically."
        )
    torch_mod.backends.cudnn.allow_tf32 = False
    torch_mod.backends.cuda.matmul.allow_tf32 = False
    torch_mod.backends.cudnn.benchmark = False
    torch_mod.backends.cudnn.deterministic = True
    torch_mod.use_deterministic_algorithms(True, warn_only=False)


def runner_git_provenance(repo: Path = REPO) -> dict[str, Any]:
    """The runner's git commit + dirty state (fail-soft to nulls off a repo)."""
    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(repo), *args],
                capture_output=True, text=True, timeout=10, check=True,
            )
            return out.stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return None

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status.strip()),
    }


# --------------------------------------------------------------------------- #
# 7. verify-sources: real-byte weight receipt (the S2 freeze mechanism)
# --------------------------------------------------------------------------- #
def _hash_file(path: Path, *, block_size: int = 1 << 20) -> tuple[str, int]:
    """Stream a file's sha256 and byte count in blocks (never load it whole).

    Weight checkpoints are gigabyte-scale; reading one into memory to hash it can
    OOM the S2b verification host, so hash it a megabyte at a time.
    """
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(block_size), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def compute_weight_receipt(weights_dir: Path, model: ModelEntry) -> dict[str, Any]:
    """Hash the mounted weight files and compare them to the registry pins.

    Produces the dated weight-receipt content a maintainer commits in S2b (real
    sha256 + byte size per file, plus the registry's CANDIDATE/pinned state and a
    match verdict). This never rewrites the registry: freezing CANDIDATE pins is a
    reviewed diff, so an unverified sha can never slip in through an automated edit.
    """
    files: list[dict[str, Any]] = []
    all_present = True
    for w in model.weights:
        path = weights_dir / w.filename
        entry: dict[str, Any] = {
            "filename": w.filename,
            "role": w.role,
            "license_spdx": w.license_spdx,
            "registry_sha256": w.sha256,
            "registry_size_bytes": w.size_bytes,
            "registry_state": "pinned" if w.pinned() else "candidate",
        }
        if not path.is_file():
            entry["present"] = False
            entry["verdict"] = "missing"
            all_present = False
        else:
            digest, size = _hash_file(path)
            entry["present"] = True
            entry["actual_sha256"] = digest
            entry["actual_size_bytes"] = size
            if w.sha256 is None:
                entry["verdict"] = "candidate-measured"
            elif w.sha256 == digest and (w.size_bytes is None or size == w.size_bytes):
                # Mirror _verify_weight_bytes: a pinned file matches only when BOTH
                # sha256 and (when pinned) size agree, so the receipt cannot report
                # "match" for bytes the real load gate would reject on size.
                entry["verdict"] = "match"
            else:
                entry["verdict"] = "MISMATCH"
        files.append(entry)
    return {
        "kind": "synthdetect_weight_receipt",
        "model_id": model.model_id,
        "inference_space": model.inference_space,
        "weights_dir": str(weights_dir),
        "all_present": all_present,
        "weights_pinned": model.weights_pinned(),
        "files": files,
    }


# --------------------------------------------------------------------------- #
# 8. CLI
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LoadedEngine:
    """A built engine plus the provenance only the adapter can measure.

    ``weights`` is the verified {filename: {sha256, size_bytes}} map that went into
    the model; ``model_eval`` is the MEASURED assertion that every module is in
    eval mode after device placement (not a hard-coded hope); ``fairseq_version``
    is the installed runtime version. All three flow into the journal header.
    """

    engine: Engine
    weights: dict[str, Any]
    fairseq_version: str | None
    model_eval: bool


# The verbatim, sha-pinned upstream model definition (SSL_Anti-spoofing model.py).
# Loaded by file path so it never has to be on sys.path as a package, and only
# when the real engine is built (it imports fairseq/torch). Provenance +
# integrity hash live in tools/synthdetect_vendor/provenance.json; a contract
# test binds that hash to the pinned model_repo.commit.
_VENDORED_MODEL_PATH = (
    Path(__file__).resolve().parent / "synthdetect_vendor" / "ssl_antispoofing_model.py"
)

# The upstream SSLModel hard-codes this relative XLS-R checkpoint filename; the
# adapter resolves it by constructing the model with the weights dir as CWD, so
# the vendored file stays byte-identical to upstream.
_XLSR_UPSTREAM_FILENAME = "xlsr2_300m.pt"


@contextlib.contextmanager
def _pushd(target: Path) -> Any:
    """Temporarily change the working directory (Python 3.10 has no contextlib.chdir).

    The eval image is Python 3.10, and the vendored upstream ``SSLModel`` loads
    its XLS-R base from a hard-coded relative path. Constructing the model inside
    this context makes that relative path resolve against the mounted weights dir
    without editing the vendored file.
    """
    prev = os.getcwd()
    os.chdir(target)
    try:
        yield
    finally:
        os.chdir(prev)


def _verify_weight_bytes(
    model: ModelEntry, weights_dir: Path
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Hash each mounted weight and fail closed unless it matches the frozen pin.

    Returns the ``{filename: {sha256, size_bytes}}`` map that flows into the
    journal header and a ``{role: path}`` map for loading. A CANDIDATE (null) pin,
    a missing file, or any sha/size mismatch raises: the engine never runs against
    bytes the registry has not frozen.
    """
    verified: dict[str, Any] = {}
    by_role: dict[str, Path] = {}
    for w in model.weights:
        path = weights_dir / w.filename
        if not path.is_file():
            raise InferError(f"weight file {w.filename!r} not found under {weights_dir}")
        if w.sha256 is None:
            raise InferError(
                f"weight {w.filename!r} is CANDIDATE (sha256=None): freeze the registry pins "
                "(S2b) before running the real engine. Use 'verify-sources' for the receipt."
            )
        digest, size = _hash_file(path)
        if digest != w.sha256:
            raise InferError(
                f"weight {w.filename!r} sha256 mismatch: mounted {digest} != pinned {w.sha256}"
            )
        if w.size_bytes is not None and size != w.size_bytes:
            raise InferError(
                f"weight {w.filename!r} size mismatch: mounted {size} != pinned {w.size_bytes}"
            )
        verified[w.filename] = {"sha256": digest, "size_bytes": size}
        by_role[w.role] = path
    return verified, by_role


class _FairseqAASISTEngine:
    """Real forward pass for w2v2-aasist: prepared windows -> synthetic-ness score.

    Receives an already-preprocessed ``[n_windows, 64600]`` float32 batch (all
    cropping/repeat-padding done in the pure layer), runs the fine-tuned
    wav2vec2-XLS-R + AASIST model, and returns a finite float64 score per window
    in the fixed polarity. The upstream checkpoint emits column 1 = bona-fide
    logit, so the score is its NEGATION (higher = more synthetic). No decoding,
    windowing, or pooling happens here.
    """

    def __init__(self, net: Any, device: Any) -> None:
        self._net = net
        self._device = device

    def score_windows(self, batch: np.ndarray) -> np.ndarray:
        import torch

        if batch.ndim != 2:
            raise InferError(
                f"engine expects a 2-D [n_windows, width] batch, got shape {batch.shape}"
            )
        x = torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32)).to(self._device)
        # inference_mode (not just no_grad) matches what the journal header records
        # under flags.inference_mode; numerically identical to a plain forward.
        with torch.inference_mode():
            out = self._net(x)  # [n_windows, 2]; column 1 = bona-fide logit
            scores = (-out[:, 1]).double().detach().cpu().numpy()
        return np.asarray(scores, dtype=np.float64)


def _load_real_engine(model: ModelEntry, weights_dir: Path, device_index: int) -> LoadedEngine:
    """Lazily import fairseq/torch and build the real engine (GPU/weights-bound).

    Imported here, never at module load, so the pure orchestration above stays
    importable and unit-tested without torch, fairseq, a GPU, or weights. Steps,
    in order (each fails closed): verify every weight sha against the mounted
    bytes; load the vendored upstream model definition; construct it, move the
    WHOLE model to the device, and THEN call eval; assert every module is out of
    training mode (the MEASURED bool returned in ``model_eval``, which flows into
    the journal flags). The XLS-R base is loaded by the upstream ``SSLModel`` from
    a hard-coded relative path, so construction happens with the weights dir as
    CWD; the AASIST checkpoint is loaded strictly (no missing/unexpected keys).
    """
    import importlib.util

    import torch

    verified, by_role = _verify_weight_bytes(model, weights_dir)
    try:
        xlsr_path = by_role["xlsr_ssl_base"]
        aasist_path = by_role["aasist_checkpoint"]
    except KeyError as exc:
        raise InferError(f"registry for {model.model_id} is missing weight role {exc}") from exc
    if xlsr_path.name != _XLSR_UPSTREAM_FILENAME:
        raise InferError(
            f"the vendored SSLModel loads the XLS-R base from the hard-coded relative name "
            f"{_XLSR_UPSTREAM_FILENAME!r}, but the registry names it {xlsr_path.name!r}"
        )

    spec = importlib.util.spec_from_file_location(
        "synthdetect_ssl_antispoofing_model", _VENDORED_MODEL_PATH
    )
    if spec is None or spec.loader is None:
        raise InferError(f"cannot load the vendored model definition at {_VENDORED_MODEL_PATH}")
    ssl_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ssl_mod)

    # Fail closed with a clean InferError (not a raw torch RuntimeError traceback)
    # when the requested GPU is absent or the index is out of range. This is a
    # pre-flight check only: a valid device is unaffected, so it never changes the
    # scored numerics.
    if not torch.cuda.is_available():
        raise InferError(
            "CUDA is not available: the real w2v2-aasist engine requires a GPU. "
            "Run inside the eval container on GPU hardware."
        )
    visible = torch.cuda.device_count()
    if device_index >= visible:
        raise InferError(
            f"CUDA device index {device_index} is out of range: only {visible} device(s) visible "
            f"(pass --device cuda:<idx> for an index < {visible})"
        )
    device = torch.device(f"cuda:{device_index}")
    # Construct with the weights dir as CWD so the upstream SSLModel's hard-coded
    # relative 'xlsr2_300m.pt' resolves to the mounted, sha-verified base.
    with _pushd(weights_dir):
        net = ssl_mod.Model(None, device)
    net = net.to(device)
    # weights_only=False (torch 2.1 default): the checkpoint is a pickled state
    # dict, and its bytes are already sha-verified above, so this loads no
    # untrusted code. strict=True rejects any missing or unexpected key.
    state = torch.load(str(aasist_path), map_location=device, weights_only=False)
    net.load_state_dict(state, strict=True)
    net.eval()

    model_eval = all(not m.training for m in net.modules())
    if not model_eval:
        raise InferError(
            "a module is still in training mode after eval(): the run would be non-deterministic "
            "(dropout / batchnorm active). Refusing to score."
        )

    import fairseq

    fairseq_version = getattr(fairseq, "__version__", None)
    engine = _FairseqAASISTEngine(net, device)
    return LoadedEngine(
        engine=engine, weights=verified, fairseq_version=fairseq_version, model_eval=model_eval
    )


def cmd_run(args: argparse.Namespace) -> int:
    import torch

    model = get_model(args.model_id)
    assert_runnable(model)
    device_index = parse_device_index(args.device)
    manifest_bytes = Path(args.manifest).read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = load_manifest(json.loads(manifest_bytes.decode("utf-8")))
    clips = select_clips(manifest, args.split)
    if args.batch_size <= 0:
        raise InferError(f"--batch-size must be positive, got {args.batch_size}")

    configure_determinism(torch)
    seeds = seed_everything(torch, args.seed)
    loaded = _load_real_engine(model, Path(args.weights_dir), device_index)
    runtime = capture_runtime(
        torch, device_index=device_index, image_digest=os.environ.get("SYNTHDETECT_IMAGE_DIGEST"),
        provenance_sha256=_provenance_sha256(), fairseq_version=loaded.fairseq_version,
    )
    flags = capture_flags(
        torch, batch_size=args.batch_size, model_eval=loaded.model_eval, seeds=seeds,
    )

    header = build_header(
        model=model, manifest_sha256=manifest_sha, split=args.split,
        selected_clip_ids=[c.clip_id for c in clips], windowing_mode=args.windowing,
        runtime=runtime, flags=flags, weights=loaded.weights,
        runner_git=runner_git_provenance(), created_at=_utcnow(), run_id=args.run_id or _utcnow(),
        host=os.uname().nodename,
    )

    out = Path(args.out)
    already = prepare_journal(
        out, header, resume=args.resume, selected_ids=frozenset(c.clip_id for c in clips),
    )

    with JournalWriter(out) as writer:
        counts = run_inference(
            clips=clips, corpus_root=Path(args.corpus_root), engine=loaded.engine, model=model,
            header=header, writer=writer, windowing_mode=args.windowing,
            batch_size=args.batch_size, already_done=already, stop_on_error=not args.skip_errors,
        )
    sys.stdout.write(json.dumps({"ok": True, "journal": str(out), **counts}, sort_keys=True) + "\n")
    return 0


def cmd_verify_sources(args: argparse.Namespace) -> int:
    model = get_model(args.model_id)
    receipt = compute_weight_receipt(Path(args.weights_dir), model)
    sys.stdout.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    # A mismatch is a hard failure (a pinned sha that no longer matches the bytes);
    # a candidate-measured or missing receipt is informational (exit 0).
    if any(f.get("verdict") == "MISMATCH" for f in receipt["files"]):
        return 2
    return 0


def _provenance_sha256() -> str | None:
    path = REPO / "services" / "synthdetect" / "provenance.eval.json"
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _utcnow() -> str:
    # Use timezone.utc, not the 3.11+ datetime alias: this runner executes in the
    # py3.10 eval image, where that alias does not exist. Ruff UP017 would rewrite
    # this to the 3.11-only form, so the file is exempt from UP017 in pyproject.toml
    # and a static guard in test_synthdetect_container.py backs it.
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="synthdetect GPU inference runner (#144, M1 S2)"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _weights_default = os.environ.get("SYNTHDETECT_WEIGHTS_DIR", "/weights")

    p_run = sub.add_parser("run", help="score a manifest into a raw-score journal")
    p_run.add_argument("--manifest", required=True, help="corpus manifest JSON")
    p_run.add_argument(
        "--corpus-root", required=True, help="root the manifest rel_paths resolve under"
    )
    p_run.add_argument("--out", required=True, help="journal path (JSONL, append/resume)")
    p_run.add_argument("--model-id", default="w2v2-aasist")
    p_run.add_argument("--weights-dir", default=_weights_default)
    p_run.add_argument("--split", default=None, help="restrict to one manifest split")
    p_run.add_argument("--windowing", default="upstream", choices=("upstream", "production"))
    p_run.add_argument("--batch-size", type=int, default=8)
    p_run.add_argument("--device", default="cuda", help="cuda or cuda:N")
    p_run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_run.add_argument("--run-id", default=None)
    p_run.add_argument("--resume", action="store_true", help="continue an existing journal")
    p_run.add_argument(
        "--skip-errors", action="store_true",
        help="journal a bad clip as a skip instead of stopping",
    )
    p_run.set_defaults(func=cmd_run)

    p_vs = sub.add_parser(
        "verify-sources", help="compute a real-byte weight receipt from mounted weights"
    )
    p_vs.add_argument("--weights-dir", default=_weights_default)
    p_vs.add_argument("--model-id", default="w2v2-aasist")
    p_vs.set_defaults(func=cmd_verify_sources)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except InferError as exc:
        # A fail-closed runner problem is honest information: a clear message and a
        # non-zero exit, never a stack trace masquerading as a crash. Anything that
        # is NOT an InferError is an unexpected bug and propagates unswallowed.
        sys.stderr.write(f"error: {exc}\n")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
