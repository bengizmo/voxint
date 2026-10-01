"""Shared, numpy-only maths for the voiceclone target-voice pack (#664).

Two maintainer tools import this module:

* ``tools/generate_voiceclone_target_voices.py``: the PEP 723 generator that runs
  Qwen3-TTS VoiceDesign on a rented GPU. It uses the spec validation, the
  post-processing to the Chatterbox VC reference format, and the WAV encoding.
* ``tools/screen_voiceclone_target_voices.py``: the TitaNet collision screen,
  which runs in the Voxint dev venv. It uses the WAV read-back and the
  screening maths.

The generator's environment holds only its own inline pins, so this module must
stay importable with nothing but numpy and the standard library. It never
touches a model, a network or a GPU, and ``tests/unit`` covers it directly.

Post-processing targets Chatterbox VC at commit ``5de7a54a``: the reference is
loaded at 24 kHz mono, truncated to ``DEC_COND_LEN = 240000`` samples, and a
length that is not a multiple of 960 samples (one 25 Hz speech token) makes VC
warn and trim. Every constant comes from the committed spec; the spec is the
only file a human edits.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import math
import re
import wave
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
# Chatterbox VC reference contract (vc.py / s3gen.py at 5de7a54a).
CHATTERBOX_SAMPLE_RATE = 24000
CHATTERBOX_DEC_COND_LEN = 240000
CHATTERBOX_TOKEN_SAMPLES = 960
# Voxint's speaker-match floor (voxint.speakers.matching.MatchingGates.min_cosine).
# The contract test pins this copy to the live value; the generator environment
# cannot import voxint.
VOXINT_MATCH_FLOOR = 0.6
PCM16_FULL_SCALE = 32767

_RELEASE_TAG = re.compile(r"^voiceclone-target-voices-v[1-9][0-9]*$")
_VOICE_ID = re.compile(r"^target-[a-z]$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
# A description names attributes (gender presentation, age band, pitch, pace,
# timbre, delivery). Anything that points at a person or a style of one is out.
_NOT_ATTRIBUTE_ONLY = re.compile(
    r"\b(sounds?\s+like|like\s+an?|in\s+the\s+style\s+of|imitat\w*|impression\s+of|"
    r"resembl\w*|celebrit\w*|famous|voice\s+of)\b",
    re.IGNORECASE,
)
_GENERATOR_KEYS = (
    "model_id",
    "hf_revision",
    "hf_card_url",
    "hf_card_license",
    "hf_card_checked",
    "code_repo",
    "code_commit",
    "code_license",
    "package",
    "weights_sha256",
    "dtype",
    "language",
    "reference_recording_used",
)


class SpecError(ValueError):
    """The committed spec violates an invariant of the pack."""


class CandidateRejected(Exception):
    """A generated candidate cannot become a target (rejected, never padded)."""


class ScreenError(ValueError):
    """Screening inputs are unusable; the screen fails closed."""


# --- spec ----------------------------------------------------------------------


def _req(node: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in node:
        raise SpecError(f"{where}.{key} is missing")
    return node[key]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_num(value: Any) -> bool:
    return (_is_int(value) or isinstance(value, float)) and math.isfinite(value)


def _validate_generator(gen: Mapping[str, Any]) -> None:
    for key in _GENERATOR_KEYS:
        _req(gen, key, "generator")
    if gen["reference_recording_used"] is not False:
        raise SpecError("generator.reference_recording_used must be false")
    if not _SHA1.match(str(gen["hf_revision"])):
        raise SpecError("generator.hf_revision must be a full 40-hex commit")
    if not _SHA1.match(str(gen["code_commit"])):
        raise SpecError("generator.code_commit must be a full 40-hex commit")
    weights = gen["weights_sha256"]
    if not isinstance(weights, dict) or not weights:
        raise SpecError("generator.weights_sha256 must name at least one file")
    for name, digest in weights.items():
        if not isinstance(digest, str) or not _SHA256.match(digest):
            raise SpecError(f"generator.weights_sha256[{name!r}] is not a lowercase sha256")
    if gen["dtype"] != "bfloat16":
        # fp16 produces NaNs on this model; the pack never mixes dtypes.
        raise SpecError("generator.dtype must be bfloat16")


def _validate_voices(voices: Any) -> None:
    if not isinstance(voices, list) or len(voices) != 4:
        raise SpecError("the pack must define exactly four voices")
    seen_ids: set[str] = set()
    seen_seeds: set[int] = set()
    bands: list[str] = []
    for i, voice in enumerate(voices):
        where = f"voices[{i}]"
        vid = _req(voice, "id", where)
        if not isinstance(vid, str) or not _VOICE_ID.match(vid):
            raise SpecError(f"{where}.id {vid!r} must be a neutral id like 'target-a'")
        if vid in seen_ids:
            raise SpecError(f"{where}.id {vid!r} is not unique")
        seen_ids.add(vid)
        desc = _req(voice, "description", where)
        if not isinstance(desc, str) or not desc.strip():
            raise SpecError(f"{where}.description is empty")
        if _NOT_ATTRIBUTE_ONLY.search(desc):
            raise SpecError(f"{where}.description must be attribute-only (no person, no style)")
        band = _req(voice, "pitch_band", where)
        if band not in ("lower", "higher"):
            raise SpecError(f"{where}.pitch_band must be 'lower' or 'higher'")
        bands.append(band)
        seeds = _req(voice, "seeds", where)
        if (
            not isinstance(seeds, list)
            or not seeds
            or not all(_is_int(s) for s in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise SpecError(f"{where}.seeds must be a non-empty list of unique integers")
        for seed in seeds:
            if seed in seen_seeds:
                raise SpecError(f"{where}: seed {seed} is reused across voices")
            seen_seeds.add(seed)
    if bands.count("lower") != 2:
        raise SpecError("pitch_band must split two 'lower' and two 'higher' voices")


def _validate_dsp(dsp: Mapping[str, Any]) -> None:
    where = "dsp"
    if _req(dsp, "sample_rate", where) != CHATTERBOX_SAMPLE_RATE:
        raise SpecError(f"dsp.sample_rate must be {CHATTERBOX_SAMPLE_RATE} (Chatterbox S3GEN_SR)")
    if _req(dsp, "channels", where) != 1:
        raise SpecError("dsp.channels must be 1")
    if _req(dsp, "sample_format", where) != "pcm_s16le":
        raise SpecError("dsp.sample_format must be pcm_s16le")
    if _req(dsp, "dither", where) is not False:
        raise SpecError("dsp.dither must be false")
    if _req(dsp, "align_samples", where) != CHATTERBOX_TOKEN_SAMPLES:
        raise SpecError(f"dsp.align_samples must be {CHATTERBOX_TOKEN_SAMPLES}")
    if _req(dsp, "max_samples", where) != CHATTERBOX_DEC_COND_LEN:
        raise SpecError(
            f"dsp.max_samples must equal {CHATTERBOX_DEC_COND_LEN} (Chatterbox DEC_COND_LEN)"
        )
    min_samples = _req(dsp, "min_samples", where)
    if (
        not _is_int(min_samples)
        or min_samples < 8 * CHATTERBOX_SAMPLE_RATE
        or min_samples > CHATTERBOX_DEC_COND_LEN
        or min_samples % CHATTERBOX_TOKEN_SAMPLES
    ):
        raise SpecError("dsp.min_samples must be a 960-aligned count from 8.0 s to max_samples")
    peak = _req(dsp, "peak_dbfs", where)
    if not _is_num(peak) or not -6.0 <= peak < 0.0:
        raise SpecError("dsp.peak_dbfs must be in [-6.0, 0.0)")
    frame = _req(dsp, "trim_frame_samples", where)
    if not _is_int(frame) or frame <= 0:
        raise SpecError("dsp.trim_frame_samples must be a positive integer")
    for key in ("trim_pad_samples", "fade_samples"):
        value = _req(dsp, key, where)
        if not _is_int(value) or value < 0:
            raise SpecError(f"dsp.{key} must be a non-negative integer")
    threshold = _req(dsp, "trim_threshold_db_below_peak", where)
    if not _is_num(threshold) or threshold <= 0:
        raise SpecError("dsp.trim_threshold_db_below_peak must be positive")


def validate_spec(spec: Mapping[str, Any]) -> None:
    """Raise :class:`SpecError` naming the first violated invariant."""
    if _req(spec, "schema_version", "spec") != SCHEMA_VERSION:
        raise SpecError(f"spec.schema_version must be {SCHEMA_VERSION}")
    tag = _req(spec, "release_tag", "spec")
    if not isinstance(tag, str) or not _RELEASE_TAG.match(tag):
        raise SpecError("spec.release_tag must look like voiceclone-target-voices-vN")
    if _req(spec, "asset_license", "spec") != "CC0-1.0":
        raise SpecError("spec.asset_license must be CC0-1.0")
    _validate_generator(_req(spec, "generator", "spec"))
    text = _req(spec, "text", "spec")
    if not isinstance(text, str) or not text.strip():
        raise SpecError("spec.text is empty")
    sampling = _req(spec, "sampling", "spec")
    for key in ("do_sample", "temperature", "top_k", "top_p", "repetition_penalty"):
        _req(sampling, key, "sampling")
    _validate_dsp(_req(spec, "dsp", "spec"))
    analysis = _req(spec, "analysis", "spec")
    fmin = _req(analysis, "f0_fmin_hz", "analysis")
    fmax = _req(analysis, "f0_fmax_hz", "analysis")
    if not (_is_num(fmin) and _is_num(fmax) and 0 < fmin < fmax):
        raise SpecError("analysis.f0_fmin_hz must be positive and below f0_fmax_hz")
    ScreenRules.from_spec(spec)
    ChatterboxRules.from_spec(spec)
    _validate_voices(_req(spec, "voices", "spec"))


def load_spec(path: Path) -> dict[str, Any]:
    """Read and validate the committed spec."""
    spec = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise SpecError(f"{path} must hold a JSON object")
    validate_spec(spec)
    return spec


@dataclass(frozen=True)
class Dsp:
    sample_rate: int
    trim_frame_samples: int
    trim_threshold_db_below_peak: float
    trim_pad_samples: int
    fade_samples: int
    min_samples: int
    max_samples: int
    align_samples: int
    peak_dbfs: float

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> Dsp:
        d = spec["dsp"]
        return cls(
            sample_rate=d["sample_rate"],
            trim_frame_samples=d["trim_frame_samples"],
            trim_threshold_db_below_peak=float(d["trim_threshold_db_below_peak"]),
            trim_pad_samples=d["trim_pad_samples"],
            fade_samples=d["fade_samples"],
            min_samples=d["min_samples"],
            max_samples=d["max_samples"],
            align_samples=d["align_samples"],
            peak_dbfs=float(d["peak_dbfs"]),
        )


# --- post-processing -----------------------------------------------------------


def trim_silence(audio: np.ndarray, dsp: Dsp) -> np.ndarray:
    """Drop leading and trailing frames quieter than the threshold below the
    loudest frame, keeping ``trim_pad_samples`` of context on each side."""
    frame = dsp.trim_frame_samples
    n_frames = -(-len(audio) // frame)
    padded = np.zeros(n_frames * frame, dtype=np.float64)
    padded[: len(audio)] = audio
    rms = np.sqrt(np.mean(padded.reshape(n_frames, frame) ** 2, axis=1))
    peak = float(rms.max()) if n_frames else 0.0
    if peak <= 1e-9:
        raise CandidateRejected("candidate is silent")
    active = np.flatnonzero(rms >= peak * 10 ** (-dsp.trim_threshold_db_below_peak / 20))
    start = max(0, int(active[0]) * frame - dsp.trim_pad_samples)
    end = min(len(audio), (int(active[-1]) + 1) * frame + dsp.trim_pad_samples)
    return audio[start:end]


def _apply_fades(audio: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return audio
    out = audio.copy()
    ramp = np.linspace(0.0, 1.0, n)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def to_pcm16(audio: np.ndarray) -> np.ndarray:
    """Round-to-nearest PCM16, no dither. Input is expected inside [-1, 1]."""
    scaled: np.ndarray = np.round(np.clip(audio, -1.0, 1.0) * PCM16_FULL_SCALE)
    return scaled.astype(np.int16)


def postprocess(audio: np.ndarray, sample_rate: int, dsp: Dsp) -> np.ndarray:
    """Model output (float, mono, 24 kHz) to a target-ready PCM16 array.

    Order: finite check, silence trim, cap at ``max_samples``, align down to a
    multiple of ``align_samples``, reject below ``min_samples``, edge fades,
    peak normalisation to ``peak_dbfs``, PCM16 quantisation.
    """
    if sample_rate != dsp.sample_rate:
        raise ValueError(f"sample rate {sample_rate} != {dsp.sample_rate}")
    signal = np.asarray(audio, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("audio must be mono (1-D)")
    if not np.all(np.isfinite(signal)):
        raise CandidateRejected("candidate contains non-finite samples")
    signal = trim_silence(signal, dsp)[: dsp.max_samples]
    signal = signal[: len(signal) - len(signal) % dsp.align_samples]
    if len(signal) < dsp.min_samples:
        raise CandidateRejected(
            f"candidate is {len(signal)} samples after trim, shorter than {dsp.min_samples}"
        )
    signal = _apply_fades(signal, dsp.fade_samples)
    peak = float(np.max(np.abs(signal)))
    if peak <= 0.0:
        raise CandidateRejected("candidate is silent")
    signal = signal / peak * 10 ** (dsp.peak_dbfs / 20)
    return to_pcm16(signal)


# --- WAV encode + read-back ----------------------------------------------------


@dataclass(frozen=True)
class WavFormat:
    """Format fields read back from WAV bytes, never from intended parameters."""

    sample_rate: int
    channels: int
    sample_width_bytes: int
    n_samples: int
    duration_seconds: float
    peak_dbfs: float | None


def encode_wav(pcm: np.ndarray, sample_rate: int) -> bytes:
    if pcm.dtype != np.int16 or pcm.ndim != 1:
        raise ValueError("encode_wav takes a 1-D int16 array")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.astype("<i2").tobytes())
    return buf.getvalue()


def read_wav_format(data: bytes) -> WavFormat:
    with wave.open(io.BytesIO(data), "rb") as w:
        channels = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        n = w.getnframes()
        frames = w.readframes(n)
    peak_dbfs: float | None = None
    if width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.int32)
        peak = int(np.max(np.abs(samples))) if samples.size else 0
        if peak > 0:
            peak_dbfs = 20 * math.log10(peak / PCM16_FULL_SCALE)
    return WavFormat(rate, channels, width, n, n / rate if rate else 0.0, peak_dbfs)


def read_wav_pcm16(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("expected 16-bit PCM")
        if w.getnchannels() != 1:
            raise ValueError("expected mono")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.int16)


def format_problems(fmt: WavFormat, dsp: Dsp) -> list[str]:
    """Every way ``fmt`` misses the target format; empty means it conforms."""
    problems: list[str] = []
    if fmt.sample_rate != dsp.sample_rate:
        problems.append(f"sample_rate {fmt.sample_rate} != {dsp.sample_rate}")
    if fmt.channels != 1:
        problems.append(f"channels {fmt.channels} != 1")
    if fmt.sample_width_bytes != 2:
        problems.append(f"sample_width {fmt.sample_width_bytes} bytes != 2")
    if fmt.n_samples < dsp.min_samples:
        problems.append(f"{fmt.n_samples} samples is shorter than {dsp.min_samples}")
    if fmt.n_samples > dsp.max_samples:
        problems.append(f"{fmt.n_samples} samples is longer than {dsp.max_samples}")
    if fmt.n_samples % dsp.align_samples:
        problems.append(f"{fmt.n_samples} samples is not a multiple of {dsp.align_samples}")
    if fmt.peak_dbfs is None:
        problems.append("audio is silent or not PCM16")
    elif fmt.peak_dbfs > dsp.peak_dbfs + 0.01:
        problems.append(f"peak {fmt.peak_dbfs:.3f} dBFS is above {dsp.peak_dbfs}")
    return problems


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- screening maths -------------------------------------------------------------


def unit(vector: np.ndarray) -> np.ndarray | None:
    """L2-normalised float64 copy, or None for a zero or non-finite vector."""
    v = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if not math.isfinite(norm) or norm == 0.0:
        return None
    return v / norm


def centroid(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Unit mean of a speaker's unit clip vectors (equal weight per clip).

    Degenerate members are skipped; no usable member, or members that cancel
    out, fail closed rather than produce a meaningless direction.
    """
    units = [u for u in (unit(v) for v in vectors) if u is not None]
    if not units:
        raise ScreenError("no usable vectors for a centroid")
    total = np.sum(units, axis=0)
    out = unit(total)
    if out is None or float(np.linalg.norm(total)) < 1e-9 * len(units):
        raise ScreenError("centroid members cancel out")
    return out


def _check_unit(vector: np.ndarray, what: str) -> np.ndarray:
    u = unit(vector)
    if u is None:
        raise ScreenError(f"{what} vector is degenerate")
    return u


def cross_speaker_cosines(centroids: Mapping[str, np.ndarray]) -> np.ndarray:
    """Sorted cosines over every pair of distinct real speakers."""
    keys = sorted(centroids)
    if len(keys) < 2:
        raise ScreenError("a null distribution needs at least two speakers")
    units = [_check_unit(centroids[k], f"speaker {k}") for k in keys]
    return np.sort(
        np.array(
            [float(units[i] @ units[j]) for i, j in itertools.combinations(range(len(keys)), 2)]
        )
    )


def nearest_neighbour_cosines(centroids: Mapping[str, np.ndarray]) -> np.ndarray:
    """Per real speaker, the cosine to its nearest other real speaker.

    This is the like-for-like null for a candidate's maximum centroid cosine:
    both are "how close is the nearest real speaker".
    """
    keys = sorted(centroids)
    if len(keys) < 2:
        raise ScreenError("a null distribution needs at least two speakers")
    units = np.stack([_check_unit(centroids[k], f"speaker {k}") for k in keys])
    sims = units @ units.T
    np.fill_diagonal(sims, -np.inf)
    return np.asarray(sims.max(axis=1))


def percentile_rank(value: float, null: np.ndarray) -> float:
    """Percentage of the null distribution at or below ``value``."""
    if null.size == 0:
        raise ScreenError("empty null distribution")
    return 100.0 * float(np.count_nonzero(null <= value)) / null.size


@dataclass(frozen=True)
class ScreenRules:
    reject_at_or_above: float
    swap_headroom: float

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> ScreenRules:
        s = _req(spec, "screening", "spec")
        floor = _req(s, "reject_at_or_above", "screening")
        headroom = _req(s, "swap_headroom", "screening")
        if floor != VOXINT_MATCH_FLOOR:
            raise SpecError(f"screening.reject_at_or_above must be {VOXINT_MATCH_FLOOR}")
        if not _is_num(headroom) or not 0.0 <= headroom < floor:
            raise SpecError("screening.swap_headroom must be in [0, reject_at_or_above)")
        return cls(float(floor), float(headroom))


@dataclass(frozen=True)
class Verdict:
    headroom: float
    rejected: bool
    swap_candidate: bool


def screen_verdict(worst_cosine: float, rules: ScreenRules) -> Verdict:
    """At or above the floor rejects; a pass closer than ``swap_headroom`` to it
    is a swap candidate."""
    headroom = rules.reject_at_or_above - worst_cosine
    rejected = worst_cosine >= rules.reject_at_or_above
    return Verdict(headroom, rejected, not rejected and headroom < rules.swap_headroom)


@dataclass(frozen=True)
class ScreenResult:
    max_centroid_cosine: float
    max_centroid_speaker: str
    max_clip_cosine: float
    max_clip_speaker: str
    max_clip_id: str
    headroom: float
    rejected: bool
    swap_candidate: bool
    centroid_cosines: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def screen_candidate(
    vector: np.ndarray,
    pool: Mapping[str, Sequence[tuple[str, np.ndarray]]],
    centroids: Mapping[str, np.ndarray],
    rules: ScreenRules,
) -> ScreenResult:
    """Score one candidate against every real speaker's centroid and clips."""
    cand = _check_unit(vector, "candidate")
    if set(pool) != set(centroids) or not pool:
        raise ScreenError("pool speakers and centroid speakers differ (or are empty)")
    best_c = (-math.inf, "")
    best_clip = (-math.inf, "", "")
    per_speaker: dict[str, float] = {}
    for speaker in sorted(pool):
        cos_c = float(cand @ _check_unit(centroids[speaker], f"speaker {speaker}"))
        per_speaker[speaker] = cos_c
        if cos_c > best_c[0]:
            best_c = (cos_c, speaker)
        for clip_id, clip_vec in pool[speaker]:
            cos_k = float(cand @ _check_unit(clip_vec, f"clip {clip_id}"))
            if cos_k > best_clip[0]:
                best_clip = (cos_k, speaker, clip_id)
    verdict = screen_verdict(max(best_c[0], best_clip[0]), rules)
    return ScreenResult(
        max_centroid_cosine=best_c[0],
        max_centroid_speaker=best_c[1],
        max_clip_cosine=best_clip[0],
        max_clip_speaker=best_clip[1],
        max_clip_id=best_clip[2],
        headroom=verdict.headroom,
        rejected=verdict.rejected,
        swap_candidate=verdict.swap_candidate,
        centroid_cosines=per_speaker,
    )


def pairwise_matrix(vectors: Mapping[str, np.ndarray]) -> dict[str, dict[str, float]]:
    units = {k: _check_unit(v, k) for k, v in vectors.items()}
    keys = sorted(units)
    return {a: {b: float(units[a] @ units[b]) for b in keys} for a in keys}


def select_finalists(
    shortlist: Mapping[str, Sequence[str]], vectors: Mapping[str, np.ndarray]
) -> dict[str, str]:
    """One candidate per slot, minimising the largest pairwise cosine (that is,
    maximising the minimum pairwise distance). Ties go to the lexicographically
    smallest choice so the selection is deterministic."""
    slots = sorted(shortlist)
    owner: dict[str, str] = {}
    for slot in slots:
        if not shortlist[slot]:
            raise ScreenError(f"slot {slot} has an empty shortlist")
        for cand in shortlist[slot]:
            if cand not in vectors:
                raise ScreenError(f"slot {slot} lists unknown candidate {cand}")
            if owner.setdefault(cand, slot) != slot:
                raise ScreenError(f"candidate {cand} is listed in more than one slot")
    units = {k: _check_unit(vectors[k], k) for k in owner}
    best: tuple[float, tuple[str, ...]] | None = None
    for combo in itertools.product(*(sorted(shortlist[s]) for s in slots)):
        worst = max(
            (float(units[a] @ units[b]) for a, b in itertools.combinations(combo, 2)),
            default=-1.0,
        )
        if best is None or worst < best[0]:
            best = (worst, combo)
    assert best is not None  # every slot is non-empty, so product is non-empty
    return dict(zip(slots, best[1], strict=True))


@dataclass(frozen=True)
class ChatterboxRules:
    target_cosine_floor: float
    sources: int
    min_sources_reached: int

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> ChatterboxRules:
        c = _req(spec, "chatterbox", "spec")
        floor = _req(c, "target_cosine_floor", "chatterbox")
        sources = _req(c, "sources", "chatterbox")
        reached = _req(c, "min_sources_reached", "chatterbox")
        if not _is_num(floor) or not 0.0 < floor < 1.0:
            raise SpecError("chatterbox.target_cosine_floor must be in (0, 1)")
        if not _is_int(sources) or sources <= 0:
            raise SpecError("chatterbox.sources must be a positive integer")
        if not _is_int(reached) or not 0 < reached <= sources:
            raise SpecError("chatterbox.min_sources_reached must be in [1, sources]")
        return cls(float(floor), sources, reached)


def reached_count(cosines: Sequence[float], floor: float) -> int:
    return sum(1 for c in cosines if c >= floor)


def slot_needs_swap(cosines: Sequence[float], rules: ChatterboxRules) -> bool:
    """The pre-registered swap rule: a finalist reached by fewer held-out
    sources than the Gate 1 built-in baseline is swapped."""
    if len(cosines) != rules.sources:
        raise ScreenError(f"expected {rules.sources} source cosines, got {len(cosines)}")
    return reached_count(cosines, rules.target_cosine_floor) < rules.min_sources_reached
