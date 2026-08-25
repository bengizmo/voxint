"""Attributed audio-clip extraction (issue #88).

An operator extracts a short, sample-accurate audio clip of exactly what was
said, cut from the normalized 16 kHz mono PCM ``normalized.wav`` and attributed
to a ``word_range`` annotation. This module is the pure, broker-free,
subprocess-free core worth unit-testing in isolation:

- :func:`resolve_sample_bounds` converts validated finite seconds to explicit
  integer sample bounds (half-open ``[start, end)``) with honest rounding.
- :func:`clip_idempotency_key` derives the content-addressed cache key.
- :func:`clip_relative_path` derives the deterministic on-disk name.
- :func:`extract_clip` frame-copies the span with the stdlib :mod:`wave`,
  writing a conforming WAV via a temp sibling and an atomic rename.

Extraction, not editing (no fades/trim/resample): the source is already the
canonical 16 kHz mono s16le timeline, so a straight PCM frame copy is
sample-exact by construction, needs no ffmpeg, and has an a-priori bounded
output size. ffmpeg ``atrim=start_sample:end_sample`` is the documented fallback
if the source ever stops being a plain PCM WAV (it never does by the
normalize.py contract).
"""

from __future__ import annotations

import hashlib
import os
import uuid
import wave
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from math import isfinite
from pathlib import Path
from typing import BinaryIO

from voxint.media.normalize import TARGET_CHANNELS, TARGET_SAMPLE_RATE

# s16le: the normalized-WAV invariant manufactured by normalize.py. Two bytes
# per mono frame, so an N-frame clip is 2*N payload bytes + a 44-byte header.
SAMPLE_WIDTH_BYTES = 2
# Cache-key + on-disk-name schema version. Bump only on a change that would make
# an existing clip's bytes no longer match its key (e.g. a different crop rule).
CLIP_KEY_VERSION = "v1"


class ClipError(Exception):
    """Base for clip-extraction failures. Deterministic for a given input."""


class ClipBoundsError(ClipError):
    """Requested seconds are non-finite, disordered, out of range, or too long.

    A client/anchor problem, mapped to 422 by the route; never a transient."""


class ClipSourceError(ClipError):
    """The normalized source is missing or not the expected 16 kHz mono PCM WAV."""


@dataclass(frozen=True)
class ClipBounds:
    """Validated integer sample bounds, half-open ``[start_sample, end_sample)``."""

    start_sample: int
    end_sample: int

    @property
    def frame_count(self) -> int:
        return self.end_sample - self.start_sample


@dataclass(frozen=True)
class ClipFile:
    """Outcome of a completed extraction."""

    frame_count: int
    byte_size: int


def resolve_sample_bounds(
    start_seconds: float,
    end_seconds: float,
    total_frames: int,
    *,
    max_clip_frames: int,
    sample_rate: int = TARGET_SAMPLE_RATE,
    tail_tolerance_seconds: float = 0.05,
) -> ClipBounds:
    """Convert finite annotation seconds to exact half-open sample bounds.

    ``start`` floors and ``end`` ceils (via :class:`~decimal.Decimal`, so binary
    float drift can't drop a boundary sample), conservatively covering the
    annotated interval. An ``end`` overrunning the recording by no more than
    ``tail_tolerance_seconds`` is clamped to ``total_frames`` (the same slack
    ``playback.py`` allows between a Whisper segment end and the WAV duration);
    any larger overrun is rejected. ``total_frames`` must come from the WAV
    header, not a stored ``duration_seconds``.
    """
    for label, value in (("start", start_seconds), ("end", end_seconds)):
        if not isinstance(value, (int, float)) or not isfinite(float(value)):
            raise ClipBoundsError(f"{label}_seconds must be finite, got {value!r}")
    if end_seconds <= start_seconds:
        raise ClipBoundsError(
            f"end_seconds ({end_seconds}) must be greater than start_seconds "
            f"({start_seconds})"
        )
    if start_seconds < 0:
        raise ClipBoundsError(f"start_seconds must be >= 0, got {start_seconds}")
    if total_frames <= 0:
        raise ClipSourceError(f"source has no frames (total_frames={total_frames})")

    rate = Decimal(sample_rate)
    start_sample = int(
        (Decimal(str(start_seconds)) * rate).to_integral_value(rounding=ROUND_FLOOR)
    )
    end_sample = int(
        (Decimal(str(end_seconds)) * rate).to_integral_value(rounding=ROUND_CEILING)
    )

    # Tail slack: an end within tolerance of the recording's end is clamped to
    # the exact frame count; a genuine overrun is a bounds error.
    tail_slack_frames = int(
        (Decimal(str(tail_tolerance_seconds)) * rate).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
    if end_sample > total_frames:
        if end_sample <= total_frames + tail_slack_frames:
            end_sample = total_frames
        else:
            raise ClipBoundsError(
                f"end_seconds ({end_seconds}) runs past the recording "
                f"({total_frames / sample_rate:.3f}s) beyond tolerance"
            )
    if start_sample >= total_frames:
        raise ClipBoundsError(
            f"start_seconds ({start_seconds}) is at or past the recording end"
        )
    if end_sample <= start_sample:
        raise ClipBoundsError("clip collapses to zero frames after rounding")
    if end_sample - start_sample > max_clip_frames:
        raise ClipBoundsError(
            f"clip is {end_sample - start_sample} frames, over the "
            f"{max_clip_frames}-frame ({max_clip_frames / sample_rate:.0f}s) limit"
        )
    return ClipBounds(start_sample=start_sample, end_sample=end_sample)


def clip_idempotency_key(
    *,
    normalized_artifact_id: uuid.UUID | str,
    annotation_id: uuid.UUID | str,
    start_sample: int,
    end_sample: int,
) -> str:
    """Content-addressed cache key for one clip.

    A clip's bytes are a pure function of the source audio and the integer
    sample bounds, so the key is derived from those alone. The normalized
    artifact id pins the source version (a prepare rerun mints a fresh
    ``preprocessed_audio`` row, so its clips get new keys); the annotation id
    keeps clips per-annotation. Deliberately NOT keyed on the annotation's
    mutable text/``source_text_hash``: a text-only edit reuses the same audio,
    while a re-anchor changes the sample bounds and so makes a new clip.
    """
    payload = (
        f"clip:{CLIP_KEY_VERSION}:{normalized_artifact_id}:{annotation_id}:"
        f"{start_sample}:{end_sample}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def clip_relative_path(run_id: uuid.UUID | str, idempotency_key: str) -> str:
    """Deterministic media-root-relative path for a clip.

    ``artifacts/{run_id}/clips/{digest}.wav`` where ``digest`` is the first 32
    hex chars of the key's sha256. Deterministic so a crash between the atomic
    rename and the DB insert self-heals: a retry with the same key writes the
    same path.
    """
    digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()[:32]
    return f"artifacts/{run_id}/clips/{digest}.wav"


def read_total_frames(source: BinaryIO) -> int:
    """Frame count of the normalized source, validating the PCM WAV invariant.

    ``source`` must be an open, seekable binary handle (e.g. the descriptor
    ``MediaGate`` already opened and confined). Position is restored to 0.
    """
    try:
        source.seek(0)
        with wave.open(source, "rb") as wav:
            _require_pcm_invariant(wav)
            frames = wav.getnframes()
    except wave.Error as exc:
        raise ClipSourceError(f"source is not a readable PCM WAV: {exc}") from exc
    finally:
        source.seek(0)
    return frames


def extract_clip(source: BinaryIO, bounds: ClipBounds, dest_path: Path) -> ClipFile:
    """Frame-copy ``bounds`` from ``source`` into a conforming WAV at ``dest_path``.

    Writes a randomized temp sibling in the destination directory, verifies the
    written frame count/format, ``fsync``s, then atomically ``os.replace``s into
    place, so a crash or retry can never leave a half-written clip where the
    resolver will look. ``source`` is an open, seekable, confined PCM WAV handle.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_name(f"{dest_path.name}.{uuid.uuid4().hex}.part")
    try:
        source.seek(0)
        with wave.open(source, "rb") as reader:
            _require_pcm_invariant(reader)
            total = reader.getnframes()
            if not (0 <= bounds.start_sample < bounds.end_sample <= total):
                raise ClipBoundsError(
                    f"bounds [{bounds.start_sample}, {bounds.end_sample}) escape "
                    f"the source ({total} frames)"
                )
            reader.setpos(bounds.start_sample)
            frames = reader.readframes(bounds.frame_count)
        expected_bytes = bounds.frame_count * TARGET_CHANNELS * SAMPLE_WIDTH_BYTES
        if len(frames) != expected_bytes:
            raise ClipSourceError(
                f"short read: got {len(frames)} bytes, expected {expected_bytes}"
            )
        with open(tmp_path, "wb") as raw:
            with wave.open(raw, "wb") as writer:
                writer.setnchannels(TARGET_CHANNELS)
                writer.setsampwidth(SAMPLE_WIDTH_BYTES)
                writer.setframerate(TARGET_SAMPLE_RATE)
                writer.writeframes(frames)
            raw.flush()
            os.fsync(raw.fileno())
        _verify_written_clip(tmp_path, bounds.frame_count)
        byte_size = tmp_path.stat().st_size
        os.replace(tmp_path, dest_path)
        return ClipFile(frame_count=bounds.frame_count, byte_size=byte_size)
    finally:
        tmp_path.unlink(missing_ok=True)


def _require_pcm_invariant(wav: wave.Wave_read) -> None:
    if (
        wav.getnchannels() != TARGET_CHANNELS
        or wav.getframerate() != TARGET_SAMPLE_RATE
        or wav.getsampwidth() != SAMPLE_WIDTH_BYTES
    ):
        raise ClipSourceError(
            "source is not 16 kHz mono s16le "
            f"(channels={wav.getnchannels()}, rate={wav.getframerate()}, "
            f"sampwidth={wav.getsampwidth()})"
        )


def _verify_written_clip(path: Path, expected_frames: int) -> None:
    with wave.open(str(path), "rb") as wav:
        _require_pcm_invariant(wav)
        if wav.getnframes() != expected_frames:
            raise ClipSourceError(
                f"written clip has {wav.getnframes()} frames, expected "
                f"{expected_frames}"
            )
