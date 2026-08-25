"""Unit tests for the pure audio-clip extraction core (issue #88)."""

from __future__ import annotations

import io
import struct
import uuid
import wave
from pathlib import Path

import pytest

from voxint.media.clips import (
    ClipBounds,
    ClipBoundsError,
    ClipSourceError,
    clip_idempotency_key,
    clip_relative_path,
    extract_clip,
    read_total_frames,
    resolve_sample_bounds,
)

SR = 16000


def _ramp_wav(n_frames: int, *, channels: int = 1, rate: int = SR, width: int = 2) -> io.BytesIO:
    """A mono s16le WAV whose sample i holds value (i % 32768) — a monotone ramp
    so a frame copy can be checked sample-exact against its start index."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        frames = b"".join(struct.pack("<h", i % 32768) for i in range(n_frames))
        wav.writeframes(frames)
    buf.seek(0)
    return buf


def _samples(path: Path) -> list[int]:
    with wave.open(str(path), "rb") as wav:
        raw = wav.readframes(wav.getnframes())
    return [v for (v,) in struct.iter_unpack("<h", raw)]


# --------------------------------------------------------------------------- #
# resolve_sample_bounds
# --------------------------------------------------------------------------- #


def test_bounds_floor_start_ceil_end() -> None:
    # 1.00003 s -> floor(16000.48) = 16000; 2.00007 s -> ceil(32001.12) = 32002.
    bounds = resolve_sample_bounds(
        1.00003, 2.00007, total_frames=SR * 10, max_clip_frames=SR * 300
    )
    assert bounds.start_sample == 16000
    assert bounds.end_sample == 32002
    assert bounds.frame_count == 16002


def test_bounds_exact_second_boundaries() -> None:
    bounds = resolve_sample_bounds(1.0, 2.0, total_frames=SR * 3, max_clip_frames=SR * 300)
    assert bounds.start_sample == SR
    assert bounds.end_sample == 2 * SR


def test_bounds_tail_within_tolerance_clamps() -> None:
    total = SR * 5
    # end 0.04 s past the tail (< 0.05 tolerance) clamps to total_frames.
    bounds = resolve_sample_bounds(
        4.0, 5.04, total_frames=total, max_clip_frames=SR * 300
    )
    assert bounds.end_sample == total


def test_bounds_tail_beyond_tolerance_rejected() -> None:
    with pytest.raises(ClipBoundsError):
        resolve_sample_bounds(4.0, 5.5, total_frames=SR * 5, max_clip_frames=SR * 300)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_bounds_non_finite_rejected(bad: float) -> None:
    with pytest.raises(ClipBoundsError):
        resolve_sample_bounds(bad, 1.0, total_frames=SR * 5, max_clip_frames=SR * 300)
    with pytest.raises(ClipBoundsError):
        resolve_sample_bounds(0.0, bad, total_frames=SR * 5, max_clip_frames=SR * 300)


def test_bounds_disordered_rejected() -> None:
    with pytest.raises(ClipBoundsError):
        resolve_sample_bounds(2.0, 2.0, total_frames=SR * 5, max_clip_frames=SR * 300)
    with pytest.raises(ClipBoundsError):
        resolve_sample_bounds(3.0, 2.0, total_frames=SR * 5, max_clip_frames=SR * 300)


def test_bounds_negative_start_rejected() -> None:
    with pytest.raises(ClipBoundsError):
        resolve_sample_bounds(-0.1, 1.0, total_frames=SR * 5, max_clip_frames=SR * 300)


def test_bounds_start_past_end_rejected() -> None:
    with pytest.raises(ClipBoundsError):
        resolve_sample_bounds(6.0, 6.5, total_frames=SR * 5, max_clip_frames=SR * 300)


def test_bounds_over_max_rejected() -> None:
    with pytest.raises(ClipBoundsError):
        resolve_sample_bounds(0.0, 200.0, total_frames=SR * 500, max_clip_frames=SR * 60)


def test_bounds_empty_source_rejected() -> None:
    with pytest.raises(ClipSourceError):
        resolve_sample_bounds(0.0, 1.0, total_frames=0, max_clip_frames=SR * 300)


# --------------------------------------------------------------------------- #
# idempotency key + path
# --------------------------------------------------------------------------- #


def test_key_is_deterministic() -> None:
    art = uuid.uuid4()
    ann = uuid.uuid4()
    k1 = clip_idempotency_key(
        normalized_artifact_id=art, annotation_id=ann, start_sample=100, end_sample=200
    )
    k2 = clip_idempotency_key(
        normalized_artifact_id=art, annotation_id=ann, start_sample=100, end_sample=200
    )
    assert k1 == k2
    assert len(k1) == 64


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start_sample": 101},
        {"end_sample": 201},
        {"annotation_id": uuid.uuid4()},
        {"normalized_artifact_id": uuid.uuid4()},
    ],
)
def test_key_changes_on_any_component(kwargs: dict[str, object]) -> None:
    base = {
        "normalized_artifact_id": uuid.UUID(int=1),
        "annotation_id": uuid.UUID(int=2),
        "start_sample": 100,
        "end_sample": 200,
    }
    assert clip_idempotency_key(**base) != clip_idempotency_key(**{**base, **kwargs})  # type: ignore[arg-type]


def test_relative_path_shape_and_determinism() -> None:
    run = uuid.uuid4()
    key = "a" * 64
    p1 = clip_relative_path(run, key)
    p2 = clip_relative_path(run, key)
    assert p1 == p2
    assert p1.startswith(f"artifacts/{run}/clips/")
    assert p1.endswith(".wav")
    assert len(Path(p1).stem) == 32


# --------------------------------------------------------------------------- #
# extract_clip + read_total_frames
# --------------------------------------------------------------------------- #


def test_read_total_frames() -> None:
    assert read_total_frames(_ramp_wav(12345)) == 12345


def test_read_total_frames_rejects_non_pcm_invariant() -> None:
    with pytest.raises(ClipSourceError):
        read_total_frames(_ramp_wav(100, channels=2))
    with pytest.raises(ClipSourceError):
        read_total_frames(_ramp_wav(100, rate=44100))


def test_extract_is_sample_exact(tmp_path: Path) -> None:
    source = _ramp_wav(1000)
    bounds = ClipBounds(start_sample=250, end_sample=750)
    dest = tmp_path / "clip.wav"
    result = extract_clip(source, bounds, dest)

    assert result.frame_count == 500
    assert result.byte_size == 44 + 500 * 2  # header + s16le mono payload
    samples = _samples(dest)
    assert len(samples) == 500
    assert samples[0] == 250  # first kept sample == start index (ramp)
    assert samples[-1] == 749  # last kept sample == end-1 (half-open)


def test_extract_full_span(tmp_path: Path) -> None:
    source = _ramp_wav(64)
    dest = tmp_path / "clip.wav"
    extract_clip(source, ClipBounds(start_sample=0, end_sample=64), dest)
    assert _samples(dest) == list(range(64))


def test_extract_rejects_out_of_range_bounds(tmp_path: Path) -> None:
    source = _ramp_wav(100)
    with pytest.raises(ClipBoundsError):
        extract_clip(source, ClipBounds(start_sample=50, end_sample=200), tmp_path / "c.wav")


def test_extract_leaves_no_temp_file(tmp_path: Path) -> None:
    source = _ramp_wav(100)
    extract_clip(source, ClipBounds(start_sample=0, end_sample=50), tmp_path / "c.wav")
    leftovers = [p.name for p in tmp_path.iterdir() if ".part" in p.name]
    assert leftovers == []


def test_extract_cleans_temp_on_failure(tmp_path: Path) -> None:
    source = _ramp_wav(100)
    with pytest.raises(ClipBoundsError):
        extract_clip(source, ClipBounds(start_sample=0, end_sample=999), tmp_path / "c.wav")
    assert list(tmp_path.iterdir()) == []
