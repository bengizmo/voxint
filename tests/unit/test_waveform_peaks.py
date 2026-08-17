"""The waveform peaks reducer (issue #57): format guards, bucket math, payload.

Everything here is pure — synthetic WAVs written with stdlib ``wave`` into
memory or tmp_path, no DB, no route. The route/caching behavior lives in
``tests/integration/test_peaks_api.py``.
"""

import io
import json
import math
import wave

import numpy as np
import pytest

from voxint.media.peaks import (
    PEAK_BUCKETS,
    PEAKS_VERSION,
    PeaksError,
    PeaksPayload,
    SourceFingerprint,
    compute_peaks,
)


def wav_bytes(
    samples: np.ndarray,
    *,
    rate: int = 16000,
    channels: int = 1,
    sampwidth: int = 2,
) -> io.BytesIO:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        if sampwidth == 2:
            w.writeframes(samples.astype("<i2").tobytes())
        else:
            w.writeframes(samples.astype(np.uint8).tobytes())
    buf.seek(0)
    return buf


class TestFormatGuards:
    def test_stereo_rejected(self) -> None:
        with pytest.raises(PeaksError, match="mono"):
            compute_peaks(wav_bytes(np.zeros(1000, dtype=np.int16), channels=2))

    def test_8bit_rejected(self) -> None:
        with pytest.raises(PeaksError, match="16-bit"):
            compute_peaks(wav_bytes(np.zeros(1000, dtype=np.uint8), sampwidth=1))

    def test_wrong_rate_rejected(self) -> None:
        with pytest.raises(PeaksError, match="16000"):
            compute_peaks(wav_bytes(np.zeros(1000, dtype=np.int16), rate=44100))

    def test_garbage_header_rejected(self) -> None:
        with pytest.raises(PeaksError, match="unreadable"):
            compute_peaks(io.BytesIO(b"not a wav at all, definitely"))

    def test_empty_file_rejected(self) -> None:
        with pytest.raises(PeaksError, match="unreadable"):
            compute_peaks(io.BytesIO(b""))

    def test_zero_frames_rejected(self) -> None:
        with pytest.raises(PeaksError, match="no frames"):
            compute_peaks(wav_bytes(np.zeros(0, dtype=np.int16)))

    def test_truncated_data_rejected(self) -> None:
        # wave returns a SHORT read (no exception) when the data chunk is cut;
        # the reducer must catch the header/data mismatch itself.
        raw = wav_bytes(np.zeros(160_000, dtype=np.int16)).getvalue()
        with pytest.raises(PeaksError, match="truncated"):
            compute_peaks(io.BytesIO(raw[:-50_000]))


class TestReduction:
    def test_silence_reduces_to_zero(self) -> None:
        payload = compute_peaks(wav_bytes(np.zeros(160_000, dtype=np.int16)))
        assert len(payload.peaks) == PEAK_BUCKETS
        assert payload.peaks == [0.0] * PEAK_BUCKETS
        assert payload.duration_seconds == 10.0
        assert payload.sample_rate == 16000
        assert payload.frame_count == 160_000
        assert payload.samples_per_bucket == 80

    @pytest.mark.parametrize("value", [32767, -32768])
    def test_full_scale_is_one(self, value: int) -> None:
        # -32768 is the int16 abs-overflow trap: np.abs on int16 maps it back
        # to -32768; the reducer must cast to int32 first.
        payload = compute_peaks(wav_bytes(np.full(160_000, value, dtype=np.int16)))
        assert max(payload.peaks) == 1.0
        assert min(payload.peaks) in (1.0, pytest.approx(0.99997, abs=1e-3))

    def test_ramp_is_nondecreasing(self) -> None:
        samples = np.linspace(0, 32767, 160_000).astype(np.int16)
        payload = compute_peaks(wav_bytes(samples))
        assert payload.peaks == sorted(payload.peaks)
        assert payload.peaks[-1] == 1.0

    def test_short_file_one_sample_per_bucket(self) -> None:
        payload = compute_peaks(wav_bytes(np.arange(10, dtype=np.int16)))
        assert len(payload.peaks) == 10
        assert payload.samples_per_bucket == 1

    def test_exact_bucket_multiple(self) -> None:
        # 4000 frames / 2000 buckets = exactly 2 samples per bucket, no carry.
        payload = compute_peaks(wav_bytes(np.zeros(2 * PEAK_BUCKETS, dtype=np.int16)))
        assert len(payload.peaks) == PEAK_BUCKETS
        assert payload.samples_per_bucket == 2

    def test_partial_final_bucket(self) -> None:
        # 4001 frames → spb=ceil(4001/2000)=3 → 1334 buckets, last holds 2.
        n = 2 * PEAK_BUCKETS + 1
        samples = np.zeros(n, dtype=np.int16)
        samples[-1] = 16384  # only the final (partial) bucket is nonzero
        payload = compute_peaks(wav_bytes(samples))
        assert payload.samples_per_bucket == 3
        assert len(payload.peaks) == math.ceil(n / 3)
        assert payload.peaks[-1] == 0.5
        assert set(payload.peaks[:-1]) == {0.0}

    def test_three_decimal_rounding(self) -> None:
        payload = compute_peaks(wav_bytes(np.full(100, 12345, dtype=np.int16)))
        assert payload.peaks == [round(12345 / 32768.0, 3)] * len(payload.peaks)

    def test_bucket_isolation_across_chunk_boundaries(self) -> None:
        # A spike in one bucket must not bleed into neighbors regardless of
        # where the chunked reads fall.
        samples = np.zeros(PEAK_BUCKETS * 80, dtype=np.int16)
        samples[40] = 32767  # bucket 0 only
        payload = compute_peaks(wav_bytes(samples))
        assert payload.peaks[0] == 1.0
        assert set(payload.peaks[1:]) == {0.0}


class TestPayload:
    def test_json_shape(self) -> None:
        payload = PeaksPayload(
            duration_seconds=1.5,
            sample_rate=16000,
            frame_count=24000,
            samples_per_bucket=12,
            peaks=[0.0, 0.5, 1.0],
        )
        body = json.loads(payload.to_json_bytes())
        assert body == {
            "version": PEAKS_VERSION,
            "duration": 1.5,
            "sampleRate": 16000,
            "frameCount": 24000,
            "samplesPerBucket": 12,
            "peaks": [0.0, 0.5, 1.0],
        }


class TestSourceFingerprint:
    def test_meta_round_trip(self) -> None:
        fp = SourceFingerprint(size=123, mtime_ns=456)
        assert SourceFingerprint.from_meta(fp.to_meta()) == fp

    @pytest.mark.parametrize(
        "meta", [None, "x", {}, {"size": 1}, {"size": "1", "mtime_ns": 2}]
    )
    def test_malformed_meta_is_none(self, meta: object) -> None:
        assert SourceFingerprint.from_meta(meta) is None
