"""Canonical WAV I/O primitives for the synthdetect S5 PR-2a prepare executor.

Covers the numpy-free read/write/measure helpers the executor materializes clips
with: ``read_canonical_wav_payload``, ``write_canonical_wav``, ``payload_sha_and_count``,
and ``_riff_data_chunk_size``. No ffmpeg and no codec is involved; a canonical WAV is
built directly from bytes so the round trip, the fail-closed gates, and the identity
agreement with ``synthdetect_infer.read_canonical_pcm`` are all CI-testable.
"""

from __future__ import annotations

import hashlib
import struct
import sys
import wave
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import synthdetect_corpus as corpus  # noqa: E402


def _coord_payload(n_samples: int, *, start: int = 0) -> bytes:
    """A coordinate-coded payload: sample i holds its own index (mod 2**16) as s16le.

    A global shift or an off-by-one is detectable because every sample's value equals
    its absolute index, not just an internally consistent pattern.
    """
    return b"".join(
        struct.pack("<h", ((start + i) % 65536) - 32768) for i in range(n_samples)
    )


# --------------------------------------------------------------------------- #
# round trip + identity agreement
# --------------------------------------------------------------------------- #
def test_write_then_read_round_trips(tmp_path: Path) -> None:
    payload = _coord_payload(1000)
    path = tmp_path / "clip.wav"
    corpus.write_canonical_wav(path, payload)
    assert corpus.read_canonical_wav_payload(path) == payload


def test_written_data_chunk_bytes_equal_payload(tmp_path: Path) -> None:
    """The manifest identity is the data-chunk payload; the writer must store it verbatim."""
    payload = _coord_payload(777)
    path = tmp_path / "clip.wav"
    corpus.write_canonical_wav(path, payload)
    # The RIFF data-chunk size equals the payload length exactly (no padding for even).
    assert corpus._riff_data_chunk_size(path) == len(payload)
    sha, count = corpus.payload_sha_and_count(payload)
    assert sha == hashlib.sha256(payload).hexdigest()
    assert count == 777


def test_writer_is_deterministic(tmp_path: Path) -> None:
    payload = _coord_payload(512)
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    corpus.write_canonical_wav(a, payload)
    corpus.write_canonical_wav(b, payload)
    assert a.read_bytes() == b.read_bytes()


def test_identity_matches_read_canonical_pcm(tmp_path: Path) -> None:
    """The numpy-free reader and infer's numpy reader must agree on the digest."""
    infer = pytest.importorskip("synthdetect_infer")
    payload = _coord_payload(2048)
    path = tmp_path / "clip.wav"
    corpus.write_canonical_wav(path, payload)
    ours_sha, ours_count = corpus.payload_sha_and_count(
        corpus.read_canonical_wav_payload(path)
    )
    theirs = infer.read_canonical_pcm(path)
    assert ours_sha == theirs.pcm_sha256
    assert ours_count == theirs.n_samples


# --------------------------------------------------------------------------- #
# payload_sha_and_count fail-closed
# --------------------------------------------------------------------------- #
def test_payload_sha_and_count_rejects_odd_length() -> None:
    with pytest.raises(corpus.CorpusError, match="whole number"):
        corpus.payload_sha_and_count(b"\x00\x01\x02")  # 3 bytes: not frame-aligned


def test_payload_sha_and_count_rejects_empty() -> None:
    with pytest.raises(corpus.CorpusError, match="empty"):
        corpus.payload_sha_and_count(b"")


def test_write_rejects_odd_payload(tmp_path: Path) -> None:
    with pytest.raises(corpus.CorpusError, match="whole number"):
        corpus.write_canonical_wav(tmp_path / "bad.wav", b"\x00\x01\x02")


# --------------------------------------------------------------------------- #
# read_canonical_wav_payload fail-closed gates
# --------------------------------------------------------------------------- #
def test_read_rejects_non_riff(tmp_path: Path) -> None:
    path = tmp_path / "notwav.bin"
    path.write_bytes(b"this is not a riff file at all, padding padding")
    with pytest.raises(corpus.CorpusError):
        corpus.read_canonical_wav_payload(path)


def test_read_rejects_stereo(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(_coord_payload(200))
    with pytest.raises(corpus.CorpusError, match="mono"):
        corpus.read_canonical_wav_payload(path)


def test_read_rejects_wrong_rate(tmp_path: Path) -> None:
    path = tmp_path / "sr.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(_coord_payload(200))
    with pytest.raises(corpus.CorpusError, match="16000 Hz"):
        corpus.read_canonical_wav_payload(path)


def test_read_rejects_wrong_width(tmp_path: Path) -> None:
    path = tmp_path / "w8.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)  # 8-bit
        w.setframerate(16000)
        w.writeframes(b"\x00" * 200)
    with pytest.raises(corpus.CorpusError, match="16-bit"):
        corpus.read_canonical_wav_payload(path)


def test_read_rejects_orphan_trailing_byte(tmp_path: Path) -> None:
    """An odd-sized data chunk (a truncated/corrupt payload) must not silently floor."""
    path = tmp_path / "orphan.wav"
    corpus.write_canonical_wav(path, _coord_payload(100))
    raw = bytearray(path.read_bytes())
    # Append one orphan byte and bump the declared data-chunk size by 1, so the
    # declared size is odd (not a whole number of 2-byte frames).
    # data chunk size lives in the last 4-byte size field before the payload; the
    # simplest reliable mutation is to rewrite via struct at the known 40..44 offset
    # of a 44-byte canonical header.
    data_size = struct.unpack_from("<I", raw, 40)[0]
    struct.pack_into("<I", raw, 40, data_size + 1)
    raw += b"\x7f"
    # also bump the RIFF size at offset 4 to keep the container self-consistent
    riff_size = struct.unpack_from("<I", raw, 4)[0]
    struct.pack_into("<I", raw, 4, riff_size + 1)
    path.write_bytes(bytes(raw))
    with pytest.raises(corpus.CorpusError, match="orphan byte"):
        corpus.read_canonical_wav_payload(path)


def test_riff_data_chunk_size_matches_payload(tmp_path: Path) -> None:
    payload = _coord_payload(333)
    path = tmp_path / "c.wav"
    corpus.write_canonical_wav(path, payload)
    assert corpus._riff_data_chunk_size(path) == len(payload)
