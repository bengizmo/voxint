"""Range parsing and the media-serving gate (path confinement + probe cache)."""

import os
from pathlib import Path

import pytest

from voxint.media import serving
from voxint.media.normalize import AudioInfo, NormalizationError
from voxint.media.serving import (
    ByteRange,
    MediaGate,
    MediaNotServableError,
    RangeNotSatisfiableError,
    parse_range,
)

SIZE = 1000


class TestParseRange:
    def test_no_header_serves_whole_file(self) -> None:
        assert parse_range(None, SIZE) is None

    @pytest.mark.parametrize(
        "header",
        [
            "bites=0-499",  # wrong unit
            "bytes=abc-def",  # not numbers
            "bytes=100",  # no dash
            "bytes=0-99,200-299",  # multipart — legal to ignore
            "bytes=-",  # empty on both sides
        ],
    )
    def test_malformed_or_multipart_ignored(self, header: str) -> None:
        assert parse_range(header, SIZE) is None

    def test_plain_range(self) -> None:
        assert parse_range("bytes=0-499", SIZE) == ByteRange(0, 499)

    def test_open_ended(self) -> None:
        assert parse_range("bytes=900-", SIZE) == ByteRange(900, 999)

    def test_suffix(self) -> None:
        assert parse_range("bytes=-100", SIZE) == ByteRange(900, 999)

    def test_suffix_larger_than_file_clamps_to_start(self) -> None:
        assert parse_range("bytes=-5000", SIZE) == ByteRange(0, 999)

    def test_end_clamped_to_eof(self) -> None:
        assert parse_range("bytes=990-4000", SIZE) == ByteRange(990, 999)

    def test_start_past_eof_is_unsatisfiable(self) -> None:
        with pytest.raises(RangeNotSatisfiableError):
            parse_range("bytes=1000-", SIZE)

    def test_zero_suffix_is_unsatisfiable(self) -> None:
        with pytest.raises(RangeNotSatisfiableError):
            parse_range("bytes=-0", SIZE)

    def test_inverted_range_ignored(self) -> None:
        assert parse_range("bytes=500-100", SIZE) is None

    def test_empty_file_serves_whole(self) -> None:
        assert parse_range("bytes=0-", 0) is None

    def test_length(self) -> None:
        assert ByteRange(0, 499).length == 500


@pytest.fixture()
def probed(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    calls: list[Path] = []

    def fake_probe(path: Path, **kwargs: object) -> AudioInfo:
        calls.append(path)
        return AudioInfo(duration_seconds=1.0, sample_rate=16000, channels=1, codec="pcm_s16le")

    monkeypatch.setattr(serving, "probe_audio", fake_probe)
    return calls


class TestMediaGate:
    def test_serves_validated_file_and_caches_probe(
        self, tmp_path: Path, probed: list[Path]
    ) -> None:
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"x" * 64)
        gate = MediaGate(tmp_path)
        for _ in range(2):
            fh, size = gate.open_for_serving(audio)
            with fh:
                assert size == 64
                assert fh.read(2) == b"xx"  # streaming from the validated handle
        assert len(probed) == 1  # second call hit the cache

    def test_probe_reads_the_served_descriptor(
        self, tmp_path: Path, probed: list[Path]
    ) -> None:
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"x" * 64)
        gate = MediaGate(tmp_path)
        fh, _size = gate.open_for_serving(audio)
        fh.close()
        # The probe target was this process's descriptor, not the pathname —
        # a path swapped on disk after open can't smuggle unprobed bytes.
        assert str(probed[0]).startswith(f"/proc/{os.getpid()}/fd/")

    def test_modified_file_reprobed(self, tmp_path: Path, probed: list[Path]) -> None:
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"x" * 64)
        gate = MediaGate(tmp_path)
        gate.open_for_serving(audio)[0].close()
        audio.write_bytes(b"y" * 128)
        gate.open_for_serving(audio)[0].close()
        assert len(probed) == 2

    def test_path_escape_rejected(self, tmp_path: Path, probed: list[Path]) -> None:
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "outside.wav"
        outside.write_bytes(b"x")
        gate = MediaGate(root)
        with pytest.raises(MediaNotServableError, match="escapes"):
            gate.open_for_serving(root / ".." / "outside.wav")
        assert probed == []

    def test_symlink_escape_rejected(self, tmp_path: Path, probed: list[Path]) -> None:
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "secret.wav"
        outside.write_bytes(b"x")
        link = root / "inside.wav"
        link.symlink_to(outside)
        gate = MediaGate(root)
        with pytest.raises(MediaNotServableError, match="escapes"):
            gate.open_for_serving(link)
        assert probed == []

    def test_missing_file_rejected(self, tmp_path: Path, probed: list[Path]) -> None:
        gate = MediaGate(tmp_path)
        with pytest.raises(MediaNotServableError, match="regular file"):
            gate.open_for_serving(tmp_path / "nope.wav")

    def test_probe_failure_not_cached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"not audio")
        calls: list[Path] = []

        def failing_probe(path: Path, **kwargs: object) -> AudioInfo:
            calls.append(path)
            raise NormalizationError("no audio stream")

        monkeypatch.setattr(serving, "probe_audio", failing_probe)
        gate = MediaGate(tmp_path)
        for _ in range(2):
            with pytest.raises(MediaNotServableError):
                gate.open_for_serving(audio)
        assert len(calls) == 2  # failures never enter the cache
