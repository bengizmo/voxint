"""Real-ffmpeg tests for the normalize invariant (skipped when ffmpeg is absent)."""

import shutil
import subprocess
from pathlib import Path

import pytest

from voxint.media.normalize import (
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    AudioInfo,
    NormalizationError,
    normalize_to_wav,
    probe_audio,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


@pytest.fixture()
def stereo_source(tmp_path: Path) -> Path:
    """A 2-second 44.1 kHz stereo wav with a space in its name."""
    src = tmp_path / "source audio.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-ac",
            "2",
            "-ar",
            "44100",
            str(src),
        ],
        capture_output=True,
        check=True,
    )
    return src


def test_normalize_produces_conforming_wav(stereo_source: Path, tmp_path: Path) -> None:
    dest = tmp_path / "out" / "normalized.wav"
    info = normalize_to_wav(stereo_source, dest)
    assert dest.is_file()
    assert not list(dest.parent.glob("*.tmp"))  # own temp file cleaned up
    assert isinstance(info, AudioInfo)
    assert info.sample_rate == TARGET_SAMPLE_RATE
    assert info.channels == TARGET_CHANNELS
    assert info.codec == "pcm_s16le"
    assert info.duration_seconds == pytest.approx(2.0, abs=0.1)
    # dest itself conforms, not just the returned info
    probed = probe_audio(dest)
    assert (probed.sample_rate, probed.channels) == (TARGET_SAMPLE_RATE, TARGET_CHANNELS)


def test_normalize_is_idempotent_over_partial_output(
    stereo_source: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "normalized.wav"
    # Simulate a crashed earlier attempt: garbage at dest and a stale tmp file.
    dest.write_bytes(b"not a wav")
    dest.with_name(dest.name + ".tmp").write_bytes(b"partial")
    info = normalize_to_wav(stereo_source, dest)
    assert info.sample_rate == TARGET_SAMPLE_RATE
    assert probe_audio(dest).channels == TARGET_CHANNELS


def test_invalid_media_raises_and_leaves_no_dest(tmp_path: Path) -> None:
    src = tmp_path / "garbage.mp3"
    src.write_bytes(b"\x00" * 128)
    dest = tmp_path / "out" / "normalized.wav"
    with pytest.raises(NormalizationError):
        normalize_to_wav(src, dest)
    assert not dest.exists()
    assert not list(dest.parent.glob("*.tmp"))


def test_missing_binary_raises(stereo_source: Path, tmp_path: Path) -> None:
    with pytest.raises(NormalizationError, match="failed to execute"):
        normalize_to_wav(stereo_source, tmp_path / "n.wav", ffmpeg_bin="/nonexistent/ffmpeg")


def test_probe_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(NormalizationError):
        probe_audio(tmp_path / "absent.wav")
