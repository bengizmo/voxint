"""Audio normalization: guarantee the pipeline's 16 kHz mono WAV invariant.

The GPU service contracts assume normalized input; this module is the only
place that guarantee is manufactured. Output is written to a temporary sibling
and atomically renamed, so a crashed or retried prepare stage can never leave
a half-written file where downstream stages will look for it.
"""

import json
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
_STDERR_LIMIT = 2000


class NormalizationError(Exception):
    """ffmpeg/ffprobe failed or produced non-conforming output. Deterministic
    for a given input — the pipeline's failure lane owns it, not retry."""


@dataclass(frozen=True)
class AudioInfo:
    duration_seconds: float
    sample_rate: int
    channels: int
    codec: str


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:  # missing binary, permissions
        raise NormalizationError(f"failed to execute {cmd[0]}: {exc}") from exc


def probe_audio(path: Path, *, ffprobe_bin: str = "ffprobe") -> AudioInfo:
    """Inspect the first audio stream; raises if the file has none."""
    proc = _run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    if proc.returncode != 0:
        raise NormalizationError(
            f"ffprobe failed on {path}: {proc.stderr[-_STDERR_LIMIT:]}"
        )
    try:
        data = json.loads(proc.stdout)
        stream = data["streams"][0]
        return AudioInfo(
            duration_seconds=float(data["format"]["duration"]),
            sample_rate=int(stream["sample_rate"]),
            channels=int(stream["channels"]),
            codec=str(stream["codec_name"]),
        )
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise NormalizationError(f"no decodable audio stream in {path}: {exc!r}") from exc


def normalize_to_wav(
    source: Path,
    dest: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
) -> AudioInfo:
    """Transcode source to 16 kHz mono pcm_s16le WAV at dest, atomically.

    The finished temporary file is probed and checked against the invariant
    before the rename, so ``dest`` existing implies ``dest`` conforming.
    Returns the normalized file's AudioInfo (its duration is the canonical
    one — trust it over the source container's header).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Unique per attempt: two overlapping attempts (lease expiry edge) must
    # never share a temp file. A crashed attempt's litter is bounded and inert.
    tmp = dest.with_name(f"{dest.name}.{uuid.uuid4().hex}.tmp")
    proc = _run(
        [
            ffmpeg_bin,
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            str(TARGET_CHANNELS),
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            str(tmp),
        ]
    )
    try:
        if proc.returncode != 0:
            raise NormalizationError(
                f"ffmpeg failed on {source}: {proc.stderr[-_STDERR_LIMIT:]}"
            )
        info = probe_audio(tmp, ffprobe_bin=ffprobe_bin)
        if info.sample_rate != TARGET_SAMPLE_RATE or info.channels != TARGET_CHANNELS:
            raise NormalizationError(
                f"normalized output non-conforming: {info.sample_rate} Hz,"
                f" {info.channels} ch (wanted {TARGET_SAMPLE_RATE}/{TARGET_CHANNELS})"
            )
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return info
