"""Source recording dates preserve the device's own calendar day."""

import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pytest

from voxint.media import recorded_date
from voxint.media.normalize import NormalizationError
from voxint.media.recorded_date import CREATION_DATE_TAG, parse_creation_date, probe_recorded_on


@pytest.mark.parametrize("value", [
    "2026-10-02T23:30:00-0600", "2026-10-02T23:30:00-06:00",
    "2026-10-02T10:00:00+0000", "2026-10-02T10:00:00+0530",
    "2026-10-02T10:00:00.123456+0530", " 2026-10-02 10:00:00+0530 ",
])
def test_parse_recorded_date(value: str) -> None:
    assert parse_creation_date(value) == date(2026, 10, 2)


@pytest.mark.parametrize("value", [
    "2026-10-02T10:00:00Z", "2026-10-02T10:00:00", "2026-10-02",
    "20261002T10:00:00+0000", "", "  ", "garbage", None, 123, b"date",
    "2026-13-02T10:00:00+0000", "2026-10-02T10:00:00+2400",
    "\uff12\uff10\uff12\uff16-10-02T10:00:00+0000", "2026-10-02T10:00:00.1234567+0000",
])
def test_parse_recorded_date_rejects(value: object) -> None:
    assert parse_creation_date(value) is None


def test_recorded_date_positive_offset_boundary() -> None:
    assert parse_creation_date("2026-10-03T00:30:00+0100") == date(2026, 10, 3)


@pytest.mark.parametrize("payload", [
    None, [], {}, {"format": None}, {"format": []}, {"format": {}},
    {"format": {"tags": None}}, {"format": {"tags": []}},
    {"format": {"tags": {CREATION_DATE_TAG.upper(): "2026-10-02T23:30:00-0600"}}},
    {"format": {"tags": {CREATION_DATE_TAG: 123}}},
])
def test_probe_recorded_date_malformed_payload(
    payload: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(cmd: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
        assert cmd[1:5] == ["-protocol_whitelist", "file", "-format_whitelist",
                            recorded_date._INPUT_FORMATS]
        assert cmd[5:] == ["-v", "error", "-show_entries", f"format_tags={CREATION_DATE_TAG}",
                           "-of", "json", "source.m4a"]
        assert timeout_seconds == 10.0
        return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

    monkeypatch.setattr(recorded_date, "_run", run)
    assert probe_recorded_on(Path("source.m4a")) is None


@pytest.mark.parametrize("failure", ["timeout", "invalid_json", "nonzero"])
def test_probe_recorded_date_failures(failure: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def run(cmd: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
        if failure == "timeout":
            raise NormalizationError("timeout")
        return subprocess.CompletedProcess(cmd, 1 if failure == "nonzero" else 0, "{", "")

    monkeypatch.setattr(recorded_date, "_run", run)
    assert probe_recorded_on(Path("source.m4a")) is None


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)
@pytest.mark.parametrize("kind", ["both", "utc", "wav", "missing", "binary", "z"])
def test_probe_recorded_date_real_media(tmp_path: Path, kind: str) -> None:
    path = tmp_path / ("source.wav" if kind == "wav" else "source.m4a")
    if kind != "missing":
        cmd = ["ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "1"]
        if kind in ("both", "z", "binary"):
            value = "2026-10-02T23:30:00Z" if kind == "z" else "2026-10-02T23:30:00-0600"
            cmd += ["-metadata", f"{CREATION_DATE_TAG}={value}"]
        if kind in ("both", "utc"):
            cmd += ["-metadata", "creation_time=2026-10-03T05:30:00Z"]
        if kind != "wav":
            cmd += ["-movflags", "use_metadata_tags"]
        subprocess.run([*cmd, str(path)], capture_output=True, check=True)
    binary = "/nonexistent/ffprobe" if kind == "binary" else "ffprobe"
    assert probe_recorded_on(path, ffprobe_bin=binary) == (
        date(2026, 10, 2) if kind == "both" else None
    )
