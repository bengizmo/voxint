from __future__ import annotations

import io
import os
import shutil
from collections import namedtuple
from pathlib import Path

import pytest

from voxint.api.host_metrics import (
    HostMetricsSnapshot,
    _read_cpu_percent,
    _read_disk,
    _read_memory_linux,
    collect_host_metrics,
    collect_host_metrics_or_empty,
)


def test_read_cpu_percent_normal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "getloadavg", lambda: (2.0, 1.0, 0.5))
    monkeypatch.setattr(os, "cpu_count", lambda: 4)

    assert _read_cpu_percent() == 50


def test_read_cpu_percent_clamped_to_100(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "getloadavg", lambda: (20.0, 10.0, 5.0))
    monkeypatch.setattr(os, "cpu_count", lambda: 4)

    assert _read_cpu_percent() == 100


@pytest.mark.parametrize("cpu_count", [None, 0])
def test_read_cpu_percent_zero_cores(
    monkeypatch: pytest.MonkeyPatch, cpu_count: int | None
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: cpu_count)

    assert _read_cpu_percent() is None


def test_read_cpu_percent_no_getloadavg(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_os_error() -> tuple[float, float, float]:
        raise OSError

    monkeypatch.setattr(os, "getloadavg", raise_os_error)
    monkeypatch.setattr(os, "cpu_count", lambda: 4)

    assert _read_cpu_percent() is None


def test_read_memory_normal(monkeypatch: pytest.MonkeyPatch) -> None:
    contents = """MemTotal:       16384000 kB
MemFree:         1000000 kB
MemAvailable:    4096000 kB
Buffers:          100000 kB
"""
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: io.StringIO(contents))

    assert _read_memory_linux() == ((16_384_000 - 4_096_000) * 1024, 16_384_000 * 1024)


def test_read_memory_missing_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    contents = "MemTotal:       16384000 kB\nMemFree:         1000000 kB\n"
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: io.StringIO(contents))

    assert _read_memory_linux() == (None, None)


def test_read_memory_file_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_file_not_found(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr("builtins.open", raise_file_not_found)

    assert _read_memory_linux() == (None, None)


def test_read_disk_normal(monkeypatch: pytest.MonkeyPatch) -> None:
    DiskUsage = namedtuple("DiskUsage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda path: DiskUsage(1_000, 600, 400))

    assert _read_disk(Path("/media")) == (600, 1_000)


def test_read_disk_path_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_file_not_found(path: Path) -> None:
        raise FileNotFoundError

    monkeypatch.setattr(shutil, "disk_usage", raise_file_not_found)

    assert _read_disk(Path("/missing")) == (None, None)


def test_collect_host_metrics_assembles_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("voxint.api.host_metrics._read_cpu_percent", lambda: 25)
    monkeypatch.setattr("voxint.api.host_metrics._read_memory_linux", lambda: (300, 500))
    monkeypatch.setattr("voxint.api.host_metrics._read_disk", lambda path: (700, 1_000))

    assert collect_host_metrics(Path("/media")) == HostMetricsSnapshot(25, 300, 500, 700, 1_000)


def test_collect_host_metrics_or_empty_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_runtime_error(media_root: Path) -> HostMetricsSnapshot:
        raise RuntimeError

    monkeypatch.setattr("voxint.api.host_metrics.collect_host_metrics", raise_runtime_error)

    assert collect_host_metrics_or_empty(Path("/media")) == HostMetricsSnapshot(
        None, None, None, None, None
    )
