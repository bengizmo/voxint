from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

import pytest

from voxint.api.host_metrics import (
    HostMetricsSnapshot,
    _parse_macos_memory,
    _read_cpu_percent,
    _read_disk,
    _read_memory,
    _read_memory_linux,
    _read_memory_macos,
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


# `vm_stat` output captured on an Apple-silicon Mac (16 KiB pages), trimmed.
_VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                     4382.
Pages active:                                 197509.
Pages inactive:                               192405.
Pages speculative:                              4625.
Pages throttled:                                   0.
Pages wired down:                             200538.
Pages purgeable:                                   4.
"Translation faults":                     5048144410.
Pages stored in compressor:                   612345.
"""
_MEMSIZE = "17179869184\n"


def test_parse_macos_memory_uses_free_plus_inactive_as_available() -> None:
    total = 17_179_869_184
    available = (4382 + 192_405) * 16_384

    assert _parse_macos_memory(_MEMSIZE, _VM_STAT) == (total - available, total)


def test_parse_macos_memory_reads_page_size_from_header() -> None:
    vm_stat = _VM_STAT.replace("16384 bytes", "4096 bytes")
    total = 17_179_869_184

    assert _parse_macos_memory(_MEMSIZE, vm_stat) == (
        total - (4382 + 192_405) * 4096,
        total,
    )


@pytest.mark.parametrize(
    ("memsize", "vm_stat"),
    [
        (_MEMSIZE, _VM_STAT.replace("(page size of 16384 bytes)", "")),
        (_MEMSIZE, _VM_STAT.replace("Pages inactive", "Pages idle")),
        ("not a number\n", _VM_STAT),
        ("0\n", _VM_STAT),
        ("1024\n", _VM_STAT),
    ],
    ids=["no-page-size", "no-inactive-line", "bad-memsize", "zero-total", "available-over-total"],
)
def test_parse_macos_memory_rejects_unusable_output(memsize: str, vm_stat: str) -> None:
    with pytest.raises((ValueError, KeyError)):
        _parse_macos_memory(memsize, vm_stat)


def test_read_memory_macos_runs_sysctl_and_vm_stat(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(args: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(args))
        stdout = _MEMSIZE if args[0].endswith("sysctl") else _VM_STAT
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    used, total = _read_memory_macos()

    assert total == 17_179_869_184
    assert used == total - (4382 + 192_405) * 16_384
    assert calls == [("/usr/sbin/sysctl", "-n", "hw.memsize"), ("/usr/bin/vm_stat",)]


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("vm_stat"),
        subprocess.CalledProcessError(1, "sysctl"),
        subprocess.TimeoutExpired("vm_stat", 2.0),
    ],
)
def test_read_memory_macos_unknown_when_a_command_fails(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def fake_run(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert _read_memory_macos() == (None, None)


@pytest.mark.parametrize(
    ("platform", "expected"),
    [("darwin", (1, 2)), ("linux", (3, 4))],
)
def test_read_memory_dispatches_on_platform(
    monkeypatch: pytest.MonkeyPatch, platform: str, expected: tuple[int, int]
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr("voxint.api.host_metrics._read_memory_macos", lambda: (1, 2))
    monkeypatch.setattr("voxint.api.host_metrics._read_memory_linux", lambda: (3, 4))

    assert _read_memory() == expected


@pytest.mark.skipif(sys.platform != "darwin", reason="reads the real macOS counters")
def test_read_memory_macos_on_this_host() -> None:
    used, total = _read_memory_macos()

    assert total is not None and used is not None
    assert 0 < used < total


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
    monkeypatch.setattr("voxint.api.host_metrics._read_memory", lambda: (300, 500))
    monkeypatch.setattr("voxint.api.host_metrics._read_disk", lambda path: (700, 1_000))

    assert collect_host_metrics(Path("/media")) == HostMetricsSnapshot(25, 300, 500, 700, 1_000)


def test_collect_host_metrics_or_empty_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_runtime_error(media_root: Path) -> HostMetricsSnapshot:
        raise RuntimeError

    monkeypatch.setattr("voxint.api.host_metrics.collect_host_metrics", raise_runtime_error)

    assert collect_host_metrics_or_empty(Path("/media")) == HostMetricsSnapshot(
        None, None, None, None, None
    )
