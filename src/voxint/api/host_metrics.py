"""Host-level CPU / memory / disk metrics for the status page."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HostMetricsSnapshot:
    cpu_percent: int | None
    memory_used_bytes: int | None
    memory_total_bytes: int | None
    disk_used_bytes: int | None
    disk_total_bytes: int | None


def _read_cpu_percent() -> int | None:
    try:
        cpu_count = os.cpu_count()
        if not cpu_count:
            return None
        percent = round(100 * os.getloadavg()[0] / cpu_count)
        return max(0, min(100, percent))
    except Exception:
        return None


def _read_memory_linux() -> tuple[int | None, int | None]:
    try:
        values: dict[str, int] = {}
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            for line in meminfo:
                name, separator, value = line.partition(":")
                if separator and name in {"MemTotal", "MemAvailable"}:
                    values[name] = int(value.split()[0]) * 1024
        total = values["MemTotal"]
        available = values["MemAvailable"]
        return total - available, total
    except Exception:
        return None, None


_MACOS_COMMAND_TIMEOUT_SECONDS = 2.0
_VM_STAT_PAGE_SIZE = re.compile(r"page size of (\d+) bytes")
_VM_STAT_COUNT = re.compile(r"^(?P<name>[^:]+):\s+(?P<pages>\d+)\.?$")


def _run_macos_command(*args: str) -> str:
    return subprocess.run(
        args,
        capture_output=True,
        check=True,
        text=True,
        timeout=_MACOS_COMMAND_TIMEOUT_SECONDS,
    ).stdout


def _parse_macos_memory(memsize: str, vm_stat: str) -> tuple[int, int]:
    """Return (used, total) bytes from `sysctl -n hw.memsize` and `vm_stat` output.

    Available memory is free plus inactive pages, the macOS counterpart of
    Linux's MemAvailable (the same split psutil uses), so "used" means the
    same thing on both platforms: total minus what could be handed out now.
    """
    total = int(memsize.strip())
    page_size_match = _VM_STAT_PAGE_SIZE.search(vm_stat)
    if page_size_match is None:
        raise ValueError("vm_stat output has no page size")
    page_size = int(page_size_match.group(1))
    pages: dict[str, int] = {}
    for line in vm_stat.splitlines():
        match = _VM_STAT_COUNT.match(line.strip())
        if match:
            pages[match.group("name")] = int(match.group("pages"))
    available = (pages["Pages free"] + pages["Pages inactive"]) * page_size
    if total <= 0 or available > total:
        raise ValueError("vm_stat counts do not fit hw.memsize")
    return total - available, total


def _read_memory_macos() -> tuple[int | None, int | None]:
    try:
        return _parse_macos_memory(
            _run_macos_command("/usr/sbin/sysctl", "-n", "hw.memsize"),
            _run_macos_command("/usr/bin/vm_stat"),
        )
    except Exception:
        return None, None


def _read_memory() -> tuple[int | None, int | None]:
    if sys.platform == "darwin":
        return _read_memory_macos()
    return _read_memory_linux()


def _read_disk(path: Path) -> tuple[int | None, int | None]:
    try:
        usage = shutil.disk_usage(path)
        return usage.used, usage.total
    except Exception:
        return None, None


def collect_host_metrics(media_root: Path) -> HostMetricsSnapshot:
    memory_used, memory_total = _read_memory()
    disk_used, disk_total = _read_disk(media_root)
    return HostMetricsSnapshot(
        cpu_percent=_read_cpu_percent(),
        memory_used_bytes=memory_used,
        memory_total_bytes=memory_total,
        disk_used_bytes=disk_used,
        disk_total_bytes=disk_total,
    )


def collect_host_metrics_or_empty(media_root: Path) -> HostMetricsSnapshot:
    try:
        return collect_host_metrics(media_root)
    except Exception:
        return HostMetricsSnapshot(None, None, None, None, None)
