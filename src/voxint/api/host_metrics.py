"""Host-level CPU / memory / disk metrics for the status page."""

from __future__ import annotations

import os
import shutil
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


def _read_disk(path: Path) -> tuple[int | None, int | None]:
    try:
        usage = shutil.disk_usage(path)
        return usage.used, usage.total
    except Exception:
        return None, None


def collect_host_metrics(media_root: Path) -> HostMetricsSnapshot:
    memory_used, memory_total = _read_memory_linux()
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
