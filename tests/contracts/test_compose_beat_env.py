"""Beat must validate the worker's settings and schedule its opt-in sweeps.

Resolve only the relevant Compose fields without reading the operator's .env:
identical env files plus identical overrides give both processes the same values.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
import yaml

from tests.contracts.conftest import REPO_ROOT


def _services(files: list[str]) -> dict[str, Any]:
    services: dict[str, Any] = {"beat": {}, "worker": {}}
    for filename in files:
        layer = yaml.safe_load((REPO_ROOT / filename).read_text())["services"]
        for name, effective in services.items():
            service = layer.get(name, {})
            # Compose merges environment/dependencies by key, with later values
            # winning; env_file is appended, and mounts merge by target path.
            for field in ("environment", "depends_on"):
                effective.setdefault(field, {}).update(service.get(field, {}))
            effective.setdefault("env_file", []).extend(service.get("env_file", []))
            mounts = effective.setdefault("volumes", {})
            for mount in service.get("volumes", []):
                target = re.search(r":(/[^:]+)(?::(?:ro|rw))?$", mount)
                assert target is not None, mount
                mounts[target[1]] = mount
    return services


@pytest.mark.parametrize("tier", [None, "cpu", "gpu", "rocm", "metal"])
@pytest.mark.parametrize("gpu_phase", [False, True])
def test_beat_matches_worker_settings(tier: str | None, gpu_phase: bool) -> None:
    files = ["compose.yaml"]
    if tier is not None:
        files.append(f"compose.{tier}.yaml")
    if gpu_phase:
        files.append("compose.gpu-phase.yaml")
    services = _services(files)
    beat, worker = services["beat"], services["worker"]
    assert beat["env_file"] == worker["env_file"] == [{"path": ".env", "required": False}]
    for key in ("MEDIA_ROOT", "DATABASE_URL", "REDIS_URL"):
        assert beat["environment"][key] == worker["environment"][key], key
    # Base/GPU inherit COMPUTE_TIER from the same env_file (or Settings default).
    assert beat["environment"].get("COMPUTE_TIER") == worker["environment"].get("COMPUTE_TIER")
    if tier in ("cpu", "rocm", "metal"):
        assert beat["environment"]["COMPUTE_TIER"] == tier
    assert beat["environment"]["MEDIA_ROOT"] == "/data/media"
    assert beat["volumes"]["/data/media"] == "${MEDIA_ROOT:-./media}:/data/media:ro"
    assert worker["volumes"]["/data/media"] == "${MEDIA_ROOT:-./media}:/data/media"
    assert beat["depends_on"]["migrate"] == worker["depends_on"]["migrate"] == {
        "condition": "service_completed_successfully"
    }
