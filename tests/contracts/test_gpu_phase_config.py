"""The opt-in GPU phase configuration stays documented with safe defaults."""

import re

from tests.contracts.conftest import REPO_ROOT
from voxint.config import Settings


def test_gpu_phase_defaults_and_documentation() -> None:
    expected = {
        "gpu_phase_enabled": False,
        "gpu_lease_acquire_url": "",
        "gpu_lease_release_url": "",
        "gpu_lease_status_url": "",
        "gpu_lease_token": "",
        "gpu_phase_tick_seconds": 30,
        "gpu_phase_min_dwell_seconds": 600,
        "gpu_phase_max_audio_seconds": 7200,
    }
    fields = {
        name for name in Settings.model_fields if name.startswith(("gpu_phase_", "gpu_lease_"))
    }
    assert fields == set(expected)
    env = (REPO_ROOT / ".env.example").read_text()
    for name, default in expected.items():
        assert Settings.model_fields[name].default == default
        assert re.search(rf"^#\s*{name.upper()}=", env, re.MULTILINE), name
