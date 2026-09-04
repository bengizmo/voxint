"""Contract: the watch-folder ingest settings stay documented and gated (issue #60).

Mirrors ``test_notify_config.py`` for the watch-folder surface: every
``watch_folder_*`` field is documented in ``.env.example`` under its env-var name
(a field added without documentation fails here), the feature is disabled by
default, and the interval/settle bounds hold. The runtime enable path is the
nullable ``app_settings.watch_folder_enabled`` column (covered by the app-settings
resolver tests); this contract pins the env-config surface only.
"""

import re

import pytest
from pydantic import ValidationError

from tests.contracts.conftest import REPO_ROOT
from voxint.config import Settings

_WATCH_FIELDS = [name for name in Settings.model_fields if name.startswith("watch_folder_")]


def test_every_watch_folder_field_is_documented_in_env_example() -> None:
    env_example = (REPO_ROOT / ".env.example").read_text()
    assert _WATCH_FIELDS, "watch-folder settings fields disappeared"
    missing = [
        name
        for name in _WATCH_FIELDS
        if not re.search(rf"^#?\s*{name.upper()}=", env_example, re.MULTILINE)
    ]
    assert not missing, f".env.example lacks lines for: {missing}"


def test_watch_folder_default_off() -> None:
    assert Settings(_env_file=None).watch_folder_enabled is False


def test_watch_folder_sweep_interval_has_floor() -> None:
    with pytest.raises(ValidationError, match="watch_folder_sweep_seconds"):
        Settings(_env_file=None, watch_folder_sweep_seconds=5)


def test_watch_folder_settle_allows_zero_but_not_negative() -> None:
    assert Settings(_env_file=None, watch_folder_settle_seconds=0).watch_folder_settle_seconds == 0
    with pytest.raises(ValidationError, match="watch_folder_settle_seconds"):
        Settings(_env_file=None, watch_folder_settle_seconds=-1)


def test_watch_folder_batch_size_has_floor() -> None:
    assert Settings(_env_file=None).watch_folder_batch_size == 8
    with pytest.raises(ValidationError, match="watch_folder_batch_size"):
        Settings(_env_file=None, watch_folder_batch_size=0)


def test_watch_folder_not_tier_scaled() -> None:
    """Wall-clock pickup latency, not a compute-tier timing budget — the sweep
    cadence must not be rescaled by the compute-tier profile."""
    from voxint.config import TIER_SCALED_TIMING_FIELDS

    assert "watch_folder_sweep_seconds" not in TIER_SCALED_TIMING_FIELDS
    assert "watch_folder_settle_seconds" not in TIER_SCALED_TIMING_FIELDS
