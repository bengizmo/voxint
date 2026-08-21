"""Contract: the app-side resource-status setting stays documented and gated (W2).

Mirrors ``test_watch_folder_config.py``: the ``resource_status_ttl_seconds``
knob is documented in ``.env.example`` under its env-var name, defaults sanely,
bounds hold, and it is not rescaled by the compute-tier profile (it is a UI
freshness knob, not a compute timing budget).
"""

import re

import pytest
from pydantic import ValidationError

from tests.contracts.conftest import REPO_ROOT
from voxint.config import TIER_SCALED_TIMING_FIELDS, Settings


def test_documented_in_env_example() -> None:
    env_example = (REPO_ROOT / ".env.example").read_text()
    assert re.search(
        r"^#?\s*RESOURCE_STATUS_TTL_SECONDS=", env_example, re.MULTILINE
    ), ".env.example lacks a RESOURCE_STATUS_TTL_SECONDS line"


def test_default() -> None:
    assert Settings(_env_file=None).resource_status_ttl_seconds == 10.0


def test_zero_allowed_negative_rejected() -> None:
    assert Settings(_env_file=None, resource_status_ttl_seconds=0).resource_status_ttl_seconds == 0
    with pytest.raises(ValidationError, match="resource_status_ttl_seconds"):
        Settings(_env_file=None, resource_status_ttl_seconds=-1)


def test_not_tier_scaled() -> None:
    assert "resource_status_ttl_seconds" not in TIER_SCALED_TIMING_FIELDS
