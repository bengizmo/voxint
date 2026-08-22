"""Contract: the diarization speaker-ceiling setting stays documented and gated.

Mirrors ``test_resource_config.py``: the ``diarization_max_speakers`` knob
(issue #128) is documented in ``.env.example`` under its env-var name, defaults
to unset (so a default install sends no bound and the service applies its own
default), holds its 1..20 bounds when set, and is not rescaled by the
compute-tier profile (it is a diarization ceiling, not a compute timing budget).
"""

import re

import pytest
from pydantic import ValidationError

from tests.contracts.conftest import REPO_ROOT
from voxint.config import TIER_SCALED_TIMING_FIELDS, Settings


def test_documented_in_env_example() -> None:
    env_example = (REPO_ROOT / ".env.example").read_text()
    assert re.search(
        r"^#?\s*DIARIZATION_MAX_SPEAKERS=", env_example, re.MULTILINE
    ), ".env.example lacks a DIARIZATION_MAX_SPEAKERS line"


def test_default_is_unset() -> None:
    # Unset means "no bound" so a default install is byte-identical to pre-#128:
    # the client sends no max_speakers and the service applies its own default.
    assert Settings(_env_file=None).diarization_max_speakers is None


def test_bounds_reject_out_of_range() -> None:
    assert Settings(_env_file=None, diarization_max_speakers=1).diarization_max_speakers == 1
    assert Settings(_env_file=None, diarization_max_speakers=20).diarization_max_speakers == 20
    with pytest.raises(ValidationError, match="diarization_max_speakers"):
        Settings(_env_file=None, diarization_max_speakers=0)
    with pytest.raises(ValidationError, match="diarization_max_speakers"):
        Settings(_env_file=None, diarization_max_speakers=21)


def test_not_tier_scaled() -> None:
    assert "diarization_max_speakers" not in TIER_SCALED_TIMING_FIELDS
