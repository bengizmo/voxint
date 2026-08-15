"""Contract: the run-asset settings stay documented and gated (issue #41).

Mirrors ``test_web_researcher_config.py`` for the run-asset surface: every
``enrichment_run_assets_*`` / ``run_assets_*`` field documented in
``.env.example``; disabled by default; and the composition invariants — the
feature flag REQUIRES the LLM (fail-closed at startup), and the post-finalize
autogenerate step REQUIRES the feature flag."""

import re

import pytest
from pydantic import ValidationError

from tests.contracts.conftest import REPO_ROOT
from voxint.config import Settings

_ASSET_FIELDS = [
    name
    for name in Settings.model_fields
    if name.startswith("enrichment_run_assets_") or name.startswith("run_assets_")
]


def test_every_run_assets_field_is_documented_in_env_example() -> None:
    env_example = (REPO_ROOT / ".env.example").read_text()
    assert _ASSET_FIELDS, "run-asset settings fields disappeared"
    missing = [
        name
        for name in _ASSET_FIELDS
        if not re.search(rf"^#?\s*{name.upper()}=", env_example, re.MULTILINE)
    ]
    assert not missing, f".env.example lacks lines for: {missing}"


def test_run_assets_default_off() -> None:
    settings = Settings(_env_file=None)
    assert settings.enrichment_run_assets_enabled is False
    assert settings.enrichment_run_assets_autogenerate is False


def test_run_assets_flag_fails_closed_without_llm() -> None:
    with pytest.raises(ValidationError, match="enrichment_run_assets_enabled"):
        Settings(_env_file=None, enrichment_run_assets_enabled=True)


def test_autogenerate_fails_closed_without_feature() -> None:
    # Even with the LLM configured: autogenerate rides the feature flag.
    with pytest.raises(ValidationError, match="enrichment_run_assets_autogenerate"):
        Settings(_env_file=None, enrichment_run_assets_autogenerate=True, llm_enabled=True)


def test_run_assets_flag_constructs_with_llm() -> None:
    settings = Settings(
        _env_file=None,
        enrichment_run_assets_enabled=True,
        enrichment_run_assets_autogenerate=True,
        llm_enabled=True,
    )
    assert settings.enrichment_run_assets_enabled is True
    assert settings.enrichment_run_assets_autogenerate is True
    assert settings.run_assets_max_input_chars == 48_000
