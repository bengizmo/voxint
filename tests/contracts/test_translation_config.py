"""Contract: the translation settings stay documented and gated (issue #133).

Mirrors ``test_run_assets_config.py`` for the translation surface: every
``translation_*`` field documented in ``.env.example``; off/unset by default;
and the self-contained invariants — autogenerate REQUIRES a target language,
and a target must be a code the vendored language map knows (fail-closed at
startup). The LLM dependency is deliberately a runtime gate
(``translation_gates_open``), not an env-time invariant, because LLM
enablement is itself runtime-togglable."""

import re

import pytest
from pydantic import ValidationError

from tests.contracts.conftest import REPO_ROOT
from voxint.api.languages import LANGUAGE_NAMES
from voxint.config import Settings
from voxint.db.models import AppSettings

_TRANSLATION_FIELDS = [
    name for name in Settings.model_fields if name.startswith("translation_")
]


def test_every_translation_field_is_documented_in_env_example() -> None:
    env_example = (REPO_ROOT / ".env.example").read_text()
    assert _TRANSLATION_FIELDS, "translation settings fields disappeared"
    missing = [
        name
        for name in _TRANSLATION_FIELDS
        if not re.search(rf"^#?\s*{name.upper()}=", env_example, re.MULTILINE)
    ]
    assert not missing, f".env.example lacks lines for: {missing}"


def test_translation_default_unconfigured() -> None:
    settings = Settings(_env_file=None)
    assert settings.translation_target_language is None
    assert settings.translation_autogenerate is False


def test_autogenerate_fails_closed_without_target() -> None:
    with pytest.raises(ValidationError, match="translation_autogenerate"):
        Settings(_env_file=None, translation_autogenerate=True)


def test_unknown_target_code_fails_closed() -> None:
    with pytest.raises(ValidationError, match="not a language code"):
        Settings(_env_file=None, translation_target_language="tlh")


def test_valid_combination_constructs() -> None:
    settings = Settings(
        _env_file=None,
        translation_target_language="es",
        translation_autogenerate=True,
    )
    assert settings.translation_target_language == "es"
    assert settings.translation_autogenerate is True


def test_row_columns_mirror_config_fields() -> None:
    # The tri-state override columns must exist under the exact config names
    # the shared resolvers rely on (the "column name mirrors the config field"
    # contract every resolve_effective_* helper assumes).
    for name in _TRANSLATION_FIELDS:
        assert hasattr(AppSettings, name), f"AppSettings lacks column {name}"


def test_language_map_covers_the_target_validation() -> None:
    # The invariant validates against the vendored map; a gutted map would
    # silently reject every configuration.
    assert len(LANGUAGE_NAMES) >= 90
    assert LANGUAGE_NAMES["es"] == "Spanish"
