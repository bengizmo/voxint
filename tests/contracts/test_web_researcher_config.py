"""Contract: the web-research *producer* settings stay documented and gated
(issue #40).

Mirrors ``test_web_research_config.py`` for the producer's surface: every
``enrichment_web_research_enabled`` / ``research_*`` field documented in
``.env.example``; disabled by default; and the composition invariant — the
producer flag REQUIRES both the retrieval capability and the LLM (fail-closed
at startup), while the two prerequisites themselves stay independent of each
other (#39's contract, untouched)."""

import re

import pytest
from pydantic import ValidationError

from tests.contracts.conftest import REPO_ROOT
from voxint.config import Settings

_PRODUCER_FIELDS = [
    name
    for name in Settings.model_fields
    if name == "enrichment_web_research_enabled" or name.startswith("research_")
]


def test_every_producer_field_is_documented_in_env_example() -> None:
    env_example = (REPO_ROOT / ".env.example").read_text()
    assert _PRODUCER_FIELDS, "web-research producer settings fields disappeared"
    missing = [
        name
        for name in _PRODUCER_FIELDS
        if not re.search(rf"^#?\s*{name.upper()}=", env_example, re.MULTILINE)
    ]
    assert not missing, f".env.example lacks lines for: {missing}"


def test_producer_defaults_off() -> None:
    assert Settings(_env_file=None).enrichment_web_research_enabled is False


@pytest.mark.parametrize(
    "overrides",
    [
        # No prerequisites at all.
        {},
        # Retrieval without the LLM.
        {
            "voxint_web_research": True,
            "web_search_base_url": "http://searx.lan:8888",
        },
        # LLM without retrieval.
        {"llm_enabled": True},
    ],
)
def test_producer_flag_fails_closed_without_both_prerequisites(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="enrichment_web_research_enabled"):
        Settings(
            _env_file=None,
            enrichment_web_research_enabled=True,
            **overrides,  # type: ignore[arg-type]
        )


def test_producer_flag_constructs_with_both_prerequisites() -> None:
    settings = Settings(
        _env_file=None,
        enrichment_web_research_enabled=True,
        voxint_web_research=True,
        web_search_base_url="http://searx.lan:8888",
        llm_enabled=True,
    )
    assert settings.enrichment_web_research_enabled is True
    # Budgets carry sane bounded defaults for the console's preview.
    assert settings.research_max_searches == 3
    assert settings.research_max_reads == 5
    assert settings.research_max_rounds == 5
    assert settings.research_max_actions_per_round == 3
    assert settings.research_deadline_seconds == 300.0
