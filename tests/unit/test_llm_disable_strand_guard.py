"""The LLM form's disable-strand guard + its label map (issue #77).

`_persist_llm_settings` must never persist an LLM disable that strands a feature
`validate_effective_flags` requires ``llm_enabled=true`` for — a combination the
boot validator (config.py) would reject on restart — yet it must not block a
disable over an *unrelated* pre-existing violation, and must never auto-disable
the dependent (#62). These pin `_llm_disable_strand_error`'s delta logic and copy
directly (row=None so the whole combination comes from a boot-valid ``Settings``,
exercising the pure decision without a database), plus an anti-drift lock that
keeps `_LLM_DEPENDENCY_LABELS` in lockstep with the shared validator's messages.
"""

from voxint.api.app import (
    _LLM_DEPENDENCY_LABELS,
    _join_operator_labels,
    _llm_disable_strand_error,
)
from voxint.app_settings import EffectiveFlags, validate_effective_flags
from voxint.config import Settings


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_disable_safe_when_no_dependent_on() -> None:
    # LLM on, nothing depends on it → turning it off strands nothing.
    assert _llm_disable_strand_error(None, _settings(llm_enabled=True)) is None


def test_disable_blocked_by_run_assets() -> None:
    settings = _settings(llm_enabled=True, enrichment_run_assets_enabled=True)
    message = _llm_disable_strand_error(None, settings)
    assert message is not None
    assert "run assets" in message
    assert "it needs the LLM" in message  # singular blocker


def test_disable_blocked_by_names_llm() -> None:
    settings = _settings(
        llm_enabled=True,
        enrichment_names_enabled=True,
        enrichment_names_llm_enabled=True,
    )
    message = _llm_disable_strand_error(None, settings)
    assert message is not None
    assert "the LLM name pass" in message


def test_disable_blocked_by_web_research_producer() -> None:
    settings = _settings(
        llm_enabled=True,
        voxint_web_research=True,
        web_search_base_url="https://searx.example",
        enrichment_web_research_enabled=True,
    )
    message = _llm_disable_strand_error(None, settings)
    assert message is not None
    assert "web-research enrichment" in message


def test_multiple_blockers_join_into_one_grammatical_message() -> None:
    settings = _settings(
        llm_enabled=True,
        enrichment_names_enabled=True,
        enrichment_names_llm_enabled=True,
        enrichment_run_assets_enabled=True,
        voxint_web_research=True,
        web_search_base_url="https://searx.example",
        enrichment_web_research_enabled=True,
    )
    message = _llm_disable_strand_error(None, settings)
    assert message is not None
    # Stable validator order: names_llm, web-research, run assets.
    assert (
        "Turn off the LLM name pass, web-research enrichment, and run assets"
        in message
    )
    assert "they need the LLM" in message  # plural blockers


def test_message_carries_no_config_identifiers() -> None:
    # The whole arc exists to keep flag identifiers out of the operator's way.
    settings = _settings(llm_enabled=True, enrichment_run_assets_enabled=True)
    message = _llm_disable_strand_error(None, settings)
    assert message is not None
    for jargon in ("enrichment_", "llm_enabled", "voxint_web_research", "web_search_"):
        assert jargon not in message


def test_join_operator_labels_shapes() -> None:
    assert _join_operator_labels(["a"]) == "a"
    assert _join_operator_labels(["a", "b"]) == "a and b"
    assert _join_operator_labels(["a", "b", "c"]) == "a, b, and c"


def test_label_map_locks_every_llm_dependency_invariant() -> None:
    # Each invariant that requires llm_enabled=true, crafted to fire exactly once,
    # must produce a message that is a key in _LLM_DEPENDENCY_LABELS — so a reworded
    # invariant in #74 can never silently drop the operator label here.
    def _flags(**overrides: object) -> EffectiveFlags:
        base: dict[str, object] = {
            "llm_enabled": False,
            "enrichment_names_enabled": False,
            "enrichment_names_llm_enabled": False,
            "enrichment_run_assets_enabled": False,
            "enrichment_run_assets_autogenerate": False,
            "voxint_web_research": False,
            "enrichment_web_research_enabled": False,
            "web_search_base_url": "",
        }
        base.update(overrides)
        return EffectiveFlags(**base)  # type: ignore[arg-type]

    reachable = [
        _flags(enrichment_names_enabled=True, enrichment_names_llm_enabled=True),
        _flags(enrichment_run_assets_enabled=True),
        _flags(
            voxint_web_research=True,
            web_search_base_url="https://searx.example",
            enrichment_web_research_enabled=True,
        ),
    ]
    for flags in reachable:
        messages = validate_effective_flags(flags)
        assert len(messages) == 1, messages
        assert messages[0] in _LLM_DEPENDENCY_LABELS, messages[0]


def test_label_map_values_are_plain_language() -> None:
    for label in _LLM_DEPENDENCY_LABELS.values():
        assert "enrichment_" not in label
        assert "llm_enabled" not in label
        assert "_" not in label  # no raw config identifiers leak through
