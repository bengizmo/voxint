"""The Features section's plain-language invariant copy stays in lockstep (#62).

`validate_effective_flags` is the single source of WHICH cross-flag combinations
are invalid; its messages name the flag identifiers. The Settings → Features
boundary (`_FEATURE_INVARIANT_COPY` in `voxint.api.app`) translates the four
reachable messages to operator-plain copy. This anti-drift test rebuilds each
reachable violation and asserts the exact message it produces is a mapping key,
so a reworded invariant in #74 cannot silently fall back to jargon in the UI.
"""

from voxint.api.app import _FEATURE_INVARIANT_COPY
from voxint.app_settings import EffectiveFlags, validate_effective_flags


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


def test_every_reachable_features_invariant_has_plain_copy() -> None:
    # The four invariants the Features form (names / names_llm / run_assets /
    # autogenerate) can trigger — each crafted to violate exactly one.
    reachable = [
        _flags(enrichment_names_enabled=True, enrichment_names_llm_enabled=True),  # names_llm ⇒ llm
        _flags(llm_enabled=True, enrichment_names_llm_enabled=True),  # names_llm ⇒ names
        _flags(enrichment_run_assets_enabled=True),  # run_assets ⇒ llm
        _flags(llm_enabled=True, enrichment_run_assets_autogenerate=True),  # autogen ⇒ run_assets
    ]
    for flags in reachable:
        messages = validate_effective_flags(flags)
        assert len(messages) == 1, messages  # each crafted to a single violation
        assert messages[0] in _FEATURE_INVARIANT_COPY, messages[0]


def test_plain_copy_drops_the_flag_identifiers() -> None:
    # The whole point: the operator-facing copy must not carry the raw config
    # identifiers the arc exists to hide.
    for plain in _FEATURE_INVARIANT_COPY.values():
        assert "enrichment_" not in plain
        assert "llm_enabled=true" not in plain
