"""Versioned English preset contract."""

from voxint.export.filler_lists import NEVER, PRESET_VERSION, TIER_1, TIER_2, normalize_entries


def test_preset_literals() -> None:
    assert PRESET_VERSION == "en-1"
    assert TIER_1 == ("um", "uh", "umm", "uhh", "uhm", "erm")
    assert TIER_2 == (
        "hm",
        "mm",
        "mmm",
        "you know",
        "kind of",
        "sort of",
        "I mean",
        "I guess",
        "I suppose",
        "or something",
        "you know what I mean",
        "you see",
    )
    assert NEVER == ("like", "so", "right", "well")


def test_disjoint_valid_presets() -> None:
    first, second, never = (
        {entry.casefold() for entry in tier} for tier in (TIER_1, TIER_2, NEVER)
    )
    assert first.isdisjoint(second)
    assert never.isdisjoint(first | second)
    for entry in (*TIER_1, *TIER_2):
        assert normalize_entries([entry]) == (entry,)
