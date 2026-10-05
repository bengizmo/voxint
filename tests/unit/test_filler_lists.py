"""Validation and resolution of pure filler lists."""

from dataclasses import replace
from string import ascii_lowercase

import pytest

from voxint.export.filler_lists import (
    DEFAULT_FILLER_LIST,
    TIER_1,
    FillerListError,
    effective_filler_list,
    normalize_entries,
)


def test_normalize_entries() -> None:
    assert normalize_entries(["  You\t know\n", "you KNOW", "", "  ", "Um"]) == ("You know", "Um")
    assert normalize_entries(["can't", "can\u2019t", "speakers'", "speakers\u2019", "uh-huh"]) == (
        "can't",
        "can\u2019t",
        "speakers'",
        "speakers\u2019",
        "uh-huh",
    )
    assert normalize_entries(["a" * 40, "one two three four five"]) == (
        "a" * 40,
        "one two three four five",
    )


@pytest.mark.parametrize(
    "entry",
    [
        "um1",
        "uh--huh",
        "-um",
        "um-",
        "um!",
        "'um",
        "um''",
        "a_b",
        "one two three four five six",
        "a" * 41,
    ],
)
def test_refuse_invalid_entries(entry: str) -> None:
    with pytest.raises(FillerListError):
        normalize_entries([entry])


def test_entry_limit_after_deduplication() -> None:
    entries = [a + b for a in ascii_lowercase for b in ascii_lowercase][:101]
    assert normalize_entries(entries[:100]) == tuple(entries[:100])
    assert normalize_entries([*entries[:100], entries[0].upper()]) == tuple(entries[:100])
    with pytest.raises(FillerListError, match="100"):
        normalize_entries(entries)


def test_keep_and_provenance() -> None:
    result = effective_filler_list(
        ["You Know", "YOU KNOW", "HM"],
        ["UM", "mm", "you know"],
        add_source="settings",
        keep_source="environment",
        env_overridden=("add",),
    )
    assert result.added == ("You Know", "HM")
    assert result.kept == ("UM", "mm", "you know")
    assert result.kept_without_effect == ("mm",)
    assert result.words == ("erm", "HM", "uh", "uhh", "uhm", "umm")
    assert result.phrases == ()
    assert (result.language, result.preset_version) == ("en", "en-1")
    assert (result.add_source, result.keep_source, result.env_overridden) == (
        "settings",
        "environment",
        ("add",),
    )
    assert not result.is_default
    assert effective_filler_list(keep=["hm"]).kept_without_effect == ("hm",)


def test_empty_and_default_sets() -> None:
    empty = effective_filler_list(keep=TIER_1)
    assert empty.words == empty.phrases == ()
    assert not empty.is_default
    assert DEFAULT_FILLER_LIST.is_default
    assert replace(DEFAULT_FILLER_LIST, words=tuple(reversed(TIER_1))).is_default
    assert effective_filler_list(["UM"]).is_default


def test_ordering_and_exact_keep() -> None:
    result = effective_filler_list(
        ["You Know", "You know what I mean", "I mean", "aa bb", "cc dd", "HM"],
        ["YOU KNOW"],
    )
    assert result.words == ("erm", "HM", "uh", "uhh", "uhm", "um", "umm")
    assert result.phrases == ("You know what I mean", "I mean", "aa bb", "cc dd")


@pytest.mark.parametrize("add", [(), ("UM",), ("You Know", "HM", "I mean", "um")])
@pytest.mark.parametrize("keep", [(), ("hmm",), ("UM", "you know", "hmm"), TIER_1])
def test_reporting_entries(add: tuple[str, ...], keep: tuple[str, ...]) -> None:
    result = effective_filler_list(add, keep)
    effective = {entry.casefold() for entry in (*result.words, *result.phrases)}
    assert result.extra_entries == tuple(
        entry for entry in add if entry.casefold() in effective and entry.casefold() not in TIER_1
    )
    assert result.effective_keeps == tuple(
        entry for entry in keep if entry.casefold() in {e.casefold() for e in (*TIER_1, *add)}
    )
    assert result.is_default or result.extra_entries or result.effective_keeps


def test_reporting_order_and_cancelled_addition() -> None:
    result = effective_filler_list(["I mean", "HM", "You Know", "UM"], ["hm", "hmm", "Um"])
    assert result.extra_entries == ("I mean", "You Know")
    assert result.effective_keeps == ("hm", "Um")
