"""Pure unit tests for the substitution extractor (learning.py, #476)."""

from __future__ import annotations

from voxint.domain_packs.learning import (
    MAX_BASE_WORDS_FOR_LEARNING,
    MAX_LEARNED_WORDS_PER_SIDE,
    MAX_PAIRS_PER_SEGMENT,
    Substitution,
    extract_substitutions,
)


def _extract(base: str, edited: str, *, raw: str | None = None) -> tuple[Substitution, ...]:
    """Shorthand: raw defaults to base when not given."""
    return extract_substitutions(base, edited, raw=raw if raw is not None else base)


class TestSingleWord:
    def test_simple_replacement(self) -> None:
        result = _extract("the hvac system", "the HVAC system")
        assert result == (Substitution("hvac", "HVAC"),)

    def test_case_only_pair(self) -> None:
        result = _extract("it works", "IT works")
        assert result == (Substitution("it", "IT"),)


class TestMultiWord:
    def test_two_word_replacement(self) -> None:
        result = _extract("the air handler unit", "the AHU unit")
        assert result == (Substitution("air handler", "AHU"),)

    def test_four_word_replacement(self) -> None:
        result = _extract(
            "the return on investment calculation here",
            "the ROI calculation here",
        )
        assert result == (Substitution("return on investment", "ROI"),)


class TestIntraWordCharacters:
    def test_apostrophe_in_word(self) -> None:
        result = _extract("it doesnt work", "it doesn't work")
        assert result == (Substitution("doesnt", "doesn't"),)

    def test_hyphenated_word(self) -> None:
        result = _extract("the non toxic material", "the non-toxic material")
        assert result == (Substitution("non toxic", "non-toxic"),)

    def test_combining_mark(self) -> None:
        result = _extract("the cafe is open", "the café is open")
        assert result == (Substitution("cafe", "café"),)


class TestSkips:
    def test_punctuation_only_diff_skipped(self) -> None:
        result = _extract("hello world", "hello, world")
        assert result == ()

    def test_insert_opcode_ignored(self) -> None:
        result = _extract("hello world", "hello beautiful world")
        assert result == ()

    def test_delete_opcode_ignored(self) -> None:
        result = _extract("hello beautiful world", "hello world")
        assert result == ()

    def test_over_four_word_side_dropped(self) -> None:
        match = " ".join(["word"] * (MAX_LEARNED_WORDS_PER_SIDE + 1))
        result = _extract(f"the {match} here", "the replacement here")
        assert result == ()

    def test_self_refire_pair_dropped(self) -> None:
        result = _extract("zoom call today", "zoom board call today")
        assert result == ()


class TestRawGating:
    def test_pair_not_in_raw_dropped(self) -> None:
        result = _extract(
            "the enhanced text here",
            "the fixed text here",
            raw="the original text here",
        )
        assert result == ()

    def test_pair_in_both_base_and_raw_kept(self) -> None:
        result = _extract(
            "the hvac system here",
            "the HVAC system here",
            raw="the hvac system here",
        )
        assert result == (Substitution("hvac", "HVAC"),)


class TestRewriteCutoff:
    def test_one_word_segment_fully_replaced_is_learned(self) -> None:
        result = _extract("hvac", "HVAC")
        assert result == (Substitution("hvac", "HVAC"),)

    def test_cutoff_rejects_when_too_many_replaced(self) -> None:
        words = ["word"] * 10
        base = " ".join(words)
        edited = " ".join(["changed"] * 10)
        result = _extract(base, edited)
        assert result == ()


class TestCaps:
    def test_pair_cap_at_max(self) -> None:
        base_words = [f"w{i}" for i in range(MAX_PAIRS_PER_SEGMENT + 2)]
        edited_words = [f"x{i}" for i in range(MAX_PAIRS_PER_SEGMENT + 2)]
        base = " ".join(base_words)
        edited = " ".join(edited_words)
        result = _extract(base, edited)
        assert len(result) <= MAX_PAIRS_PER_SEGMENT

    def test_base_too_long_skipped(self) -> None:
        words = ["word"] * (MAX_BASE_WORDS_FOR_LEARNING + 1)
        base = " ".join(words)
        edited = base.replace("word", "changed", 1)
        result = _extract(base, edited)
        assert result == ()


class TestDeterminism:
    def test_order_preserving(self) -> None:
        base = "the hvac system and the ahu unit"
        edited = "the HVAC system and the AHU unit"
        result = _extract(base, edited)
        assert result == (
            Substitution("hvac", "HVAC"),
            Substitution("ahu", "AHU"),
        )

    def test_deduplicated(self) -> None:
        base = "hvac and hvac"
        edited = "HVAC and HVAC"
        result = _extract(base, edited)
        assert result == (Substitution("hvac", "HVAC"),)


class TestWhitespace:
    def test_whitespace_normalized(self) -> None:
        result = _extract("the  hvac  system", "the  HVAC  system")
        assert result == (Substitution("hvac", "HVAC"),)

    def test_no_change_produces_nothing(self) -> None:
        result = _extract("hello world", "hello world")
        assert result == ()
