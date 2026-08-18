"""Engine mechanics + Gate B corpus for the deterministic corrector (issue #81).

Covers the frozen apply semantics (leftmost-longest, non-cascading, exact-literal
replace, offset-shifted trace spans, atomic growth rejection) and drives the
``tests/fixtures/rules_correct/`` corpus. The stricter-than-LLM faithfulness gate
(reused enhancement fixtures, empty rule set) lives in
``test_corrector_faithfulness.py``.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

import pytest

from voxint.domain_packs.base import DomainPackError
from voxint.domain_packs.corrections import CorrectionRule, parse_corrections
from voxint.domain_packs.corrector import (
    AppliedCorrection,
    CorrectionResult,
    apply_corrections,
)

RULES_CORRECT_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "rules_correct"


def _rule(
    rid: str,
    match: str,
    replace: str,
    *,
    case_sensitive: bool = True,
    whole_word: bool = True,
) -> CorrectionRule:
    return CorrectionRule(
        id=rid,
        match=match,
        replace=replace,
        case_sensitive=case_sensitive,
        whole_word=whole_word,
    )


def _assert_trace_spans_final(result: CorrectionResult) -> None:
    """Every trace span addresses ``result.text`` and equals its ``to_text``."""
    for entry in result.trace:
        start, end = entry.span
        assert 0 <= start < end <= len(result.text)
        assert result.text[start:end] == entry.to_text


def _assert_trace_faithful(result: CorrectionResult, source: str) -> None:
    """Bidirectional walk: prove ZERO unauthorized edits.

    Walk ``source`` and ``result.text`` in lockstep across the ordered trace:
    every unchanged gap must be byte-identical on both sides, each trace entry's
    ``from_text`` must be exactly the next source slice and ``to_text`` exactly the
    next result slice, and the suffixes must match after the last entry. This is a
    stronger oracle than reconstructing the output alone — it proves no character
    outside an applied span changed, in either coordinate space.
    """
    # spans ordered, half-open, in-bounds, non-overlapping in the result space.
    prev_end = 0
    for entry in result.trace:
        start, end = entry.span
        assert start >= prev_end, "trace spans overlap or are out of order"
        assert 0 <= start < end <= len(result.text)
        prev_end = end

    src_cursor = 0
    out_cursor = 0
    for entry in result.trace:
        out_start, out_end = entry.span
        # unchanged gap before this edit is byte-identical on both sides.
        gap = result.text[out_cursor:out_start]
        assert source[src_cursor : src_cursor + len(gap)] == gap
        src_cursor += len(gap)
        # the matched source slice equals from_text; the result slice equals to_text.
        assert source[src_cursor : src_cursor + len(entry.from_text)] == entry.from_text
        assert result.text[out_start:out_end] == entry.to_text
        src_cursor += len(entry.from_text)
        out_cursor = out_end
    # suffixes after the last edit match byte-for-byte.
    assert source[src_cursor:] == result.text[out_cursor:]


# --------------------------------------------------------------------------- #
# The codex-flagged critical regression: cross-rule overlapping matches.
# --------------------------------------------------------------------------- #
def test_cross_rule_overlap_rediscovers_hidden_match() -> None:
    # iter_matches(aba, "xababa") yields only [1,4) and hides [3,6). Once xab wins
    # [0,3), the cursor-relative rediscovery must surface aba at [3,6) => "XA".
    b = _rule("B", "xab", "X", whole_word=False)
    a = _rule("A", "aba", "A", whole_word=False)
    result = apply_corrections("xababa", [b, a])
    assert result.text == "XA"
    assert [(e.id, e.from_text, e.to_text) for e in result.trace] == [
        ("B", "xab", "X"),
        ("A", "aba", "A"),
    ]
    _assert_trace_faithful(result, "xababa")


# --------------------------------------------------------------------------- #
# Leftmost-longest selection + tie-breaks.
# --------------------------------------------------------------------------- #
def test_longest_wins_at_same_start() -> None:
    short = _rule("short", "cat", "S", whole_word=False)
    long = _rule("long", "catalog", "L", whole_word=False)
    # order should not matter for a length tie-break.
    assert apply_corrections("catalog", [short, long]).text == "L"
    assert apply_corrections("catalog", [long, short]).text == "L"


def test_earlier_start_beats_later_longer() -> None:
    early = _rule("early", "ab", "E", whole_word=False)
    late = _rule("late", "bcd", "L", whole_word=False)
    # "ab" starts at 0 (len 2); "bcd" starts at 1 (len 3). Leftmost start wins,
    # even though the later match is longer.
    result = apply_corrections("abcd", [early, late])
    assert result.text == "Ecd"


def test_manifest_order_breaks_identical_span_tie() -> None:
    first = _rule("first", "cat", "1", whole_word=False)
    second = _rule("second", "cat", "2", whole_word=False)
    # same start, same length -> earliest rule in the sequence wins.
    assert apply_corrections("cat", [first, second]).text == "1"
    assert apply_corrections("cat", [second, first]).text == "2"


def test_adjacent_winners() -> None:
    a = _rule("a", "foo", "X", whole_word=False)
    b = _rule("b", "bar", "Y", whole_word=False)
    result = apply_corrections("foobar", [a, b])
    assert result.text == "XY"
    _assert_trace_faithful(result, "foobar")


# --------------------------------------------------------------------------- #
# Non-cascading + offset-shifted spans (expanding and shrinking).
# --------------------------------------------------------------------------- #
def test_non_cascading_replacement_not_rescanned() -> None:
    # Replacing "a"->"ba" must not then match a "ba"->"Z" rule on the inserted "ba".
    a = _rule("a", "a", "ba", whole_word=False)
    z = _rule("z", "ba", "Z", whole_word=False)
    # In "a", only "a" matches originally; the produced "ba" is never re-scanned.
    result = apply_corrections("a", [a, z])
    assert result.text == "ba"


def test_expanding_replacements_shift_later_spans() -> None:
    r = _rule("r", "x", "LONG", whole_word=False)
    result = apply_corrections("x y x", [r])
    assert result.text == "LONG y LONG"
    assert [e.span for e in result.trace] == [(0, 4), (7, 11)]
    _assert_trace_spans_final(result)
    _assert_trace_faithful(result, "x y x")


def test_shrinking_replacements_shift_later_spans() -> None:
    r = _rule("r", "hello", "hi", whole_word=False)
    result = apply_corrections("hello a hello", [r])
    assert result.text == "hi a hi"
    assert [e.span for e in result.trace] == [(0, 2), (5, 7)]
    _assert_trace_faithful(result, "hello a hello")


# --------------------------------------------------------------------------- #
# Exact-literal replace (no case inheritance) + case sensitivity.
# --------------------------------------------------------------------------- #
def test_exact_literal_replace_under_ignorecase() -> None:
    # case-insensitive match, but replace is inserted verbatim (no case inheritance).
    r = _rule("r", "selectboard", "Selectboard", case_sensitive=False, whole_word=False)
    assert apply_corrections("SELECTBOARD", [r]).text == "Selectboard"
    assert apply_corrections("selectboard", [r]).text == "Selectboard"


def test_case_sensitive_default_does_not_match_other_case() -> None:
    r = _rule("r", "cat", "DOG", whole_word=False)  # case_sensitive=True default
    assert apply_corrections("CAT", [r]).text == "CAT"


def test_regex_metachar_match_is_literal() -> None:
    r = _rule("r", "C.D.B.G.", "CDBG", whole_word=False)
    # The dots are literal, so "CxDxBxGx" (regex ".") must NOT match.
    assert apply_corrections("CxDxBxGx", [r]).text == "CxDxBxGx"
    assert apply_corrections("C.D.B.G.", [r]).text == "CDBG"


# --------------------------------------------------------------------------- #
# Boundary / whole-word edges.
# --------------------------------------------------------------------------- #
def test_whole_word_collision_substring_not_matched() -> None:
    r = _rule("r", "cat", "DOG")  # whole_word=True default
    assert apply_corrections("catalog", [r]).text == "catalog"


def test_possessive_contraction_not_split() -> None:
    r = _rule("it", "it", "IT")  # whole_word default; apostrophe is intra-word.
    # "it's" must not fire; the standalone "it" must.
    assert apply_corrections("it's here it is", [r]).text == "it's here IT is"


def test_hyphenated_compound_not_split() -> None:
    r = _rule("co", "co", "CO")
    assert apply_corrections("co-op and co", [r]).text == "co-op and CO"


def test_punctuation_adjacent_matches() -> None:
    r = _rule("z", "zoom board", "Zoning Board")
    assert apply_corrections("the zoom board.", [r]).text == "the Zoning Board."


def test_nfd_combining_mark_grapheme_not_corrupted() -> None:
    # "Zoë" decomposed = Z o e U+0308. A whole-word "e" rule must NOT fire inside it.
    zoe = "Zo" + "ë"
    assert unicodedata.is_normalized("NFC", zoe) is False
    r = _rule("e", "e", "3")
    assert apply_corrections(zoe, [r]).text == zoe


# --------------------------------------------------------------------------- #
# No-op paths.
# --------------------------------------------------------------------------- #
def test_empty_rules_is_identity() -> None:
    result = apply_corrections("anything at all", [])
    assert result == CorrectionResult(text="anything at all", trace=(), growth_rejected=False)


def test_empty_text() -> None:
    r = _rule("r", "x", "y", whole_word=False)
    assert apply_corrections("", [r]) == CorrectionResult(text="", trace=(), growth_rejected=False)


def test_no_match_is_identity() -> None:
    r = _rule("r", "absent", "X", whole_word=False)
    assert apply_corrections("present", [r]).text == "present"


# --------------------------------------------------------------------------- #
# Idempotence — over a validated set (and the documented cross-boundary limit).
# --------------------------------------------------------------------------- #
def test_idempotent_over_validated_set() -> None:
    rules = parse_corrections(
        [
            {"id": "a", "match": "zoom board", "replace": "Zoning Board", "whole_word": True},
            {"id": "b", "match": "c d b g", "replace": "CDBG", "whole_word": True},
        ]
    )
    text = "the zoom board discussed c d b g today"
    once = apply_corrections(text, rules)
    twice = apply_corrections(once.text, rules)
    assert once.text == twice.text


def test_chain_rejected_at_load() -> None:
    # [a->b, b->c] is refused by #80's load-time guard, which is what makes
    # idempotence hold for any set the engine actually receives.
    with pytest.raises(DomainPackError):
        parse_corrections(
            [
                {"id": "1", "match": "a", "replace": "b", "whole_word": False},
                {"id": "2", "match": "b", "replace": "c", "whole_word": False},
            ]
        )


def test_cross_boundary_refire_is_not_guaranteed_idempotent() -> None:
    # HONEST LIMIT (codex): #80's guard covers intra-replacement re-firing only.
    # [ab->x, xc->y] passes load validation, yet a second pass matches "xc" spanning
    # the first pass's replacement and adjacent original text. The single engine pass
    # is deterministic; idempotence across such cross-boundary chains is NOT promised.
    rules = parse_corrections(
        [
            {"id": "1", "match": "ab", "replace": "x", "whole_word": False},
            {"id": "2", "match": "xc", "replace": "y", "whole_word": False},
        ]
    )
    once = apply_corrections("abc", rules)
    twice = apply_corrections(once.text, rules)
    assert once.text == "xc"  # single pass: only "ab"->x fires; produced "xc" not rescanned
    assert twice.text == "y"  # second pass now sees "xc" -> y
    assert once.text != twice.text


# --------------------------------------------------------------------------- #
# Growth rejection (atomic) + arg validation.
# --------------------------------------------------------------------------- #
def test_growth_one_over_rejected() -> None:
    r = _rule("g", "a", "XXXX", whole_word=False)
    result = apply_corrections("aaa", [r], max_output_chars=6)  # projected 12
    assert result == CorrectionResult(text="aaa", trace=(), growth_rejected=True)


def test_growth_exactly_at_limit_allowed() -> None:
    r = _rule("g", "a", "XX", whole_word=False)
    # "aa" -> "XXXX", projected length 4 == limit 4: allowed.
    result = apply_corrections("aa", [r], max_output_chars=4)
    assert result.text == "XXXX"
    assert result.growth_rejected is False


def test_growth_shrinking_output_never_rejected() -> None:
    r = _rule("g", "hello", "hi", whole_word=False)
    result = apply_corrections("hello hello", [r], max_output_chars=6)
    assert result.text == "hi hi"
    assert result.growth_rejected is False


def test_growth_multi_winner_atomic_rollback() -> None:
    # Two winners individually small, together exceed the limit -> whole segment
    # rejected atomically (no partial application, empty trace).
    a = _rule("a", "one", "AAAA", whole_word=False)
    b = _rule("b", "two", "BBBB", whole_word=False)
    result = apply_corrections("one two", [a, b], max_output_chars=7)  # projected 9
    assert result == CorrectionResult(text="one two", trace=(), growth_rejected=True)


def test_growth_noop_precedence_oversize_input_not_rejected() -> None:
    r = _rule("r", "absent", "X", whole_word=False)
    result = apply_corrections("a very long untouched string", [r], max_output_chars=2)
    assert result.growth_rejected is False
    assert result.text == "a very long untouched string"


@pytest.mark.parametrize("bad", [0, -1, True, False, 1.5, "10"])
def test_invalid_max_output_chars_rejected(bad: Any) -> None:
    r = _rule("r", "x", "y", whole_word=False)
    with pytest.raises(ValueError):
        apply_corrections("x", [r], max_output_chars=bad)


# --------------------------------------------------------------------------- #
# to_mapping serialization shape.
# --------------------------------------------------------------------------- #
def test_applied_correction_to_mapping() -> None:
    entry = AppliedCorrection(id="r", from_text="zoom board", to_text="Zoning Board", span=(4, 16))
    assert entry.to_mapping() == {
        "id": "r",
        "from": "zoom board",
        "to": "Zoning Board",
        "span": [4, 16],
    }


# --------------------------------------------------------------------------- #
# Gate B corpus driver — data-declared cases in tests/fixtures/rules_correct/.
# --------------------------------------------------------------------------- #
def _load_corpus() -> list[tuple[str, dict[str, Any]]]:
    if not RULES_CORRECT_DIR.is_dir():
        return []
    cases: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(RULES_CORRECT_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        cases.append((path.stem, data))
    return cases


CORPUS = _load_corpus()


def test_corpus_is_present() -> None:
    # Guard against an empty glob silently passing the parametrized gate.
    assert CORPUS, "Gate B corpus tests/fixtures/rules_correct/ is empty"


@pytest.mark.parametrize("name,case", CORPUS, ids=[c[0] for c in CORPUS])
def test_rules_correct_corpus(name: str, case: dict[str, Any]) -> None:
    """One frozen Gate B case: exact expected text + trace, or a declared failure.

    Valid-pack rule sets round-trip through ``parse_corrections`` so a case cannot
    smuggle an invalid set past load validation as production-valid.
    """
    text = case["input"]

    if case.get("expect_load_error"):
        with pytest.raises(DomainPackError):
            parse_corrections(case["rules"])
        return

    rules = parse_corrections(case["rules"])
    max_output_chars = case.get("max_output_chars")
    result = apply_corrections(text, rules, max_output_chars=max_output_chars)

    if case.get("expect_growth_rejected"):
        assert result.growth_rejected is True
        assert result.text == text
        assert result.trace == ()
        return

    assert result.growth_rejected is False
    assert result.text == case["expected_text"]
    # Exact pinned trace (not engine internals as oracle).
    expected_trace = case["expected_trace"]
    actual_trace = [e.to_mapping() for e in result.trace]
    assert actual_trace == expected_trace
    # And the strong bidirectional faithfulness invariant.
    _assert_trace_faithful(result, text)
