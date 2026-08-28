"""Unit tests for the benchmark WER scorer and normalizer."""

from __future__ import annotations

import pytest

from voxint.benchmark.scorer import (
    WERCounts,
    compute_wer,
    normalize,
    pool_wer,
    protocol_hash,
)


class TestNormalize:
    def test_lowercase(self) -> None:
        assert normalize("HELLO WORLD") == "hello world"

    def test_strip_punctuation(self) -> None:
        assert normalize("Hello, world!") == "hello world"

    def test_collapse_whitespace(self) -> None:
        assert normalize("hello   world") == "hello world"

    def test_strip_leading_trailing(self) -> None:
        assert normalize("  hello  ") == "hello"

    def test_tabs_and_newlines(self) -> None:
        assert normalize("hello\t\nworld") == "hello world"

    def test_empty_string(self) -> None:
        assert normalize("") == ""

    def test_punctuation_only(self) -> None:
        assert normalize("!@#$%^&*()") == ""

    def test_mixed_case_punctuation(self) -> None:
        assert normalize("It's a NICE day, isn't it?") == "its a nice day isnt it"

    def test_apostrophe_contraction(self) -> None:
        assert normalize("THAT'S THE WAY") == "thats the way"

    def test_unicode_preserved(self) -> None:
        assert normalize("café") == "café"


class TestComputeWER:
    def test_identical(self) -> None:
        result = compute_wer("hello world", "hello world")
        assert result == WERCounts(0, 0, 0, 2)
        assert result.wer == 0.0

    def test_all_wrong(self) -> None:
        result = compute_wer("hello world", "foo bar")
        assert result.substitutions == 2
        assert result.reference_words == 2
        assert result.wer == 1.0

    def test_empty_both(self) -> None:
        result = compute_wer("", "")
        assert result == WERCounts(0, 0, 0, 0)
        assert result.wer == 0.0

    def test_empty_reference_nonempty_hypothesis(self) -> None:
        result = compute_wer("", "hello world")
        assert result.insertions == 2
        assert result.reference_words == 0
        assert result.wer == 0.0

    def test_nonempty_reference_empty_hypothesis(self) -> None:
        result = compute_wer("hello world", "")
        assert result.deletions == 2
        assert result.reference_words == 2
        assert result.wer == 1.0

    def test_insertion(self) -> None:
        result = compute_wer("a b", "a c b")
        assert result.insertions == 1
        assert result.substitutions == 0
        assert result.deletions == 0

    def test_deletion(self) -> None:
        result = compute_wer("a b c", "a c")
        assert result.deletions == 1

    def test_substitution(self) -> None:
        result = compute_wer("a b c", "a x c")
        assert result.substitutions == 1
        assert result.insertions == 0
        assert result.deletions == 0

    def test_wer_over_100_percent(self) -> None:
        result = compute_wer("a", "x y z")
        assert result.wer > 1.0

    def test_normalization_applied(self) -> None:
        result = compute_wer("HELLO WORLD", "hello world")
        assert result.wer == 0.0

    def test_punctuation_ignored(self) -> None:
        result = compute_wer("Hello, world!", "hello world")
        assert result.wer == 0.0

    def test_known_counts(self) -> None:
        # "the cat sat on the mat" vs "the dog sat on a mat"
        # cat->dog (S), the->a (S) = 2 subs, 6 ref words
        result = compute_wer("the cat sat on the mat", "the dog sat on a mat")
        assert result.substitutions == 2
        assert result.insertions == 0
        assert result.deletions == 0
        assert result.reference_words == 6
        assert result.wer == pytest.approx(2 / 6)

    def test_mixed_operations(self) -> None:
        # ref: "a b c d" hyp: "a x d e"
        # a=match, b->x (S), c deleted (D), d=match, e inserted (I)
        result = compute_wer("a b c d", "a x d e")
        assert result.errors == 3
        assert result.reference_words == 4


class TestPoolWER:
    def test_single_count(self) -> None:
        counts = [WERCounts(1, 0, 0, 10)]
        assert pool_wer(counts) == pytest.approx(0.1)

    def test_multiple_counts(self) -> None:
        counts = [
            WERCounts(2, 1, 0, 10),  # 3 errors
            WERCounts(0, 0, 1, 5),  # 1 error
        ]
        assert pool_wer(counts) == pytest.approx(4 / 15)

    def test_empty_list(self) -> None:
        assert pool_wer([]) == 0.0

    def test_all_zero_reference(self) -> None:
        counts = [WERCounts(0, 2, 0, 0)]
        assert pool_wer(counts) == 0.0


class TestProtocolHash:
    def test_deterministic(self) -> None:
        h1 = protocol_hash()
        h2 = protocol_hash()
        assert h1 == h2

    def test_length(self) -> None:
        assert len(protocol_hash()) == 16

    def test_hex_string(self) -> None:
        h = protocol_hash()
        int(h, 16)


class TestWERCounts:
    def test_errors_property(self) -> None:
        c = WERCounts(2, 3, 1, 10)
        assert c.errors == 6

    def test_to_dict(self) -> None:
        c = WERCounts(1, 2, 3, 10)
        d = c.to_dict()
        assert d == {
            "substitutions": 1,
            "insertions": 2,
            "deletions": 3,
            "reference_words": 10,
        }
