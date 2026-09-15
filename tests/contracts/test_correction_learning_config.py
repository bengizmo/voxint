"""Contract tests for learned-corrections constants and invariants (#476).

Pins the learning module's constants and verifies that learned rules are
always case-sensitive whole-word, so the frozen per-run snapshot format
never silently changes.
"""

from __future__ import annotations

from voxint.domain_packs.corrections import CorrectionRule, parse_corrections
from voxint.domain_packs.learning import (
    LEARN_THRESHOLD_SEGMENTS,
    MAX_BASE_WORDS_FOR_LEARNING,
    MAX_LEARNED_WORDS_PER_SIDE,
    MAX_PAIRS_PER_SEGMENT,
    MAX_SUGGESTIONS_PER_PROJECT,
)


class TestConstantsPinned:
    def test_threshold(self) -> None:
        assert LEARN_THRESHOLD_SEGMENTS == 3

    def test_max_words_per_side(self) -> None:
        assert MAX_LEARNED_WORDS_PER_SIDE == 4

    def test_max_pairs_per_segment(self) -> None:
        assert MAX_PAIRS_PER_SEGMENT == 4

    def test_max_base_words(self) -> None:
        assert MAX_BASE_WORDS_FOR_LEARNING == 400

    def test_max_suggestions(self) -> None:
        assert MAX_SUGGESTIONS_PER_PROJECT == 256


class TestLearnedRuleShape:
    def test_learned_rules_always_case_sensitive_whole_word(self) -> None:
        """A learned rule entering the corrections list must be case-sensitive
        and whole-word to avoid surprising substitutions."""
        rule = CorrectionRule(
            id="learned-seer",
            match="seer",
            replace="SEER",
            case_sensitive=True,
            whole_word=True,
        )
        parsed = parse_corrections([rule.to_mapping()])
        assert parsed[0].case_sensitive is True
        assert parsed[0].whole_word is True

    def test_docs_example_parses(self) -> None:
        """The typical learned pair from the docs parses cleanly."""
        mapping = {
            "id": "seer",
            "match": "seer",
            "replace": "SEER",
            "case_sensitive": True,
            "whole_word": True,
        }
        parsed = parse_corrections([mapping])
        assert len(parsed) == 1
        assert parsed[0].match == "seer"
        assert parsed[0].replace == "SEER"
