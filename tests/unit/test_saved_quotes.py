"""Unit tests for the saved quotes CRUD module (issue #338, Phase 6)."""

from __future__ import annotations

import pytest

from voxint.api.saved_quotes import _csv_safe


class TestCsvSafe:
    def test_plain_text_unchanged(self) -> None:
        assert _csv_safe("hello world") == "hello world"

    def test_empty_string_unchanged(self) -> None:
        assert _csv_safe("") == ""

    @pytest.mark.parametrize("prefix", ["=", "+", "-", "@"])
    def test_formula_prefix_escaped(self, prefix: str) -> None:
        value = f"{prefix}SUM(A1:A10)"
        assert _csv_safe(value) == f"'{prefix}SUM(A1:A10)"

    def test_safe_hyphenated_word_not_escaped(self) -> None:
        # Only leading character matters, so a hyphen at position 0 IS escaped
        assert _csv_safe("-word") == "'-word"

    def test_number_string_unchanged(self) -> None:
        assert _csv_safe("42.5") == "42.5"
