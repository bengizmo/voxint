"""Pure extraction of substitution candidates from operator transcript edits (#476).

No DB, no I/O. Mirrors corrector.py: constants and a single deterministic
function that extracts literal-substitution pairs from a base/edited text diff.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from voxint.domain_packs.base import DomainPackError
from voxint.domain_packs.corrections import (
    CorrectionRule,
    find_first,
    is_word_char,
    parse_corrections,
)

LEARN_THRESHOLD_SEGMENTS = 3
MAX_LEARNED_WORDS_PER_SIDE = 4
MAX_PAIRS_PER_SEGMENT = 4
MAX_BASE_WORDS_FOR_LEARNING = 400
MAX_SUGGESTIONS_PER_PROJECT = 256


@dataclass(frozen=True)
class Substitution:
    match: str
    replace: str


def _tokenize(text: str) -> list[str]:
    """Split text into alternating word / non-word runs."""
    runs: list[str] = []
    current: list[str] = []
    in_word: bool | None = None
    for ch in text:
        ch_word = is_word_char(ch)
        if in_word is not None and ch_word != in_word:
            runs.append("".join(current))
            current = []
        current.append(ch)
        in_word = ch_word
    if current:
        runs.append("".join(current))
    return runs


def _count_words(runs: list[str]) -> int:
    return sum(1 for r in runs if r and is_word_char(r[0]))


def _word_count_in_span(runs: list[str], start: int, end: int) -> int:
    return sum(1 for r in runs[start:end] if r and is_word_char(r[0]))


def extract_substitutions(
    base: str, edited: str, *, raw: str
) -> tuple[Substitution, ...]:
    """Extract literal-substitution pairs from a base/edited diff.

    Returns a deterministic, order-preserving, deduplicated tuple capped at
    MAX_PAIRS_PER_SEGMENT. Each pair has 1..MAX_LEARNED_WORDS_PER_SIDE word
    tokens per side, passes parse_corrections validation, and matches both
    ``base`` and ``raw`` under whole-word case-sensitive semantics.
    """
    base_runs = _tokenize(base)
    base_words = _count_words(base_runs)
    if base_words > MAX_BASE_WORDS_FOR_LEARNING:
        return ()

    edited_runs = _tokenize(edited)
    matcher = difflib.SequenceMatcher(None, base_runs, edited_runs, autojunk=False)
    opcodes = matcher.get_opcodes()

    # First pass: total ALL replaced base words to detect wholesale rewrites.
    replaced_word_count = sum(
        _word_count_in_span(base_runs, i1, i2)
        for tag, i1, i2, _, _ in opcodes
        if tag == "replace"
    )
    max_replaced = max(MAX_LEARNED_WORDS_PER_SIDE, base_words // 2)
    if replaced_word_count > max_replaced:
        return ()

    # Second pass: extract learnable candidates.
    candidates: list[Substitution] = []
    seen: set[tuple[str, str]] = set()

    for tag, i1, i2, j1, j2 in opcodes:
        if tag != "replace":
            continue

        match_text = "".join(base_runs[i1:i2]).strip()
        replace_text = "".join(edited_runs[j1:j2]).strip()

        if not match_text or not replace_text:
            continue
        if match_text == replace_text:
            continue

        match_words = _word_count_in_span(base_runs, i1, i2)
        replace_words = _word_count_in_span(edited_runs, j1, j2)

        if match_words < 1 or match_words > MAX_LEARNED_WORDS_PER_SIDE:
            continue
        if replace_words < 1 or replace_words > MAX_LEARNED_WORDS_PER_SIDE:
            continue

        key = (match_text, replace_text)
        if key in seen:
            continue
        seen.add(key)

        rule = CorrectionRule(
            id="x", match=match_text, replace=replace_text,
            case_sensitive=True, whole_word=True,
        )
        try:
            parse_corrections([rule.to_mapping()])
        except DomainPackError:
            continue

        if find_first(rule, base) is None:
            continue
        if find_first(rule, raw) is None:
            continue

        candidates.append(Substitution(match=match_text, replace=replace_text))

        if len(candidates) >= MAX_PAIRS_PER_SEGMENT:
            break

    return tuple(candidates)
