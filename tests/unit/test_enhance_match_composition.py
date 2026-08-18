"""Pure dual-pass composition for the enhance_match stage (issue #82, epic #78).

Covers :func:`voxint.pipeline.stages.enhance_match._compose_correction` — the
raw-gated dual pass that composes a pack's deterministic corrections with the
(optional) LLM enhancement — with NO database. The engine mechanics live in
``test_corrector.py``; this pins the COMPOSITION policy on top of it:

- raw base (LLM off/failed) vs llm base (enforcement on the LLM output);
- only rules that matched raw are enforced, so a surface the LLM invents is never
  blessed as operator-authored (design report §12-F6);
- the frozen ID-set enforcement (a matched-raw rule applies to all its
  occurrences in the LLM output — decision C);
- ``changed`` (final != raw) drives whether the stage persists anything; and
- growth rejection never fabricates a trace entry, on either base (decision D).

The bulk is a data-declared corpus in ``tests/fixtures/compose_dual_pass/*.json``
(mirroring the ``rules_correct/*.json`` convention); the growth-rejection cases
need a programmatic oversized replacement, so they are explicit functions. The
shared ``enhanced_size_ceiling`` regression pins that the corrector reuses the
LLM path's existing per-segment growth bound unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from voxint.clients.llm import (
    MAX_ENHANCED_GROWTH_FACTOR,
    MAX_ENHANCED_SLACK_CHARS,
    enhanced_size_ceiling,
)
from voxint.domain_packs.corrections import parse_corrections
from voxint.domain_packs.corrector import apply_corrections
from voxint.pipeline.stages.enhance_match import _compose_correction

COMPOSE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "compose_dual_pass"


def _compose_from_case(case: dict[str, Any]) -> Any:
    """Drive ``_compose_correction`` exactly as ``run()`` does: rules round-trip
    through ``parse_corrections`` (so a case cannot smuggle an invalid set past
    load validation), the raw pass is precomputed under the shared ceiling, and
    the (possibly null) ``enhanced`` text is the second-pass base."""
    rules = parse_corrections(case["rules"])
    raw = case["raw"]
    raw_result = apply_corrections(
        raw, rules, max_output_chars=enhanced_size_ceiling(raw)
    )
    return _compose_correction(raw, raw_result, case["enhanced"], rules)


# --------------------------------------------------------------------------- #
# Data-declared composition corpus.
# --------------------------------------------------------------------------- #
def _load_corpus() -> list[tuple[str, dict[str, Any]]]:
    if not COMPOSE_DIR.is_dir():
        return []
    cases: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(COMPOSE_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        cases.append((path.stem, data))
    return cases


CORPUS = _load_corpus()


def test_corpus_is_present() -> None:
    # Guard against an empty glob silently passing the parametrized gate.
    assert CORPUS, "composition corpus tests/fixtures/compose_dual_pass/ is empty"


@pytest.mark.parametrize("name,case", CORPUS, ids=[c[0] for c in CORPUS])
def test_compose_dual_pass_corpus(name: str, case: dict[str, Any]) -> None:
    """One frozen composition case: exact final text, input_base, trace, changed."""
    composition = _compose_from_case(case)
    assert composition.final_text == case["expected_final_text"]
    assert composition.input_base == case["expected_input_base"]
    assert [dict(entry) for entry in composition.entries] == case["expected_trace"]
    assert composition.changed is case["expected_changed"]


def test_corpus_covers_both_input_bases() -> None:
    # The corpus must exercise both provenance bases, or a regression collapsing
    # one path would pass unnoticed.
    bases = {case["expected_input_base"] for _, case in CORPUS}
    assert bases == {"raw", "llm"}


# --------------------------------------------------------------------------- #
# Growth rejection never fabricates a trace entry (decision D) — programmatic
# oversized replacement, so kept out of the readable-text JSON corpus.
# --------------------------------------------------------------------------- #
def test_growth_rejected_on_raw_base_persists_nothing() -> None:
    # A replacement that would balloon the raw text past its own ceiling is
    # rejected whole by the engine; with the LLM off, the raw-pass result stands
    # unchanged, so composition reports no change and no entry.
    rules = parse_corrections(
        [{"id": "g", "match": "a", "replace": "Q" * 205, "whole_word": False}]
    )
    raw = "a"  # ceiling 204; projected 205 -> rejected
    raw_result = apply_corrections(raw, rules, max_output_chars=enhanced_size_ceiling(raw))
    assert raw_result.growth_rejected is True

    composition = _compose_correction(raw, raw_result, None, rules)
    assert composition.final_text == "a"
    assert composition.input_base == "raw"
    assert composition.entries == ()
    assert composition.changed is False


def test_growth_rejected_on_llm_base_enforces_nothing() -> None:
    # The rule fires on raw (so it is in the enforcement set), but re-applying it
    # to a shorter LLM output would exceed THAT text's ceiling, so the enforcement
    # pass is rejected whole: the oversized replacement never lands and no entry is
    # fabricated. ``changed`` still reflects the LLM's own edit (enhanced != raw).
    rules = parse_corrections(
        [{"id": "g", "match": "zz", "replace": "Q" * 209, "whole_word": True}]
    )
    raw = "zz "  # ceiling 212; projected 210 -> fires, id in raw-fire set
    enhanced = "zz"  # ceiling 208; projected 209 -> enforcement rejected
    raw_result = apply_corrections(raw, rules, max_output_chars=enhanced_size_ceiling(raw))
    assert {entry.id for entry in raw_result.trace} == {"g"}

    composition = _compose_correction(raw, raw_result, enhanced, rules)
    assert composition.final_text == "zz"  # enforcement rejected -> LLM text intact
    assert composition.input_base == "llm"
    assert composition.entries == ()
    assert ("Q" * 209) not in composition.final_text
    assert composition.changed is True  # enhanced ("zz") differs from raw ("zz ")


# --------------------------------------------------------------------------- #
# Shared growth ceiling — the corrector reuses the LLM path's existing bound.
# --------------------------------------------------------------------------- #
def test_enhanced_size_ceiling_constants_pinned() -> None:
    # The composition growth bound IS this ceiling (decision D); the LLM per-segment
    # reply bound reuses the same helper. Pin the constants so neither drifts.
    assert MAX_ENHANCED_GROWTH_FACTOR == 4
    assert MAX_ENHANCED_SLACK_CHARS == 200


@pytest.mark.parametrize("length", [0, 1, 12, 500])
def test_enhanced_size_ceiling_formula(length: int) -> None:
    text = "x" * length
    assert enhanced_size_ceiling(text) == length * 4 + 200
