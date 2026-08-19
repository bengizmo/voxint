"""Unit tests for read-time correction provenance + reconciliation (issue #83).

Covers every branch of the #83 truth table in
``src/voxint/adjudication/corrections_view.py``: snapshot-index resolution
(present / missing / corrupt), per-segment provenance (fired / no-fire / version
mismatch / unresolved id / snapshot missing / malformed entry), and run-level
reconciliation (applied / no_raw_match / raw-pass growth_rejected + the
applied > growth_rejected > no_raw_match precedence). All pure — no DB, no I/O.
"""

from __future__ import annotations

from typing import Any

from voxint.adjudication.corrections_view import (
    DeclaredRuleIndex,
    RuleDisplay,
    build_declared_rule_index,
    resolve_segment_provenance,
    run_reconciliation,
)
from voxint.domain_packs.corrector import CORRECTOR_VERSION

# --- fixtures ---------------------------------------------------------------


def _rule(
    rule_id: str,
    match: str,
    replace: str,
    *,
    case_sensitive: bool = True,
    whole_word: bool = True,
) -> dict[str, Any]:
    return {
        "id": rule_id,
        "match": match,
        "replace": replace,
        "case_sensitive": case_sensitive,
        "whole_word": whole_word,
    }


def _snapshot(name: str, rules: list[dict[str, Any]]) -> dict[str, Any]:
    return {"name": name, "corrections": rules}


def _envelope(
    entries: list[dict[str, Any]],
    *,
    version: int = CORRECTOR_VERSION,
    input_base: str = "raw",
) -> dict[str, Any]:
    return {"version": version, "input_base": input_base, "entries": entries}


def _entry(rule_id: str, from_text: str, to_text: str, span: list[int]) -> dict[str, Any]:
    return {"id": rule_id, "from": from_text, "to": to_text, "span": span}


# --- build_declared_rule_index ---------------------------------------------


def test_index_none_snapshot_is_none() -> None:
    assert build_declared_rule_index(None) is None


def test_index_non_mapping_snapshot_is_none() -> None:
    # A list (or any non-Mapping) is not a valid snapshot.
    assert build_declared_rule_index(["not", "a", "pack"]) is None  # type: ignore[arg-type]


def test_index_missing_name_is_none() -> None:
    assert build_declared_rule_index({"corrections": []}) is None


def test_index_non_str_name_is_none() -> None:
    assert build_declared_rule_index({"name": 7, "corrections": []}) is None


def test_index_corrupt_corrections_is_none() -> None:
    # Duplicate ids fail validate_corrections -> DomainPackError -> None (never
    # a fabricated default pack).
    snap = _snapshot("dup", [_rule("a", "one", "1"), _rule("a", "two", "2")])
    assert build_declared_rule_index(snap) is None


def test_index_valid_resolves_pack_rules_and_by_id() -> None:
    snap = _snapshot(
        "town",
        [_rule("r1", "selectboard", "Selectboard"), _rule("r2", "abbr", "abbreviation")],
    )
    index = build_declared_rule_index(snap)
    assert index is not None
    assert index.pack == "town"
    assert [r.id for r in index.rules] == ["r1", "r2"]  # manifest order preserved
    assert index.by_id["r1"] == RuleDisplay(
        id="r1", pack="town", match="selectboard", replace="Selectboard"
    )


def test_index_empty_corrections_is_valid_empty_index() -> None:
    index = build_declared_rule_index(_snapshot("bare", []))
    assert index is not None
    assert index.rules == ()
    assert index.by_id == {}


# --- resolve_segment_provenance --------------------------------------------


def _index(name: str, rules: list[dict[str, Any]]) -> DeclaredRuleIndex:
    index = build_declared_rule_index(_snapshot(name, rules))
    assert index is not None
    return index


def test_provenance_empty_list_trace_is_none() -> None:
    assert resolve_segment_provenance([], CORRECTOR_VERSION, None) is None


def test_provenance_none_trace_is_none() -> None:
    assert resolve_segment_provenance(None, CORRECTOR_VERSION, None) is None


def test_provenance_envelope_empty_entries_is_none() -> None:
    # An envelope with no entries did not materially fire (pure-LLM enhancement).
    assert resolve_segment_provenance(_envelope([]), CORRECTOR_VERSION, None) is None


def test_provenance_version_mismatch_in_envelope_is_unavailable() -> None:
    idx = _index("town", [_rule("r1", "abbr", "abbreviation")])
    trace = _envelope([_entry("r1", "abbr", "abbreviation", [0, 12])], version=2)
    result = resolve_segment_provenance(trace, CORRECTOR_VERSION, idx)
    assert result == {
        "status": "unavailable",
        "reason": "version_mismatch",
        "recordedVersion": 2,
    }


def test_provenance_version_mismatch_in_row_column_is_unavailable() -> None:
    # Envelope says v1 but the row's corrector_version column is a legacy/other
    # value -> refuse to replay with mismatched semantics.
    idx = _index("town", [_rule("r1", "abbr", "abbreviation")])
    trace = _envelope([_entry("r1", "abbr", "abbreviation", [0, 12])])
    result = resolve_segment_provenance(trace, 99, idx)
    assert result is not None
    assert result["status"] == "unavailable"
    assert result["reason"] == "version_mismatch"


def test_provenance_resolved_entries_carry_pack_and_rule() -> None:
    idx = _index("town", [_rule("r1", "abbr", "abbreviation")])
    trace = _envelope(
        [_entry("r1", "abbr", "abbreviation", [4, 16])], input_base="llm"
    )
    result = resolve_segment_provenance(trace, CORRECTOR_VERSION, idx)
    assert result is not None
    assert result["status"] == "shown"
    assert result["version"] == CORRECTOR_VERSION
    assert result["inputBase"] == "llm"
    assert result["entries"] == [
        {
            "id": "r1",
            "from": "abbr",
            "to": "abbreviation",
            "span": [4, 16],
            "pack": "town",
            "match": "abbr",
            "replace": "abbreviation",
            "resolved": True,
        }
    ]


def test_provenance_unknown_id_stays_visible_unresolved() -> None:
    idx = _index("town", [_rule("r1", "abbr", "abbreviation")])
    trace = _envelope([_entry("ghost", "x", "y", [0, 1])])
    result = resolve_segment_provenance(trace, CORRECTOR_VERSION, idx)
    assert result is not None
    (entry,) = result["entries"]
    assert entry["id"] == "ghost"
    assert entry["resolved"] is False
    assert entry["pack"] is None
    # The trace's own id/from/to/span are never dropped.
    assert entry["from"] == "x" and entry["to"] == "y" and entry["span"] == [0, 1]


def test_provenance_snapshot_missing_degrades_to_unresolved_but_shown() -> None:
    # index None (NULL/corrupt snapshot): still show the trace's own facts.
    trace = _envelope([_entry("r1", "abbr", "abbreviation", [0, 12])])
    result = resolve_segment_provenance(trace, CORRECTOR_VERSION, None)
    assert result is not None
    assert result["status"] == "shown"
    (entry,) = result["entries"]
    assert entry["resolved"] is False
    assert entry["pack"] is None
    assert entry["from"] == "abbr"


def test_provenance_malformed_entry_is_skipped() -> None:
    idx = _index("town", [_rule("r1", "abbr", "abbreviation")])
    trace = _envelope([_entry("r1", "abbr", "abbreviation", [0, 12])])
    trace["entries"].append("not-a-mapping")  # type: ignore[arg-type]
    result = resolve_segment_provenance(trace, CORRECTOR_VERSION, idx)
    assert result is not None
    assert len(result["entries"]) == 1  # the junk entry is dropped


# --- run_reconciliation -----------------------------------------------------

_GROWTH_REPLACE = "z" * 300  # long enough to overflow enhanced_size_ceiling on short raw


def test_reconciliation_none_index_is_empty() -> None:
    assert run_reconciliation(None, ["some raw text"]) == []


def test_reconciliation_no_rules_is_empty() -> None:
    idx = _index("bare", [])
    assert run_reconciliation(idx, ["some raw text"]) == []


def test_reconciliation_applied_counts_segments() -> None:
    idx = _index("town", [_rule("r1", "abbr", "abbreviation")])
    result = run_reconciliation(idx, ["the abbr here", "no match", "abbr again"])
    assert result == [
        {
            "id": "r1",
            "pack": "town",
            "match": "abbr",
            "replace": "abbreviation",
            "status": "applied",
            "appliedCount": 2,
        }
    ]


def test_reconciliation_no_raw_match() -> None:
    idx = _index("town", [_rule("r1", "abbr", "abbreviation")])
    result = run_reconciliation(idx, ["nothing relevant", "still nothing"])
    assert result[0]["status"] == "no_raw_match"
    assert result[0]["appliedCount"] == 0


def test_reconciliation_raw_pass_growth_rejected() -> None:
    # A short raw with a huge replacement overflows enhanced_size_ceiling: the
    # segment's raw transformation is growth-rejected, yet the rule DID match raw.
    idx = _index("town", [_rule("r1", "abbr", _GROWTH_REPLACE)])
    result = run_reconciliation(idx, ["the abbr here"])
    assert result[0]["status"] == "growth_rejected"
    assert result[0]["appliedCount"] == 0


def test_reconciliation_applied_beats_growth_rejected() -> None:
    # Same rule fires cleanly on one segment (short replacement fits) but would be
    # growth-rejected on another: precedence applied > growth_rejected.
    idx = _index("town", [_rule("r1", "x", "y" * 300)])
    # First segment: raw long enough that y*300 fits under the ceiling; second:
    # raw so short the replacement overflows.
    long_raw = "x " + ("q" * 100)  # ceiling = (2+100)*4+200 = 608 > ~301 output
    short_raw = "x"  # ceiling = 1*4+200 = 204 < 300 output -> rejected
    result = run_reconciliation(idx, [long_raw, short_raw])
    assert result[0]["status"] == "applied"
    assert result[0]["appliedCount"] == 1
