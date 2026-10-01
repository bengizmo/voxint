"""Unit tests for read-time correction provenance (issue #83).

Covers every branch of the #83 truth table in
``src/voxint/adjudication/corrections_view.py``: snapshot-index resolution
(present / missing / corrupt) and per-segment provenance (fired / no-fire /
version mismatch / unresolved id / snapshot missing / malformed entry). All pure,
no DB, no I/O. The run-level reconciliation tests went with the dead
``run_reconciliation`` helper (#674).
"""

from __future__ import annotations

from typing import Any

from voxint.adjudication.corrections_view import (
    DeclaredRuleIndex,
    RuleDisplay,
    build_declared_rule_index,
    resolve_segment_provenance,
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
    assert list(index.by_id) == ["r1", "r2"]  # every declared rule resolvable
    assert index.by_id["r1"] == RuleDisplay(
        id="r1", pack="town", match="selectboard", replace="Selectboard"
    )


def test_index_empty_corrections_is_valid_empty_index() -> None:
    index = build_declared_rule_index(_snapshot("bare", []))
    assert index is not None
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
    # recordedVersion reports the side that actually MISMATCHES (the row's 99), not
    # the current-version envelope — the UI would otherwise say "recorded by v1"
    # while the console also reads v1.
    assert result["recordedVersion"] == 99


def test_provenance_all_malformed_entries_is_none() -> None:
    # trace_has_entries is true (non-empty list) but every entry is unrenderable:
    # a non-mapping, a non-string id, and a non-string from. No honest marker can be
    # shown, so the segment renders as uncorrected rather than "corrected by pack (0)".
    idx = _index("town", [_rule("r1", "abbr", "abbreviation")])
    trace: dict[str, Any] = {
        "version": CORRECTOR_VERSION,
        "input_base": "raw",
        "entries": [
            "not-a-mapping",
            {"id": 7, "from": "a", "to": "b", "span": [0, 1]},
            {"id": "r1", "from": None, "to": "b", "span": [0, 1]},
        ],
    }
    assert resolve_segment_provenance(trace, CORRECTOR_VERSION, idx) is None


def test_provenance_malformed_span_normalizes_to_none() -> None:
    # A corrupt span (wrong arity / non-int / bool) degrades to null while the entry
    # stays visible — the wire contract types span as [number, number] | null.
    idx = _index("town", [_rule("r1", "abbr", "abbreviation")])
    trace = _envelope([{"id": "r1", "from": "abbr", "to": "abbreviation", "span": [0]}])
    result = resolve_segment_provenance(trace, CORRECTOR_VERSION, idx)
    assert result is not None
    (entry,) = result["entries"]
    assert entry["span"] is None
    assert entry["id"] == "r1" and entry["resolved"] is True


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
