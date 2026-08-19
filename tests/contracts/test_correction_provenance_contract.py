"""Contract: the #83 read-time provenance + reconciliation display surface.

``test_corrector_composition_config.py`` pins the PERSISTED envelope
(``correction_trace``); this pins the READ side — the console-facing shapes
:mod:`voxint.adjudication.corrections_view` derives from that envelope. The React
island's ``Segment.corrections`` / reconciliation types mirror these keys, so a
silent change here would rot the console. Specifically:

- the closed reconciliation status set ``{applied, no_raw_match, growth_rejected}``
  (widening it is a UI + docs change, never accidental);
- the per-segment provenance object shape (``status``/``version``/``inputBase``/
  ``entries`` and the per-entry ``id/from/to/span/pack/match/replace/resolved``);
- the reconciliation entry shape (``id/pack/match/replace/status/appliedCount``);
- the load-bearing invariant that provenance keys off ``trace_has_entries`` — the
  canonical "did a rule materially fire" predicate — NOT a text diff: an
  empty-entries envelope yields no marker even though effective≠raw.
"""

from __future__ import annotations

import typing

from voxint.adjudication.corrections_view import (
    ReconStatus,
    build_declared_rule_index,
    resolve_segment_provenance,
    run_reconciliation,
)
from voxint.domain_packs.corrector import CORRECTOR_VERSION

# The closed reconciliation status set. Widening this is a deliberate UI + docs
# change (a new "declared but never fired" reason), never an accidental drift.
RECON_STATUSES = frozenset({"applied", "no_raw_match", "growth_rejected"})

PROVENANCE_ENTRY_KEYS = frozenset(
    {"id", "from", "to", "span", "pack", "match", "replace", "resolved"}
)
PROVENANCE_SHOWN_KEYS = frozenset({"status", "version", "inputBase", "entries"})
RECON_ENTRY_KEYS = frozenset(
    {"id", "pack", "match", "replace", "status", "appliedCount"}
)


def _index(name: str, rules: list[dict[str, object]]):
    idx = build_declared_rule_index({"name": name, "corrections": rules})
    assert idx is not None
    return idx


def test_recon_status_literal_matches_the_pinned_set() -> None:
    # The Literal in code and the contract's frozenset must not drift apart.
    assert frozenset(typing.get_args(ReconStatus)) == RECON_STATUSES


def test_shown_provenance_object_shape_is_pinned() -> None:
    idx = _index("town", [{"id": "r1", "match": "abbr", "replace": "abbreviation"}])
    trace = {
        "version": CORRECTOR_VERSION,
        "input_base": "raw",
        "entries": [{"id": "r1", "from": "abbr", "to": "abbreviation", "span": [0, 12]}],
    }
    result = resolve_segment_provenance(trace, CORRECTOR_VERSION, idx)
    assert result is not None
    assert set(result) == PROVENANCE_SHOWN_KEYS
    assert result["status"] == "shown"
    (entry,) = result["entries"]
    assert set(entry) == PROVENANCE_ENTRY_KEYS


def test_reconciliation_entry_shape_and_status_are_pinned() -> None:
    idx = _index("town", [{"id": "r1", "match": "abbr", "replace": "abbreviation"}])
    (entry,) = run_reconciliation(idx, ["the abbr here"])
    assert set(entry) == RECON_ENTRY_KEYS
    assert entry["status"] in RECON_STATUSES


def test_provenance_keys_off_trace_predicate_not_a_text_diff() -> None:
    # An empty-entries envelope (a pure-LLM enhancement: effective text DIFFERS
    # from raw, but NO deterministic rule fired) must yield no provenance marker —
    # proving the read side uses trace_has_entries, never a raw-vs-effective diff.
    empty_envelope = {"version": CORRECTOR_VERSION, "input_base": "llm", "entries": []}
    assert resolve_segment_provenance(empty_envelope, CORRECTOR_VERSION, None) is None
