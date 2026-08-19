"""Contract: the persisted #82 correction-trace envelope + provenance stay pinned.

``test_corrector_config.py`` pins the pure engine's version and per-entry
``to_mapping`` shape. This pins the COMPOSITION layer's durable surface — the
envelope that ``enhance_match`` writes into the ``correction_trace`` column and
stamps into ``corrector_version`` — because those are read back by later,
already-adjudicated renderings (and by #83's console), so a silent change would
invalidate historical rows:

- the envelope object shape ``{"version", "input_base", "entries"}`` with an
  exact example (matching what ``run()`` persists for a rules-only correction);
- the closed ``input_base`` literal set ``{"raw", "llm"}`` (the only two
  provenance bases the dual pass can emit); and
- the zero-drift invariant that the shipped ``generic`` pack declares NO
  corrections, so a default install's pipeline behavior is byte-unchanged.
"""

from __future__ import annotations

from voxint.clients.llm import enhanced_size_ceiling
from voxint.domain_packs.base import load_default
from voxint.domain_packs.corrections import parse_corrections
from voxint.domain_packs.corrector import CORRECTOR_VERSION, apply_corrections
from voxint.pipeline.stages.enhance_match import _compose_correction, _Composition

# The closed provenance-base set the dual pass may emit. Widening this changes a
# persisted, versioned value; do it deliberately with a version bump.
INPUT_BASE_LITERALS = frozenset({"raw", "llm"})


def _persisted_envelope(composition: _Composition) -> dict[str, object]:
    """The exact envelope ``enhance_match.run`` writes to ``correction_trace``
    for a materially-changed segment (kept identical to the stage's writer)."""
    assert composition.changed  # only a changed segment persists an envelope
    return {
        "version": CORRECTOR_VERSION,
        "input_base": composition.input_base,
        "entries": list(composition.entries),
    }


def test_persisted_envelope_shape_pinned() -> None:
    # A rules-only correction (LLM off) — the simplest path that persists an
    # envelope. The shape and every key/value are the durable contract.
    rules = parse_corrections(
        [{"id": "zb", "match": "zoom board", "replace": "Zoning Board", "whole_word": True}]
    )
    raw = "the zoom board met"
    raw_result = apply_corrections(raw, rules, max_output_chars=enhanced_size_ceiling(raw))
    composition = _compose_correction(raw, raw_result, None, rules)

    assert _persisted_envelope(composition) == {
        "version": 1,
        "input_base": "raw",
        "entries": [
            {"id": "zb", "from": "zoom board", "to": "Zoning Board", "span": [4, 16]},
        ],
    }


def test_input_base_literal_set_is_closed() -> None:
    # Both provenance bases are reachable, and only these two. LLM off -> "raw".
    rules = parse_corrections(
        [{"id": "zb", "match": "zoom board", "replace": "Zoning Board", "whole_word": True}]
    )
    raw = "the zoom board met"
    raw_result = apply_corrections(raw, rules, max_output_chars=enhanced_size_ceiling(raw))

    off = _compose_correction(raw, raw_result, None, rules)
    on = _compose_correction(raw, raw_result, "the zoom board met.", rules)
    assert off.input_base == "raw"
    assert on.input_base == "llm"
    assert {off.input_base, on.input_base} == INPUT_BASE_LITERALS
    assert frozenset({"raw", "llm"}) == INPUT_BASE_LITERALS


def test_generic_pack_declares_no_corrections() -> None:
    # Zero-drift: the default install ships no correction rules, so its enhance
    # pipeline behaves exactly as before #82 (raw pass is a no-op on every segment).
    assert load_default().corrections == ()
