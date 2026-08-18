"""Contract: the deterministic corrector's version + trace shape stay pinned (#81).

The corrector is pure and deterministic, but its output is **persisted** by #82
(into ``enhanced_text`` + a ``correction_trace`` column), so both the version and
the serialized trace shape are a versioned compatibility contract — a change to
either can silently invalidate historical, already-adjudicated renderings. This
pins the version integer AND one exact ``to_mapping()`` example (Unicode surface +
offset-shifted spans), plus the growth-rejection and no-op invariants that define
the engine's persisted behavior. Mirrors ``test_triage_config.py``.
"""

from __future__ import annotations

from voxint.domain_packs.corrections import CorrectionRule
from voxint.domain_packs.corrector import (
    CORRECTOR_VERSION,
    AppliedCorrection,
    apply_corrections,
)


def test_corrector_version_pinned() -> None:
    # Bumping this is deliberate: it marks persisted traces as produced by a
    # different engine, and #82 stores it beside each segment's trace.
    assert CORRECTOR_VERSION == 1


def test_applied_correction_mapping_shape_pinned() -> None:
    # The exact JSON keys/shape #82 persists. Unicode `from`, and a `span` that is
    # a two-element [start, end] list in the FINAL string's coordinate space.
    entry = AppliedCorrection(
        id="nogyo",
        from_text="農業",
        to_text="Agriculture",
        span=(26, 37),
    )
    assert entry.to_mapping() == {
        "id": "nogyo",
        "from": "農業",
        "to": "Agriculture",
        "span": [26, 37],
    }


def test_trace_span_is_final_string_coordinate() -> None:
    # An expanding replacement shifts the second span; both address result.text.
    rule = CorrectionRule(id="x", match="x", replace="WIDE", whole_word=False)
    result = apply_corrections("x.x", [rule])
    assert result.text == "WIDE.WIDE"
    mappings = [e.to_mapping() for e in result.trace]
    assert mappings == [
        {"id": "x", "from": "x", "to": "WIDE", "span": [0, 4]},
        {"id": "x", "from": "x", "to": "WIDE", "span": [5, 9]},
    ]
    for entry in result.trace:
        start, end = entry.span
        assert result.text[start:end] == entry.to_text


def test_growth_rejection_is_atomic_and_no_op_never_rejects() -> None:
    grow = CorrectionRule(id="g", match="a", replace="XXXX", whole_word=False)
    rejected = apply_corrections("aaa", [grow], max_output_chars=6)
    assert rejected.text == "aaa"
    assert rejected.trace == ()
    assert rejected.growth_rejected is True

    # A pure no-op never trips the bound, even when the untouched input is already
    # larger than the limit — the engine only rejects a transformation it makes.
    inert = CorrectionRule(id="z", match="absent", replace="Q", whole_word=False)
    noop = apply_corrections("aaaaaa", [inert], max_output_chars=2)
    assert noop.text == "aaaaaa"
    assert noop.growth_rejected is False
