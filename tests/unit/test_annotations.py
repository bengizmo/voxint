"""Pure anchor mapping / classification / hashing matrix (issue #86).

Exercises ``voxint.adjudication.annotations`` with no database: build
``TranscriptSegment`` objects in memory, wrap them as ``CoveredSegment``, and
drive ``derive_anchor`` across the full classification truth table plus the
coordinate-mapping edge cases the plan's Phase 0 froze (split endpoints, outer
whitespace, reverse selections, emoji/combining marks, words-NULL degrade). The
hash has golden and framing-sensitivity pins.
"""

import dataclasses
import uuid
from typing import Any

import pytest

from voxint.adjudication.annotations import (
    SEGMENT_RANGE,
    TEXT_RANGE,
    TIMING_SEGMENT,
    TIMING_WORD,
    WORD_RANGE,
    AnnotationStaleError,
    AnnotationValidationError,
    CaptureEndpoint,
    CapturePayload,
    CoveredSegment,
    DerivedAnchor,
    ResolvedAnnotation,
    ResolvedSpan,
    StoredAnchor,
    annotation_source_hash,
    clip_lines_for_export,
    derive_anchor,
    resolve_annotation_spans,
    resolved_order_key,
    stored_anchor_from_row,
    word_eligible,
)
from voxint.adjudication.splits import derive_children
from voxint.adjudication.transcript import TranscriptLine
from voxint.db.models import (
    MAX_ANNOTATION_SPAN_SEGMENTS,
    TranscriptAnnotation,
    TranscriptSegment,
)

_A = uuid.UUID("11111111-1111-1111-1111-111111111111")
_B = uuid.UUID("22222222-2222-2222-2222-222222222222")


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _tokens_from_spaced(raw: str, start: float, end: float) -> list[dict[str, object]]:
    """Whisper-like tokens for a single-space-separated string: the first word is
    bare, each later word carries its leading space, so the tokens reconcatenate
    to ``raw`` exactly. Timings are spread evenly over ``[start, end)``."""
    pieces = raw.split(" ")
    n = len(pieces)
    step = (end - start) / n
    tokens: list[dict[str, object]] = []
    t = start
    for i, w in enumerate(pieces):
        text = w if i == 0 else " " + w
        tokens.append({"word": text, "start": round(t, 6), "end": round(t + step, 6)})
        t += step
    return tokens


def _segment(
    raw: str,
    *,
    seg_id: uuid.UUID | None = None,
    index: int = 0,
    seconds: tuple[float, float] = (0.0, 3.0),
    with_words: bool = True,
    words: list[dict[str, object]] | None = None,
    enhanced: str | None = None,
    trace: dict[str, Any] | list[Any] | None = None,
) -> TranscriptSegment:
    seg = TranscriptSegment()
    seg.id = seg_id or uuid.uuid4()
    seg.pipeline_run_id = uuid.uuid4()
    seg.segment_index = index
    seg.start_seconds, seg.end_seconds = seconds
    seg.raw_text = raw
    seg.enhanced_text = enhanced
    seg.confidence = None
    seg.diarization_label = None
    if words is not None:
        seg.words = words
    elif with_words:
        seg.words = _tokens_from_spaced(raw, seconds[0], seconds[1])
    else:
        seg.words = None
    seg.correction_trace = trace if trace is not None else []
    seg.corrector_version = None
    return seg


def _covered(
    seg: TranscriptSegment,
    *,
    corrected_text: str | None = None,
    cuts: tuple[int, ...] = (),
) -> CoveredSegment:
    return CoveredSegment(
        segment=seg,
        segment_index=seg.segment_index,
        corrected_text=corrected_text,
        cuts=cuts,
    )


def _ep(
    seg: TranscriptSegment, offset: int, child: tuple[int, int] | None = None
) -> CaptureEndpoint:
    cws, cwe = child if child is not None else (None, None)
    return CaptureEndpoint(
        segment_id=seg.id, offset=offset, child_word_start=cws, child_word_end=cwe
    )


# A single well-known 3-word segment: "Hello world there"
#   content_start = [0, 6, 12]; content_end = [5, 11, 17]; len = 17.
RAW3 = "Hello world there"


def _seg3(seg_id: uuid.UUID | None = None, index: int = 0) -> TranscriptSegment:
    return _segment(RAW3, seg_id=seg_id, index=index, seconds=(0.0, 3.0))


# --------------------------------------------------------------------------- #
# Hash
# --------------------------------------------------------------------------- #


def test_hash_golden() -> None:
    assert (
        annotation_source_hash([(_A, "Hello world there")])
        == "cb96a27e5a914ef0357208e28d197bbc7da93c8a0aec35e8f70e7386bd8d57e0"
    )


def test_hash_is_length_framed_not_concatenation() -> None:
    # Same byte concatenation, different segment partition -> different hash.
    assert annotation_source_hash([(_A, "ab"), (_B, "c")]) != annotation_source_hash(
        [(_A, "a"), (_B, "bc")]
    )


def test_hash_includes_segment_id() -> None:
    assert annotation_source_hash([(_A, "same")]) != annotation_source_hash([(_B, "same")])


def test_hash_is_hex64() -> None:
    digest = annotation_source_hash([(_A, "x")])
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


# --------------------------------------------------------------------------- #
# word_eligible
# --------------------------------------------------------------------------- #


def test_word_eligible_raw_with_words() -> None:
    assert word_eligible(_seg3(), None) is not None


def test_word_eligible_false_when_corrected() -> None:
    assert word_eligible(_seg3(), "an operator correction") is None


def test_word_eligible_false_when_enhanced_differs() -> None:
    seg = _segment(RAW3, enhanced="Hi world there")
    assert word_eligible(seg, None) is None


def test_word_eligible_false_when_no_words() -> None:
    assert word_eligible(_segment(RAW3, with_words=False), None) is None


def test_word_eligible_false_when_correction_trace_fired() -> None:
    trace = {"version": 1, "input_base": "raw", "entries": [{"id": "r", "from": "a", "to": "b"}]}
    assert word_eligible(_segment(RAW3, trace=trace), None) is None


# --------------------------------------------------------------------------- #
# Classification truth table (single segment)
# --------------------------------------------------------------------------- #


def test_whole_selection_is_segment_range() -> None:
    seg = _seg3()
    cov = [_covered(seg)]
    d = derive_anchor(cov, CapturePayload(_ep(seg, 0), _ep(seg, 17), "Hello world there"))
    assert d.anchor_kind == SEGMENT_RANGE
    assert d.start_word_index is None and d.start_char_offset is None
    assert d.start_seconds is None and d.timing_precision == TIMING_SEGMENT
    assert d.quote_text == "Hello world there"


def test_word_boundary_selection_is_word_range() -> None:
    seg = _seg3()
    cov = [_covered(seg)]
    d = derive_anchor(cov, CapturePayload(_ep(seg, 6), _ep(seg, 11), "world"))
    assert d.anchor_kind == WORD_RANGE
    assert (d.start_word_index, d.end_word_index) == (1, 2)
    assert d.start_char_offset is None
    assert d.timing_precision == TIMING_WORD
    assert d.start_seconds == pytest.approx(1.0) and d.end_seconds == pytest.approx(2.0)
    assert d.quote_text == "world"


def test_sub_word_selection_is_text_range() -> None:
    seg = _seg3()
    cov = [_covered(seg)]
    d = derive_anchor(cov, CapturePayload(_ep(seg, 7), _ep(seg, 11), "orld"))
    assert d.anchor_kind == TEXT_RANGE
    assert (d.start_char_offset, d.end_char_offset) == (7, 11)
    assert d.start_word_index is None
    assert d.start_seconds is None and d.timing_precision == TIMING_SEGMENT
    assert d.quote_text == "orld"


def test_multi_word_boundary_selection_is_word_range() -> None:
    seg = _seg3()
    cov = [_covered(seg)]
    d = derive_anchor(cov, CapturePayload(_ep(seg, 0), _ep(seg, 11), "Hello world"))
    assert d.anchor_kind == WORD_RANGE
    assert (d.start_word_index, d.end_word_index) == (0, 2)


# --------------------------------------------------------------------------- #
# Reverse (direction normalization)
# --------------------------------------------------------------------------- #


def test_reverse_selection_normalized() -> None:
    seg = _seg3()
    cov = [_covered(seg)]
    # start after end within one segment -> normalized to word_range "world".
    d = derive_anchor(cov, CapturePayload(_ep(seg, 11), _ep(seg, 6), "world"))
    assert d.anchor_kind == WORD_RANGE
    assert (d.start_word_index, d.end_word_index) == (1, 2)


def test_empty_selection_rejected() -> None:
    seg = _seg3()
    cov = [_covered(seg)]
    with pytest.raises(AnnotationValidationError):
        derive_anchor(cov, CapturePayload(_ep(seg, 6), _ep(seg, 6), ""))


# --------------------------------------------------------------------------- #
# Raw / enhanced / corrected
# --------------------------------------------------------------------------- #


def test_corrected_segment_selection_is_text_range() -> None:
    seg = _seg3()
    cov = [_covered(seg, corrected_text="Corrected words here")]
    # offsets index the corrected effective text; not word-eligible -> text_range.
    d = derive_anchor(cov, CapturePayload(_ep(seg, 0), _ep(seg, 9), "Corrected"))
    assert d.anchor_kind == TEXT_RANGE
    assert d.quote_text == "Corrected"


def test_enhanced_material_selection_is_text_range() -> None:
    seg = _segment(RAW3, enhanced="Hiya world there")
    cov = [_covered(seg)]
    d = derive_anchor(cov, CapturePayload(_ep(seg, 0), _ep(seg, 4), "Hiya"))
    assert d.anchor_kind == TEXT_RANGE
    assert d.quote_text == "Hiya"


# --------------------------------------------------------------------------- #
# words IS NULL degrade
# --------------------------------------------------------------------------- #


def test_words_null_partial_is_segment_range_verbatim_quote() -> None:
    seg = _segment("no timings here", with_words=False)
    cov = [_covered(seg)]
    d = derive_anchor(cov, CapturePayload(_ep(seg, 3), _ep(seg, 10), "timings"))
    assert d.anchor_kind == SEGMENT_RANGE
    assert d.start_char_offset is None and d.start_word_index is None
    # The sub-selection is preserved verbatim as the quote even though the anchor
    # degrades to whole-segment.
    assert d.quote_text == "timings"
    assert d.timing_precision == TIMING_SEGMENT


def test_words_null_whole_is_segment_range() -> None:
    seg = _segment("no timings here", with_words=False)
    cov = [_covered(seg)]
    d = derive_anchor(cov, CapturePayload(_ep(seg, 0), _ep(seg, 15), "no timings here"))
    assert d.anchor_kind == SEGMENT_RANGE


# --------------------------------------------------------------------------- #
# Split endpoints + projection stability
# --------------------------------------------------------------------------- #


def test_whole_split_child_is_word_range() -> None:
    seg = _seg3()
    # Split before word 1: children [0,1)="Hello", [1,3)="world there".
    cov = [_covered(seg, cuts=(1,))]
    # Select the whole second child (word_start=1, word_end=3, offset 0..len).
    child = (1, 3)
    d = derive_anchor(
        cov,
        CapturePayload(_ep(seg, 0, child), _ep(seg, len("world there"), child), "world there"),
    )
    assert d.anchor_kind == WORD_RANGE
    assert (d.start_word_index, d.end_word_index) == (1, 3)
    assert d.quote_text == "world there"


def test_split_child_partial_maps_to_parent_text_range() -> None:
    seg = _seg3()
    cov = [_covered(seg, cuts=(1,))]
    child = (1, 3)  # "world there"
    # Select "orld there" -> child-local offset 1..len -> parent 7..17, sub-word.
    d = derive_anchor(
        cov,
        CapturePayload(_ep(seg, 1, child), _ep(seg, len("world there"), child), "orld there"),
    )
    assert d.anchor_kind == TEXT_RANGE
    assert (d.start_char_offset, d.end_char_offset) == (7, 17)
    assert d.quote_text == "orld there"


def test_projection_stable_across_split() -> None:
    # The SAME visible span classifies identically whether the parent is split or
    # not: "world there" -> word_range [1, 3) either way.
    seg = _seg3()
    unsplit = derive_anchor(
        [_covered(seg)], CapturePayload(_ep(seg, 6), _ep(seg, 17), "world there")
    )
    split = derive_anchor(
        [_covered(seg, cuts=(1,))],
        CapturePayload(_ep(seg, 0, (1, 3)), _ep(seg, len("world there"), (1, 3)), "world there"),
    )
    assert unsplit.anchor_kind == split.anchor_kind == WORD_RANGE
    assert (unsplit.start_word_index, unsplit.end_word_index) == (
        split.start_word_index,
        split.end_word_index,
    )
    assert unsplit.source_text_hash == split.source_text_hash


def test_invalid_child_range_rejected() -> None:
    seg = _seg3()
    cov = [_covered(seg, cuts=(1,))]
    # (0, 2) is not a rendered child (children are [0,1) and [1,3)).
    with pytest.raises(AnnotationValidationError):
        derive_anchor(cov, CapturePayload(_ep(seg, 0, (0, 2)), _ep(seg, 3, (0, 2)), "x"))


def test_child_pair_must_be_both_or_neither() -> None:
    seg = _seg3()
    bad = CaptureEndpoint(segment_id=seg.id, offset=0, child_word_start=1, child_word_end=None)
    with pytest.raises(AnnotationValidationError):
        derive_anchor([_covered(seg, cuts=(1,))], CapturePayload(bad, _ep(seg, 5), "x"))


# --------------------------------------------------------------------------- #
# Outer whitespace delta
# --------------------------------------------------------------------------- #


def test_outer_whitespace_parent_trim_delta() -> None:
    # raw has a leading space; tokens reconcatenate to it (outer-trim tolerated).
    words = [
        {"word": " Hello", "start": 0.0, "end": 1.0},
        {"word": " world", "start": 1.0, "end": 2.0},
    ]
    seg = _segment(" Hello world", words=words, seconds=(0.0, 2.0))
    assert word_eligible(seg, None) is not None
    cov = [_covered(seg)]
    # effective text " Hello world": content_start(world)=7, content_end=12.
    d = derive_anchor(cov, CapturePayload(_ep(seg, 7), _ep(seg, 12), "world"))
    assert d.anchor_kind == WORD_RANGE
    assert (d.start_word_index, d.end_word_index) == (1, 2)
    assert d.quote_text == "world"


# --------------------------------------------------------------------------- #
# Cross-segment
# --------------------------------------------------------------------------- #


def _two_segments() -> tuple[TranscriptSegment, TranscriptSegment]:
    seg0 = _segment("Hello world", index=0, seconds=(0.0, 2.0))
    seg1 = _segment("there friend", index=1, seconds=(2.0, 4.0))
    return seg0, seg1


def test_cross_segment_word_range() -> None:
    seg0, seg1 = _two_segments()
    cov = [_covered(seg0), _covered(seg1)]
    # seg0 offset 6 = "world" start; seg1 offset 5 = "there" end.
    d = derive_anchor(cov, CapturePayload(_ep(seg0, 6), _ep(seg1, 5), "world\nthere"))
    assert d.anchor_kind == WORD_RANGE
    assert d.start_segment_id == seg0.id and d.end_segment_id == seg1.id
    assert (d.start_word_index, d.end_word_index) == (1, 1)
    assert d.quote_text == "world\nthere"


def test_cross_segment_whole_is_segment_range() -> None:
    seg0, seg1 = _two_segments()
    cov = [_covered(seg0), _covered(seg1)]
    d = derive_anchor(cov, CapturePayload(_ep(seg0, 0), _ep(seg1, 12), "Hello world\nthere friend"))
    assert d.anchor_kind == SEGMENT_RANGE
    assert d.quote_text == "Hello world\nthere friend"


def test_cross_segment_reverse_normalized() -> None:
    seg0, seg1 = _two_segments()
    cov = [_covered(seg0), _covered(seg1)]
    # Endpoints given end-first (seg1 then seg0) -> normalized by segment order.
    d = derive_anchor(cov, CapturePayload(_ep(seg1, 5), _ep(seg0, 6), "world\nthere"))
    assert d.start_segment_id == seg0.id and d.end_segment_id == seg1.id


# --------------------------------------------------------------------------- #
# Unicode: emoji + combining marks are code-point based
# --------------------------------------------------------------------------- #


def test_text_range_offsets_are_code_points_emoji() -> None:
    seg = _segment("a😀b c", with_words=False)  # words NULL -> not word range
    cov = [_covered(seg)]
    # "😀" is ONE code point at index 1..2, even though it is 2 UTF-16 units.
    # words NULL -> segment_range but quote is the verbatim sub-selection.
    d = derive_anchor(cov, CapturePayload(_ep(seg, 1), _ep(seg, 2), "😀"))
    assert d.quote_text == "😀"


def test_text_range_combining_mark_preserved() -> None:
    combining = "é"  # e + combining acute = 2 code points
    seg = _segment(f"{combining}xyz", with_words=False)
    cov = [_covered(seg)]
    d = derive_anchor(cov, CapturePayload(_ep(seg, 0), _ep(seg, 2), combining))
    assert d.quote_text == combining


# --------------------------------------------------------------------------- #
# client_quote consistency assertion
# --------------------------------------------------------------------------- #


def test_client_quote_mismatch_is_stale() -> None:
    seg = _seg3()
    cov = [_covered(seg)]
    with pytest.raises(AnnotationStaleError):
        derive_anchor(cov, CapturePayload(_ep(seg, 6), _ep(seg, 11), "wrong"))


# --------------------------------------------------------------------------- #
# Caps
# --------------------------------------------------------------------------- #


def test_span_segment_cap_enforced() -> None:
    segs = [
        _segment("word here", index=i, seconds=(float(i), float(i) + 1.0))
        for i in range(MAX_ANNOTATION_SPAN_SEGMENTS + 1)
    ]
    cov = [_covered(s) for s in segs]
    first, last = segs[0], segs[-1]
    with pytest.raises(AnnotationValidationError):
        derive_anchor(
            cov,
            CapturePayload(_ep(first, 0), _ep(last, 9), "x"),
        )


def test_offset_out_of_range_rejected() -> None:
    seg = _seg3()
    cov = [_covered(seg)]
    with pytest.raises(AnnotationValidationError):
        derive_anchor(cov, CapturePayload(_ep(seg, 0), _ep(seg, 999), "x"))


def test_unknown_segment_rejected() -> None:
    seg = _seg3()
    other = _seg3()
    with pytest.raises(AnnotationValidationError):
        derive_anchor([_covered(seg)], CapturePayload(_ep(other, 0), _ep(seg, 5), "x"))


# --------------------------------------------------------------------------- #
# Codex-review regressions: forged child range + whitespace-only tokens
# --------------------------------------------------------------------------- #


def test_child_coords_rejected_when_parent_unsplit() -> None:
    # An unsplit parent (no cuts) has no rendered split child; a forged
    # (0, word_count) child range must be refused, not accepted as word_range.
    seg = _seg3()
    cov = [_covered(seg)]  # cuts=()
    with pytest.raises(AnnotationValidationError):
        derive_anchor(
            cov,
            CapturePayload(_ep(seg, 0, (0, 3)), _ep(seg, 17, (0, 3)), "Hello world there"),
        )


def test_whitespace_only_token_falls_to_text_range() -> None:
    # A pathological whitespace-only word token makes the segment ineligible for
    # word_range (ambiguous boundary); the selection degrades to text_range with
    # code-point offsets, never anchoring to the empty token's timing.
    words = [
        {"word": "hello ", "start": 0.0, "end": 1.0},
        {"word": "   ", "start": 1.0, "end": 2.0},
        {"word": "world", "start": 2.0, "end": 3.0},
    ]
    seg = _segment("hello    world", words=words, seconds=(0.0, 3.0))
    assert word_eligible(seg, None) is None
    cov = [_covered(seg)]
    # "world" is at code points 9..14 of "hello    world".
    d = derive_anchor(cov, CapturePayload(_ep(seg, 9), _ep(seg, 14), "world"))
    assert d.anchor_kind == TEXT_RANGE
    assert (d.start_char_offset, d.end_char_offset) == (9, 14)
    assert d.start_seconds is None
    assert d.quote_text == "world"


def test_normalize_note_collapses_whitespace_only_to_none() -> None:
    # A whitespace-only note stores as NULL (no empty rendering; no fingerprint
    # drift between " " and absent), matching the client, which omits a blank draft.
    # Content notes — including intentional inner whitespace — are kept verbatim.
    from voxint.adjudication.annotations import _normalize_note

    assert _normalize_note(None) is None
    assert _normalize_note("") is None
    assert _normalize_note("   ") is None
    assert _normalize_note("\t\n ") is None
    assert _normalize_note("a note") == "a note"
    assert _normalize_note("  keeps inner  spaces  ") == "  keeps inner  spaces  "


def test_split_middle_child_no_leading_space_maps_correctly() -> None:
    # Tokens without leading spaces (reconcatenate to a space-free raw); a whole
    # middle child must map to the right parent word range.
    words = [
        {"word": "Hello", "start": 0.0, "end": 1.0},
        {"word": "world", "start": 1.0, "end": 2.0},
        {"word": "there", "start": 2.0, "end": 3.0},
    ]
    seg = _segment("Helloworldthere", words=words, seconds=(0.0, 3.0))
    cov = [_covered(seg, cuts=(1, 2))]  # children: Hello | world | there
    d = derive_anchor(cov, CapturePayload(_ep(seg, 0, (1, 2)), _ep(seg, 5, (1, 2)), "world"))
    assert d.anchor_kind == WORD_RANGE
    assert (d.start_word_index, d.end_word_index) == (1, 2)
    assert d.quote_text == "world"
    assert d.start_seconds == pytest.approx(1.0) and d.end_seconds == pytest.approx(2.0)


def test_three_segment_quote_assembly_and_timing() -> None:
    seg0 = _segment("one two", index=0, seconds=(0.0, 2.0))
    seg1 = _segment("middle words", index=1, seconds=(2.0, 4.0))
    seg2 = _segment("three four", index=2, seconds=(4.0, 6.0))
    cov = [_covered(seg0), _covered(seg1), _covered(seg2)]
    # seg0 "two" start (offset 4) .. seg2 "three" end (offset 5).
    d = derive_anchor(
        cov,
        CapturePayload(_ep(seg0, 4), _ep(seg2, 5), "two\nmiddle words\nthree"),
    )
    assert d.anchor_kind == WORD_RANGE
    assert d.start_segment_id == seg0.id and d.end_segment_id == seg2.id
    assert d.start_segment_index == 0 and d.end_segment_index == 2
    assert d.quote_text == "two\nmiddle words\nthree"
    # start = seg0 word "two" start; end = seg2 word "three" end.
    assert d.start_seconds == pytest.approx(1.0) and d.end_seconds == pytest.approx(5.0)


def test_split_child_mapping_with_emoji() -> None:
    # A split child whose text carries an astral emoji: child-local offsets are
    # code points, so the parent offset and quote land on whole code points.
    words = [
        {"word": "hi", "start": 0.0, "end": 1.0},
        {"word": " 👍ok", "start": 1.0, "end": 2.0},
    ]
    seg = _segment("hi 👍ok", words=words, seconds=(0.0, 2.0))
    cov = [_covered(seg, cuts=(1,))]  # children: "hi" | "👍ok"
    # Whole second child "👍ok" (3 code points).
    d = derive_anchor(cov, CapturePayload(_ep(seg, 0, (1, 2)), _ep(seg, 3, (1, 2)), "👍ok"))
    assert d.anchor_kind == WORD_RANGE
    assert (d.start_word_index, d.end_word_index) == (1, 2)
    assert d.quote_text == "👍ok"


# --------------------------------------------------------------------------- #
# Read resolver (resolve_annotation_spans) — the inverse mapping
# --------------------------------------------------------------------------- #


def _lines_from_covered(
    covered: list[CoveredSegment], speakers: dict[uuid.UUID, str] | None = None
) -> list[TranscriptLine]:
    """Render ``covered`` the way ``attributed_transcript`` does: a split parent
    (>= 2 children) expands into per-child lines carrying the PARENT id + the
    child's word window; every other segment is one whole line. This is the render
    the resolver maps stored anchors back onto."""
    names = speakers or {}
    lines: list[TranscriptLine] = []
    for cs in covered:
        seg = cs.segment
        spk = names.get(seg.id, "Speaker")
        children = derive_children(seg, list(cs.cuts)) if cs.cuts else None
        if children is not None and len(children) > 1:
            for ch in children:
                lines.append(
                    TranscriptLine(
                        start_seconds=ch.start_seconds,
                        end_seconds=ch.end_seconds,
                        speaker=spk,
                        text=ch.text,
                        segment_id=seg.id,
                        source_segment_id=seg.id,
                        word_start=ch.word_start,
                        word_end=ch.word_end,
                    )
                )
        else:
            lines.append(
                TranscriptLine(
                    start_seconds=seg.start_seconds,
                    end_seconds=seg.end_seconds,
                    speaker=spk,
                    text=cs.effective,
                    segment_id=seg.id,
                    source_segment_id=seg.id,
                )
            )
    return lines


def _stored(derived: DerivedAnchor, ann_id: uuid.UUID | None = None) -> StoredAnchor:
    """Project a freshly-derived anchor onto a stored anchor (what capture persists),
    so a resolve against the same render is the true capture inverse."""
    return StoredAnchor(
        annotation_id=ann_id or uuid.uuid4(),
        anchor_kind=derived.anchor_kind,
        start_segment_id=derived.start_segment_id,
        end_segment_id=derived.end_segment_id,
        start_segment_index=derived.start_segment_index,
        end_segment_index=derived.end_segment_index,
        start_word_index=derived.start_word_index,
        end_word_index=derived.end_word_index,
        start_char_offset=derived.start_char_offset,
        end_char_offset=derived.end_char_offset,
        source_text_hash=derived.source_text_hash,
        start_seconds=derived.start_seconds,
        end_seconds=derived.end_seconds,
    )


def _span_text(lines: list[TranscriptLine], resolved: object) -> list[str]:
    """The literal rendered slices a resolved annotation's spans cover."""
    return [lines[s.line_index].text[s.start : s.end] for s in resolved.spans]  # type: ignore[attr-defined]


def test_resolve_word_range_single_segment() -> None:
    seg = _seg3()
    cov = [_covered(seg)]
    d = derive_anchor(cov, CapturePayload(_ep(seg, 6), _ep(seg, 11), "world"))
    lines = _lines_from_covered(cov)
    [r] = resolve_annotation_spans(lines, cov, [_stored(d)])
    assert not r.stale
    assert r.timing_precision == TIMING_WORD
    assert r.start_seconds == pytest.approx(1.0) and r.end_seconds == pytest.approx(2.0)
    assert _span_text(lines, r) == ["world"]


def test_resolve_text_range_sub_word() -> None:
    seg = _seg3()
    cov = [_covered(seg)]
    d = derive_anchor(cov, CapturePayload(_ep(seg, 7), _ep(seg, 11), "orld"))
    lines = _lines_from_covered(cov)
    [r] = resolve_annotation_spans(lines, cov, [_stored(d)])
    assert not r.stale
    assert r.timing_precision == TIMING_SEGMENT
    # Coarse seconds fall back to the covered segment interval.
    assert r.start_seconds == pytest.approx(0.0) and r.end_seconds == pytest.approx(3.0)
    assert _span_text(lines, r) == ["orld"]


def test_resolve_segment_range_whole_lines() -> None:
    seg0, seg1 = _two_segments()
    cov = [_covered(seg0), _covered(seg1)]
    d = derive_anchor(cov, CapturePayload(_ep(seg0, 0), _ep(seg1, 12), "Hello world\nthere friend"))
    assert d.anchor_kind == SEGMENT_RANGE
    lines = _lines_from_covered(cov)
    [r] = resolve_annotation_spans(lines, cov, [_stored(d)])
    assert not r.stale
    assert _span_text(lines, r) == ["Hello world", "there friend"]


def test_resolve_cross_segment_word_range_head_and_tail() -> None:
    seg0, seg1 = _two_segments()
    cov = [_covered(seg0), _covered(seg1)]
    # "world" (seg0 tail) .. "there" (seg1 head).
    d = derive_anchor(cov, CapturePayload(_ep(seg0, 6), _ep(seg1, 5), "world\nthere"))
    assert d.anchor_kind == WORD_RANGE
    lines = _lines_from_covered(cov)
    [r] = resolve_annotation_spans(lines, cov, [_stored(d)])
    assert _span_text(lines, r) == ["world", "there"]


def test_resolve_word_range_on_split_child() -> None:
    seg = _seg3()
    cov = [_covered(seg, cuts=(1,))]  # children "Hello" | "world there"
    d = derive_anchor(
        cov,
        CapturePayload(_ep(seg, 0, (1, 3)), _ep(seg, len("world there"), (1, 3)), "world there"),
    )
    assert d.anchor_kind == WORD_RANGE
    lines = _lines_from_covered(cov)  # 2 child lines
    [r] = resolve_annotation_spans(lines, cov, [_stored(d)])
    # The whole second child line is highlighted; the first child is untouched.
    assert _span_text(lines, r) == ["world there"]
    assert all(s.line_index == 1 for s in r.spans)


def test_resolve_text_range_on_split_child_inverse_projection() -> None:
    seg = _seg3()
    cov = [_covered(seg, cuts=(1,))]  # children "Hello" | "world there"
    # "orld there" -> child-local 1..len -> parent 7..17 -> text_range.
    d = derive_anchor(
        cov, CapturePayload(_ep(seg, 1, (1, 3)), _ep(seg, len("world there"), (1, 3)), "orld there")
    )
    assert d.anchor_kind == TEXT_RANGE
    lines = _lines_from_covered(cov)
    [r] = resolve_annotation_spans(lines, cov, [_stored(d)])
    assert _span_text(lines, r) == ["orld there"]


def test_resolve_stale_when_text_changed_no_spans_but_locator() -> None:
    seg = _seg3()
    cov = [_covered(seg)]
    d = derive_anchor(cov, CapturePayload(_ep(seg, 6), _ep(seg, 11), "world"))
    stored = _stored(d)
    # A later correction changes the effective text -> hash mismatch -> stale.
    changed = [_covered(seg, corrected_text="Hello WORLD there")]
    lines = _lines_from_covered(changed)
    [r] = resolve_annotation_spans(lines, changed, [stored])
    assert r.stale
    assert r.spans == ()
    assert r.locator_line_index == 0


def test_resolve_word_range_lost_eligibility_is_stale() -> None:
    seg = _seg3()
    cov = [_covered(seg)]
    d = derive_anchor(cov, CapturePayload(_ep(seg, 6), _ep(seg, 11), "world"))
    stored = _stored(d)
    # A correction whose text still hashes differently AND removes word-eligibility;
    # even if a pathological correction re-hit the hash, eligibility loss alone
    # stales a word_range (never token-slice corrected text).
    corrected = [_covered(seg, corrected_text="Hello world there (edited)")]
    lines = _lines_from_covered(corrected)
    [r] = resolve_annotation_spans(lines, corrected, [stored])
    assert r.stale and r.spans == ()


def test_resolve_word_index_out_of_grid_degrades_to_stale_not_500() -> None:
    # Defensive: a stored word index that no longer fits the current token grid is a
    # broken invariant (unreachable under an identical source hash, since identical
    # text tokenizes identically). The read resolver must degrade it to stale with no
    # spans rather than IndexError-ing into a 500 on a read route (issue #86 review).
    seg = _seg3()
    cov = [_covered(seg)]
    d = derive_anchor(cov, CapturePayload(_ep(seg, 6), _ep(seg, 11), "world"))
    # Keep the source hash intact (so it is NOT stale by hash) but push end_word_index
    # past the 3-token grid — the exact case the old `assert`/direct-index would crash.
    stored = dataclasses.replace(_stored(d), end_word_index=99)
    lines = _lines_from_covered(cov)
    [r] = resolve_annotation_spans(lines, cov, [stored])
    assert r.stale
    assert r.spans == ()
    assert r.locator_line_index == 0


def test_resolve_speakers_are_live_not_captured() -> None:
    seg0, seg1 = _two_segments()
    cov = [_covered(seg0), _covered(seg1)]
    d = derive_anchor(cov, CapturePayload(_ep(seg0, 0), _ep(seg1, 12), "Hello world\nthere friend"))
    lines = _lines_from_covered(cov, speakers={seg0.id: "Ada", seg1.id: "Bo"})
    [r] = resolve_annotation_spans(lines, cov, [_stored(d)])
    assert r.speakers == ("Ada", "Bo")


def test_resolve_not_stale_across_split_projection_change() -> None:
    # The SAME anchor stays live and highlights the same visible text whether the
    # parent is currently split or not (projection-only change, hash unchanged).
    seg = _seg3()
    unsplit = [_covered(seg)]
    d = derive_anchor(unsplit, CapturePayload(_ep(seg, 6), _ep(seg, 17), "world there"))
    stored = _stored(d)
    split = [_covered(seg, cuts=(1,))]
    lines = _lines_from_covered(split)
    [r] = resolve_annotation_spans(lines, split, [stored])
    assert not r.stale
    assert "".join(_span_text(lines, r)) == "world there"


def test_stored_anchor_from_row_round_trips() -> None:
    row = TranscriptAnnotation()
    ann_id = uuid.uuid4()
    row.id = ann_id
    row.anchor_kind = WORD_RANGE
    row.start_segment_id = _A
    row.end_segment_id = _B
    row.start_segment_index = 0
    row.end_segment_index = 1
    row.start_word_index = 1
    row.end_word_index = 3
    row.start_char_offset = None
    row.end_char_offset = None
    row.source_text_hash = "0" * 64
    row.start_seconds = 1.0
    row.end_seconds = 2.0
    sa = stored_anchor_from_row(row)
    assert sa.annotation_id == ann_id
    assert sa.anchor_kind == WORD_RANGE
    assert (sa.start_word_index, sa.end_word_index) == (1, 3)
    assert sa.source_text_hash == "0" * 64


# --------------------------------------------------------------------------- #
# Pull-quote projection + canonical order (issue #86 Landing 2)
# --------------------------------------------------------------------------- #


def _resolved(
    *,
    annotation_id: uuid.UUID,
    spans: tuple[ResolvedSpan, ...],
    stale: bool = False,
    locator: int | None = None,
) -> ResolvedAnnotation:
    return ResolvedAnnotation(
        annotation_id=annotation_id,
        anchor_kind=WORD_RANGE,
        stale=stale,
        timing_precision=TIMING_WORD,
        start_seconds=1.0,
        end_seconds=2.0,
        speakers=("S0",),
        spans=spans,
        locator_line_index=locator,
    )


def test_clip_lines_slices_to_span_and_preserves_speaker_seconds() -> None:
    lines = [
        TranscriptLine(
            start_seconds=0.0, end_seconds=3.0, speaker="Alice", text="Hello world there"
        ),
        TranscriptLine(start_seconds=3.0, end_seconds=6.0, speaker="Bob", text="how are you"),
    ]
    resolved = _resolved(
        annotation_id=_A,
        spans=(
            ResolvedSpan(line_index=0, start=6, end=11),
            ResolvedSpan(line_index=1, start=0, end=3),
        ),
    )
    clipped = clip_lines_for_export(resolved, lines)
    assert [(c.speaker, c.text, c.start_seconds, c.end_seconds) for c in clipped] == [
        ("Alice", "world", 0.0, 3.0),
        ("Bob", "how", 3.0, 6.0),
    ]


def test_clip_lines_stale_annotation_yields_nothing() -> None:
    lines = [TranscriptLine(start_seconds=0.0, end_seconds=1.0, speaker="S", text="x")]
    resolved = _resolved(annotation_id=_A, spans=(), stale=True, locator=0)
    assert clip_lines_for_export(resolved, lines) == []


def test_resolved_order_key_by_line_then_offset_then_id() -> None:
    later_line = _resolved(annotation_id=_A, spans=(ResolvedSpan(line_index=2, start=0, end=1),))
    early_line = _resolved(annotation_id=_B, spans=(ResolvedSpan(line_index=0, start=5, end=6),))
    same_line_earlier = _resolved(
        annotation_id=_A, spans=(ResolvedSpan(line_index=0, start=1, end=2),)
    )
    ordered = sorted(
        [later_line, early_line, same_line_earlier], key=resolved_order_key
    )
    # line 0 offset 1 (id _A) < line 0 offset 5 (id _B) < line 2.
    assert [r.annotation_id for r in ordered] == [_A, _B, _A]


def test_resolved_order_key_stale_uses_locator_then_sentinel() -> None:
    stale_with_locator = _resolved(annotation_id=_A, spans=(), stale=True, locator=1)
    unresolvable = _resolved(annotation_id=_B, spans=(), stale=True, locator=None)
    live_line0 = _resolved(annotation_id=_A, spans=(ResolvedSpan(line_index=0, start=0, end=1),))
    ordered = sorted(
        [unresolvable, stale_with_locator, live_line0], key=resolved_order_key
    )
    # live line 0 < stale locator line 1 < unresolvable (sentinel last).
    assert [r.annotation_id for r in ordered] == [_A, _A, _B]
