"""Word-boundary split derivation + guard (issue #59, slice 2).

Pure tests for :mod:`voxint.adjudication.splits`: which segments are splittable,
how children partition the immutable word tokens, and the alignment guard that
keeps a split from ever inventing offsets. The writer (``record_split``) and the
routes are exercised in the integration suite.
"""

import uuid
from typing import Any

from voxint.adjudication.splits import (
    derive_children,
    splittable_words,
    word_count,
)
from voxint.db.models import TranscriptSegment


def _seg(
    *,
    raw_text: str,
    words: list[dict[str, Any]] | None,
    enhanced_text: str | None = None,
    start: float = 0.0,
    end: float = 10.0,
) -> TranscriptSegment:
    return TranscriptSegment(
        id=uuid.uuid4(),
        pipeline_run_id=uuid.uuid4(),
        segment_index=0,
        start_seconds=start,
        end_seconds=end,
        raw_text=raw_text,
        enhanced_text=enhanced_text,
        words=words,
    )


def _word(word: str, start: float, end: float) -> dict[str, Any]:
    return {"start": start, "end": end, "word": word, "confidence": 0.9}


# faster-whisper word strings carry a leading space; segment.text (raw_text) is
# their exact concatenation, so joined == raw_text is the normal path.
_HELLO = [_word(" Hello", 0.0, 0.5), _word(" world", 0.6, 1.0)]


def test_splittable_segment_exposes_word_count() -> None:
    seg = _seg(raw_text=" Hello world", words=_HELLO)
    words = splittable_words(seg)
    assert words is not None
    assert [w.text for w in words] == [" Hello", " world"]
    assert word_count(seg) == 2


def test_derive_children_partitions_and_derives_timing() -> None:
    seg = _seg(raw_text=" Hello world", words=_HELLO)
    children = derive_children(seg, [1])
    assert children is not None
    assert len(children) == 2
    first, second = children
    assert (first.word_start, first.word_end) == (0, 1)
    assert first.text == "Hello"  # outer whitespace trimmed once for display
    assert (first.start_seconds, first.end_seconds) == (0.0, 0.5)
    assert (second.word_start, second.word_end) == (1, 2)
    assert second.text == "world"
    assert (second.start_seconds, second.end_seconds) == (0.6, 1.0)


def test_children_reconcatenate_to_raw_text() -> None:
    # The partition invariant: joining the child word slices verbatim (no per-child
    # strip) must reproduce raw_text exactly — no character invented or dropped.
    seg = _seg(raw_text=" Hello world", words=_HELLO)
    words = splittable_words(seg)
    assert words is not None
    assert "".join(w.text for w in words) == seg.raw_text


def test_outer_whitespace_delta_is_tolerated() -> None:
    # raw_text stripped at store, tokens keep the leading space: only the OUTER
    # edge differs, so the segment stays splittable.
    seg = _seg(raw_text="Hello world", words=_HELLO)
    assert splittable_words(seg) is not None


def test_null_words_is_unsplittable() -> None:
    assert splittable_words(_seg(raw_text="hi", words=None)) is None
    assert word_count(_seg(raw_text="hi", words=None)) is None


def test_empty_and_single_word_are_unsplittable() -> None:
    assert splittable_words(_seg(raw_text="", words=[])) is None
    single = [_word("Hello", 0.0, 0.5)]
    assert splittable_words(_seg(raw_text="Hello", words=single)) is None


def test_malformed_tokens_are_unsplittable() -> None:
    # Non-string word, empty word, missing timing, non-finite timing, inverted
    # interval — each fails closed rather than guessing a boundary.
    assert splittable_words(_seg(raw_text="x y", words=[{"word": 1}, _word(" y", 0.1, 0.2)])) is None
    assert splittable_words(_seg(raw_text=" y", words=[_word("", 0.0, 0.1), _word(" y", 0.1, 0.2)])) is None
    assert splittable_words(_seg(raw_text="a b", words=[{"word": "a"}, _word(" b", 0.1, 0.2)])) is None
    assert (
        splittable_words(
            _seg(raw_text="a b", words=[_word("a", float("nan"), 0.1), _word(" b", 0.1, 0.2)])
        )
        is None
    )
    assert (
        splittable_words(
            _seg(raw_text="a b", words=[_word("a", 0.5, 0.1), _word(" b", 0.6, 0.7)])
        )
        is None
    )


def test_reconcatenation_mismatch_is_unsplittable() -> None:
    # Bucketing dropped/added a word: the tokens no longer reproduce raw_text.
    seg = _seg(raw_text=" Hello there world", words=_HELLO)
    assert splittable_words(seg) is None


def test_materially_enhanced_segment_is_unsplittable() -> None:
    # enhanced_text materially differs from raw_text → children could not render
    # faithfully under ?text=enhanced, so the segment is unsplittable.
    seg = _seg(raw_text=" Hello world", words=_HELLO, enhanced_text="Hello, world!")
    assert splittable_words(seg) is None
    # A whitespace-only enhanced delta is NOT material.
    ok = _seg(raw_text=" Hello world", words=_HELLO, enhanced_text="Hello world")
    assert splittable_words(ok) is not None


def test_word_timings_outside_parent_interval_are_unsplittable() -> None:
    seg = _seg(raw_text=" Hello world", words=_HELLO, start=0.0, end=0.4)
    assert splittable_words(seg) is None  # " world" ends at 1.0 > parent end 0.4


def test_derive_children_ignores_out_of_range_and_duplicate_cuts() -> None:
    seg = _seg(raw_text=" Hello world", words=_HELLO)
    # 0 and word_count are sentinels, never real cuts; duplicates collapse. With no
    # interior cut left, a single child spans the whole segment (renders exactly
    # its own text), never an empty child.
    children = derive_children(seg, [0, 2, 2, 5])
    assert children is not None
    assert len(children) == 1
    assert children[0].text == "Hello world"


def test_derive_children_on_unsplittable_returns_none() -> None:
    assert derive_children(_seg(raw_text="hi", words=None), [1]) is None
