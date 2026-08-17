"""Word bucketing in the transcribe stage (issue #59).

The whisper service emits a flat, run-level word list; the stage assigns each
word to exactly one segment by maximum temporal overlap, deterministically, so
the same transcription always persists the same per-segment words.
"""

from voxint.clients.base import TranscriptionSegment, TranscriptionWord
from voxint.pipeline.stages.transcribe import _word_payload, bucket_words


def _w(start: float, end: float, word: str = "x") -> TranscriptionWord:
    return TranscriptionWord(start_seconds=start, end_seconds=end, word=word)


def _seg(start: float, end: float) -> TranscriptionSegment:
    return TranscriptionSegment(start_seconds=start, end_seconds=end, text="seg")


def test_words_bucket_into_containing_segment() -> None:
    segments = (_seg(0.0, 4.0), _seg(4.0, 8.0))
    words = (_w(0.0, 1.0, "a"), _w(1.0, 2.0, "b"), _w(5.0, 6.0, "c"))
    buckets = bucket_words(segments, words)
    assert [[w.word for w in b] for b in buckets] == [["a", "b"], ["c"]]


def test_word_straddling_boundary_goes_to_max_overlap() -> None:
    # 3.5-4.5 overlaps seg0 by 0.5 and seg1 by 0.5 -> equal overlap; tie-break to
    # the nearer (both gap 0) then earlier segment.
    segments = (_seg(0.0, 4.0), _seg(4.0, 8.0))
    buckets = bucket_words(segments, (_w(3.5, 4.5, "edge"),))
    assert [len(b) for b in buckets] == [1, 0]
    # A word mostly in seg1 lands in seg1.
    buckets2 = bucket_words(segments, (_w(3.9, 5.0, "late"),))
    assert [len(b) for b in buckets2] == [0, 1]


def test_gap_word_goes_to_nearest_segment() -> None:
    # A word in the silent gap 4-6 (no overlap with either segment) attaches to
    # the nearest by interval distance.
    segments = (_seg(0.0, 4.0), _seg(6.0, 10.0))
    near_first = bucket_words(segments, (_w(4.2, 4.6, "g"),))
    assert [len(b) for b in near_first] == [1, 0]
    near_second = bucket_words(segments, (_w(5.6, 5.9, "g"),))
    assert [len(b) for b in near_second] == [0, 1]


def test_bucketing_is_deterministic() -> None:
    segments = (_seg(0.0, 4.0), _seg(4.0, 8.0), _seg(8.0, 12.0))
    words = tuple(_w(i * 0.5, i * 0.5 + 0.5, f"w{i}") for i in range(24))
    first = bucket_words(segments, words)
    second = bucket_words(segments, words)
    assert [[w.word for w in b] for b in first] == [
        [w.word for w in b] for b in second
    ]
    # Every word is placed exactly once.
    assert sum(len(b) for b in first) == len(words)


def test_word_before_first_and_after_last_attach_to_ends() -> None:
    # A word wholly outside every segment still lands somewhere (nearest end).
    segments = (_seg(2.0, 4.0), _seg(4.0, 6.0))
    before = bucket_words(segments, (_w(0.0, 0.5, "pre"),))
    assert [len(b) for b in before] == [1, 0]
    after = bucket_words(segments, (_w(9.0, 9.5, "post"),))
    assert [len(b) for b in after] == [0, 1]


def test_zero_length_word_lands_in_containing_segment() -> None:
    # start == end (allowed by zero_length_ok): zero overlap everywhere, so the
    # gap tie-break (distance 0 inside seg1) places it there.
    segments = (_seg(0.0, 4.0), _seg(4.0, 8.0))
    buckets = bucket_words(segments, (_w(5.0, 5.0, "z"),))
    assert [len(b) for b in buckets] == [0, 1]


def test_zero_length_segment_can_receive_a_word() -> None:
    # A degenerate zero-length segment (allowed by the interval CHECK) still wins
    # a word that sits exactly on it via the gap tie-break.
    segments = (_seg(0.0, 2.0), _seg(2.0, 2.0), _seg(2.0, 4.0))
    buckets = bucket_words(segments, (_w(2.0, 2.0, "z"),))
    assert sum(len(b) for b in buckets) == 1


def test_gap_midpoint_prefers_earlier_segment() -> None:
    # Exactly halfway across a gap: equal distance both sides -> earlier index.
    segments = (_seg(0.0, 4.0), _seg(6.0, 10.0))
    buckets = bucket_words(segments, (_w(4.5, 5.5, "mid"),))  # gap 1.0 each side
    assert [len(b) for b in buckets] == [1, 0]


def test_no_segments_returns_empty() -> None:
    assert bucket_words((), (_w(0.0, 1.0),)) == []


def test_word_payload_is_always_a_list() -> None:
    # The NULL-vs-array decision is the caller's (per run); the payload builder
    # itself always returns a list, empty for an empty bucket.
    assert _word_payload([]) == []
    payload = _word_payload([TranscriptionWord(0.0, 0.5, "hi", confidence=0.9)])
    assert payload == [{"start": 0.0, "end": 0.5, "word": "hi", "confidence": 0.9}]
    # confidence None round-trips as None, never fabricated.
    assert _word_payload([TranscriptionWord(0.0, 0.5, "hi")])[0]["confidence"] is None
