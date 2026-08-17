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


def test_no_segments_returns_empty() -> None:
    assert bucket_words((), (_w(0.0, 1.0),)) == []


def test_word_payload_shape_and_none() -> None:
    assert _word_payload([]) is None
    payload = _word_payload([TranscriptionWord(0.0, 0.5, "hi", confidence=0.9)])
    assert payload == [{"start": 0.0, "end": 0.5, "word": "hi", "confidence": 0.9}]
