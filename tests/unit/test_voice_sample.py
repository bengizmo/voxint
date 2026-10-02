"""Exact clean intervals and deterministic voice sample windows."""

import uuid

import pytest

from voxint.adjudication.attribution import AttributedInterval, AttributionScope
from voxint.adjudication.resolver import Resolution
from voxint.db.models import DiarizationTurn
from voxint.media.clips import ClipBounds, cap_bounds, resolve_sample_bounds
from voxint.speakers.voice_sample import VoiceSample, best_span, clean_spans, merge_spans


def row(start: float = 0, end: float = 20) -> AttributedInterval:
    return AttributedInterval(
        start,
        end,
        uuid.uuid4(),
        "S0",
        AttributionScope.LABEL,
        Resolution.HUMAN_ASSIGN,
        uuid.uuid4(),
        "Dana",
        None,
        None,
    )


def turn(start: float, end: float, label: str = "S0") -> DiarizationTurn:
    return DiarizationTurn(start_seconds=start, end_seconds=end, label=label, overlap=True)


@pytest.mark.parametrize(
    ("turns", "expected"),
    [
        ([turn(0, 4), turn(3, 8), turn(8, 10)], [(0, 10)]),
        ([turn(0, 10), turn(0, 1.5, "S1"), turn(8, 12, "S2")], [(1.5, 8)]),
        ([turn(0, 10), turn(3, 6, "S1")], [(0, 3), (6, 10)]),
        ([turn(0, 10), turn(0, 10, "S1")], []),
        ([turn(0, 4), turn(8, 10), turn(5, 7, "S1")], [(0, 4), (8, 10)]),
    ],
)
def test_clean_spans(turns: list[DiarizationTurn], expected: list[tuple[float, float]]) -> None:
    assert clean_spans(row(), turns) == expected


def test_floor_union_and_choice() -> None:
    assert best_span([(0, 1.99)]) is None
    assert best_span([(0, 2)]) == (0, 2)
    assert best_span([(0, 1), (1, 2)]) == (0, 2)
    assert best_span([(9, 12), (1, 4), (20, 22)]) == (1, 4)
    assert best_span([(0, 2), (9, 13)]) == (9, 13)
    assert merge_spans([(4, 3), (1, 1)]) == []


def test_fractional_window_exact_frame_cap() -> None:
    sample = VoiceSample(uuid.uuid4(), uuid.uuid4(), (0.00003, 20.00003), "available")
    assert sample.window == (0.00003, 10.00003)
    bounds = resolve_sample_bounds(*sample.window, 400000, max_clip_frames=160001)
    assert bounds.frame_count == 160001
    assert cap_bounds(bounds, 160000) == ClipBounds(0, 160000)
    assert cap_bounds(ClipBounds(42, 123), 160000) == ClipBounds(42, 123)
    assert VoiceSample(sample.speaker_id, sample.run_id, (2, 5), "gone").window == (2, 5)
