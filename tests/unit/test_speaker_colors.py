"""Unit tests for the per-speaker palette assignment (issue #50).

`speaker_palette` is the one seam that guarantees the transcript page, the
workbench card, and the JS-off fallback agree on a label's color. Its contract:
deterministic, order-independent, and bounded to the curated palette.
"""

import random

from voxint.api.speaker_colors import PALETTE_SIZE, speaker_palette


def test_empty_input_is_empty_map() -> None:
    assert speaker_palette([]) == {}


def test_deterministic_same_input_same_map() -> None:
    labels = ["SPEAKER_02", "SPEAKER_00", "SPEAKER_01"]
    assert speaker_palette(labels) == speaker_palette(labels)


def test_order_independent() -> None:
    labels = [f"SPEAKER_{i:02d}" for i in range(5)]
    shuffled = labels[:]
    random.Random(1234).shuffle(shuffled)
    assert speaker_palette(shuffled) == speaker_palette(labels)


def test_positional_over_sorted_distinct_labels() -> None:
    # Assignment is the index of the label in sorted order, mod PALETTE_SIZE.
    labels = ["b", "a", "c", "a"]  # duplicate collapses; order ignored
    assert speaker_palette(labels) == {"a": 0, "b": 1, "c": 2}


def test_distinct_labels_get_distinct_indices_below_palette_size() -> None:
    labels = [f"SPEAKER_{i:02d}" for i in range(PALETTE_SIZE)]
    palette = speaker_palette(labels)
    assert len(set(palette.values())) == PALETTE_SIZE
    assert all(0 <= idx < PALETTE_SIZE for idx in palette.values())


def test_wraparound_beyond_palette_size_reuses_indices() -> None:
    n = PALETTE_SIZE * 2 + 3
    labels = [f"SPEAKER_{i:02d}" for i in range(n)]
    palette = speaker_palette(labels)
    assert len(palette) == n
    # Every index stays inside the curated range...
    assert all(0 <= idx < PALETTE_SIZE for idx in palette.values())
    # ...and the palette wraps: the (PALETTE_SIZE)-th sorted label reuses index 0.
    ordered = sorted(labels)
    assert palette[ordered[0]] == palette[ordered[PALETTE_SIZE]] == 0
    assert palette[ordered[PALETTE_SIZE + 1]] == 1
