"""Unit tests for the synthdetect service windowing and padding logic.

These test the pure functions extracted from scoring.py without any torch
or model dependency. The windowing policy is part of the inference space
identity, so these are load-bearing: a change here means a new
inference_space string.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[2] / "services" / "synthdetect"

# We need to import _tile_windows and _repeat_pad from the scoring module.
# Since scoring.py has torch imports inside methods (not at module level),
# we can import the pure functions without torch being available in test env.
# However, the module-level imports include numpy, which is fine.


@pytest.fixture(scope="module")
def scoring():
    app_path = str(SERVICE_ROOT)
    if app_path not in sys.path:
        sys.path.insert(0, app_path)
    try:
        for mod_name in list(sys.modules):
            if mod_name.startswith("app.") and "schemas" not in mod_name:
                del sys.modules[mod_name]
        import importlib
        return importlib.import_module("app.scoring")
    finally:
        if app_path in sys.path:
            sys.path.remove(app_path)


MODEL_WINDOW = 64_600
MIN_SCORABLE = 8_000


class TestTileWindows:
    def test_exact_one_window(self, scoring):
        windows = scoring._tile_windows(100_000, 0, MODEL_WINDOW)
        assert windows == [(0, MODEL_WINDOW)]

    def test_two_full_windows(self, scoring):
        windows = scoring._tile_windows(200_000, 0, MODEL_WINDOW * 2)
        assert windows == [(0, MODEL_WINDOW), (MODEL_WINDOW, MODEL_WINDOW * 2)]

    def test_one_full_plus_scorable_tail(self, scoring):
        end = MODEL_WINDOW + MIN_SCORABLE
        windows = scoring._tile_windows(200_000, 0, end)
        assert len(windows) == 2
        assert windows[0] == (0, MODEL_WINDOW)
        assert windows[1] == (MODEL_WINDOW, end)

    def test_one_full_plus_short_tail_dropped(self, scoring):
        end = MODEL_WINDOW + MIN_SCORABLE - 1
        windows = scoring._tile_windows(200_000, 0, end)
        assert len(windows) == 1
        assert windows[0] == (0, MODEL_WINDOW)

    def test_sub_window_above_min_scorable(self, scoring):
        length = MIN_SCORABLE + 100
        windows = scoring._tile_windows(100_000, 0, length)
        assert len(windows) == 1
        assert windows[0] == (0, length)

    def test_sub_window_below_min_scorable_empty(self, scoring):
        length = MIN_SCORABLE - 1
        windows = scoring._tile_windows(100_000, 0, length)
        assert windows == []

    def test_zero_length_empty(self, scoring):
        windows = scoring._tile_windows(100_000, 500, 500)
        assert windows == []

    def test_offset_start(self, scoring):
        start = 10_000
        end = start + MODEL_WINDOW
        windows = scoring._tile_windows(200_000, start, end)
        assert windows == [(start, end)]

    def test_clipped_by_n_samples(self, scoring):
        n = MODEL_WINDOW + 5_000
        windows = scoring._tile_windows(n, 0, n)
        assert len(windows) == 1
        assert windows[0] == (0, MODEL_WINDOW)

    def test_long_interval_many_windows(self, scoring):
        length = MODEL_WINDOW * 5 + 100
        windows = scoring._tile_windows(length + 10_000, 0, length)
        assert len(windows) == 5


class TestRepeatPad:
    def test_exact_length_unchanged(self, scoring):
        arr = np.arange(MODEL_WINDOW, dtype=np.float32)
        result = scoring._repeat_pad(arr, MODEL_WINDOW)
        np.testing.assert_array_equal(result, arr)

    def test_longer_than_target_truncated(self, scoring):
        arr = np.arange(MODEL_WINDOW + 100, dtype=np.float32)
        result = scoring._repeat_pad(arr, MODEL_WINDOW)
        assert len(result) == MODEL_WINDOW
        np.testing.assert_array_equal(result, arr[:MODEL_WINDOW])

    def test_half_length_repeated(self, scoring):
        half = MODEL_WINDOW // 2
        arr = np.ones(half, dtype=np.float32) * 0.5
        result = scoring._repeat_pad(arr, MODEL_WINDOW)
        assert len(result) == MODEL_WINDOW
        assert result[0] == 0.5
        assert result[half] == 0.5

    def test_single_sample_repeated(self, scoring):
        arr = np.array([0.42], dtype=np.float32)
        result = scoring._repeat_pad(arr, 100)
        assert len(result) == 100
        np.testing.assert_allclose(result, 0.42)

    def test_short_clip_pattern_preserved(self, scoring):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = scoring._repeat_pad(arr, 10)
        expected = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0])
        np.testing.assert_array_equal(result, expected)
