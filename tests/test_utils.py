"""Tests for spellcaster.utils (smoothing, statistics)."""

import math

import numpy as np
import pytest

from spellcaster.utils.smoothing import per_1k, smooth
from spellcaster.utils.statistics import interval_summary, repetition_intervals


# ── smooth ───────────────────────────────────────────────────────────────────

class TestSmooth:
    def test_basic_averaging(self):
        result = smooth([1, 2, 3, 4, 5], window=3)
        expected = np.array([2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_window_1_is_identity(self):
        data = [10, 20, 30]
        result = smooth(data, window=1)
        np.testing.assert_array_almost_equal(result, data)

    def test_empty_input(self):
        result = smooth([], window=3)
        assert len(result) == 0

    def test_window_larger_than_data(self):
        result = smooth([1, 2], window=5)
        np.testing.assert_array_almost_equal(result, [1, 2])

    def test_output_length(self):
        data = list(range(20))
        result = smooth(data, window=5)
        assert len(result) == 20 - 5 + 1


# ── per_1k ───────────────────────────────────────────────────────────────────

class TestPer1k:
    def test_basic(self):
        assert per_1k(10, 100) == pytest.approx(100.0)

    def test_zero_total(self):
        assert math.isnan(per_1k(5, 0))

    def test_zero_count(self):
        assert per_1k(0, 100) == pytest.approx(0.0)


# ── repetition_intervals ────────────────────────────────────────────────────

class TestRepetitionIntervals:
    def test_basic(self):
        tokens = ["a", "b", "c", "a", "d", "a"]
        result = repetition_intervals(tokens, ["a"])
        assert result["a"] == [3, 2]  # index 0→3, 3→5

    def test_keyword_not_present(self):
        result = repetition_intervals(["a", "b", "c"], ["x"])
        assert result["x"] == []

    def test_single_occurrence(self):
        result = repetition_intervals(["a", "b", "c"], ["b"])
        assert result["b"] == []

    def test_multiple_keywords(self):
        tokens = ["x", "y", "x", "y", "x"]
        result = repetition_intervals(tokens, ["x", "y"])
        assert result["x"] == [2, 2]
        assert result["y"] == [2]

    def test_empty_tokens(self):
        result = repetition_intervals([], ["a"])
        assert result["a"] == []


# ── interval_summary ────────────────────────────────────────────────────────

class TestIntervalSummary:
    def test_basic(self):
        summary = interval_summary([10, 20, 30])
        assert summary["mean"] == pytest.approx(20.0)
        assert summary["median"] == 20.0
        assert summary["count"] == 3

    def test_empty(self):
        summary = interval_summary([])
        assert summary["mean"] is None
        assert summary["count"] == 0

    def test_single_value(self):
        summary = interval_summary([42])
        assert summary["mean"] == pytest.approx(42.0)
        assert summary["median"] == 42.0
        assert summary["count"] == 1
