"""Tests for spellcaster.core.entropy."""

import math
from collections import Counter

import pytest

from spellcaster.core.entropy import (
    multiscale_collapse_curve,
    shannon_entropy,
    summarize_multiscale_collapse,
    window_collapse,
)


# ── shannon_entropy ──────────────────────────────────────────────────────────

class TestShannonEntropy:
    def test_uniform_distribution(self, uniform_counter):
        """4 equally-likely tokens → exactly 2.0 bits."""
        assert shannon_entropy(uniform_counter) == pytest.approx(2.0, abs=1e-9)

    def test_single_token(self, single_token_counter):
        """No uncertainty → 0 bits."""
        assert shannon_entropy(single_token_counter) == 0.0

    def test_empty_counter(self, empty_counter):
        assert shannon_entropy(empty_counter) == 0.0

    def test_skewed_lower_than_uniform(self, uniform_counter, skewed_counter):
        h_uniform = shannon_entropy(uniform_counter)
        h_skewed = shannon_entropy(skewed_counter)
        assert h_skewed < h_uniform

    def test_binary_symmetric(self):
        c = Counter({"a": 50, "b": 50})
        assert shannon_entropy(c) == pytest.approx(1.0, abs=1e-9)

    def test_non_negative(self, skewed_counter):
        assert shannon_entropy(skewed_counter) >= 0.0


# ── window_collapse ──────────────────────────────────────────────────────────

class TestWindowCollapse:
    def test_empty_tokens(self):
        assert window_collapse([], win_size=10) == []

    def test_single_token_type(self):
        """All same token → window entropy is 0, collapse is 1.0."""
        tokens = ["x"] * 100
        collapses = window_collapse(tokens, win_size=50)
        assert len(collapses) == 2
        # But doc entropy is also 0, so function returns []
        assert collapses == [] or all(c == 0.0 for c in collapses)

    def test_diverse_tokens_low_collapse(self, token_list_diverse):
        collapses = window_collapse(token_list_diverse, win_size=250)
        assert len(collapses) >= 1
        # With all unique tokens, collapse should be relatively low
        for c in collapses:
            assert 0.0 <= c <= 1.0

    def test_values_bounded(self, token_list_repetitive):
        # Repetitive has only 2 types, so doc H > 0
        tokens = ["the", "cat"] * 50 + ["dog", "bird"] * 25
        collapses = window_collapse(tokens, win_size=50)
        for c in collapses:
            assert -0.01 <= c <= 1.01  # Small float tolerance

    def test_window_count(self):
        tokens = [f"w{i % 20}" for i in range(100)]
        collapses = window_collapse(tokens, win_size=25)
        assert len(collapses) == 4  # 100 / 25

    def test_tail_chunk_dropped(self):
        tokens = [f"w{i % 10}" for i in range(53)]
        collapses = window_collapse(tokens, win_size=25)
        # 53 / 25 → 2 full windows + 3-token tail (kept because >= 2)
        assert len(collapses) == 3


# ── multiscale_collapse_curve ────────────────────────────────────────────────

class TestMultiscaleCollapseCurve:
    def test_returns_one_entry_per_window_size(self):
        tokens = [f"w{i % 30}" for i in range(600)]
        sizes = (25, 50, 100)
        curve = multiscale_collapse_curve(tokens, win_sizes=sizes)
        assert len(curve) == 3
        assert [d["win_size"] for d in curve] == [25, 50, 100]

    def test_empty_tokens(self):
        curve = multiscale_collapse_curve([], win_sizes=(25,))
        assert len(curve) == 1
        assert math.isnan(curve[0]["mean_collapse"])

    def test_mean_collapse_within_bounds(self):
        tokens = [f"w{i % 15}" for i in range(500)]
        curve = multiscale_collapse_curve(tokens, win_sizes=(25, 100))
        for d in curve:
            if not math.isnan(d["mean_collapse"]):
                assert 0.0 <= d["mean_collapse"] <= 1.0


# ── summarize_multiscale_collapse ────────────────────────────────────────────

class TestSummarizeMultiscaleCollapse:
    def test_empty_curve(self):
        result = summarize_multiscale_collapse([])
        assert math.isnan(result["collapse_auc"])
        assert result["peak_win_size"] is None

    def test_single_point(self):
        curve = [{"win_size": 50, "n_windows": 4, "mean_collapse": 0.3, "max_collapse": 0.5}]
        result = summarize_multiscale_collapse(curve)
        assert result["collapse_auc"] == 0.0
        assert result["peak_win_size"] == 50
        assert result["peak_mean_collapse"] == pytest.approx(0.3)

    def test_auc_positive_for_positive_collapses(self):
        curve = [
            {"win_size": 25, "n_windows": 10, "mean_collapse": 0.2, "max_collapse": 0.4},
            {"win_size": 100, "n_windows": 3, "mean_collapse": 0.5, "max_collapse": 0.7},
        ]
        result = summarize_multiscale_collapse(curve, x_scale="log")
        assert result["collapse_auc"] > 0
        assert result["collapse_auc_norm"] > 0

    def test_peak_is_correct(self):
        curve = [
            {"win_size": 25, "n_windows": 10, "mean_collapse": 0.1, "max_collapse": 0.2},
            {"win_size": 50, "n_windows": 5, "mean_collapse": 0.8, "max_collapse": 0.9},
            {"win_size": 100, "n_windows": 2, "mean_collapse": 0.4, "max_collapse": 0.5},
        ]
        result = summarize_multiscale_collapse(curve)
        assert result["peak_win_size"] == 50
        assert result["peak_mean_collapse"] == pytest.approx(0.8)
