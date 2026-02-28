"""Tests for spellcaster.core.information."""

import math
from collections import Counter

import numpy as np
import pytest

from spellcaster.core.information import (
    channel_capacity,
    js_distance_from_counters,
    js_divergence_from_counters,
    js_divergence_matrix,
    mutual_information,
)


# ── mutual_information ───────────────────────────────────────────────────────

class TestMutualInformation:
    def test_independent_variables(self):
        """When X and Y are independent, MI should be ~0."""
        # Joint = product of marginals
        joint = Counter({("a", "x"): 25, ("a", "y"): 25, ("b", "x"): 25, ("b", "y"): 25})
        mx = Counter({"a": 50, "b": 50})
        my = Counter({"x": 50, "y": 50})
        mi = mutual_information(joint, mx, my, n=100)
        assert mi == pytest.approx(0.0, abs=1e-9)

    def test_perfectly_dependent(self):
        """When X determines Y, MI = H(X) = H(Y)."""
        joint = Counter({("a", "x"): 50, ("b", "y"): 50})
        mx = Counter({"a": 50, "b": 50})
        my = Counter({"x": 50, "y": 50})
        mi = mutual_information(joint, mx, my, n=100)
        assert mi == pytest.approx(1.0, abs=1e-6)  # H = 1 bit for 2 equally-likely

    def test_empty_joint(self):
        joint = Counter()
        mx = Counter()
        my = Counter()
        assert mutual_information(joint, mx, my, n=0) == 0.0

    def test_non_negative(self):
        joint = Counter({("a", "x"): 30, ("a", "y"): 10, ("b", "x"): 20, ("b", "y"): 40})
        mx = Counter({"a": 40, "b": 60})
        my = Counter({"x": 50, "y": 50})
        mi = mutual_information(joint, mx, my, n=100)
        assert mi >= 0.0

    def test_auto_n(self):
        """When n is None, it should be inferred from joint sum."""
        joint = Counter({("a", "x"): 50, ("b", "y"): 50})
        mx = Counter({"a": 50, "b": 50})
        my = Counter({"x": 50, "y": 50})
        mi = mutual_information(joint, mx, my)  # n=None
        assert mi == pytest.approx(1.0, abs=1e-6)


# ── channel_capacity ─────────────────────────────────────────────────────────

class TestChannelCapacity:
    def test_no_signal(self):
        assert channel_capacity(0.0, 100.0) == pytest.approx(0.0, abs=1e-9)

    def test_no_noise(self):
        # C = log2(1 + S/0) → log2(1 + S) where S is treated as ratio
        c = channel_capacity(10.0, 0.0)
        assert c == pytest.approx(math.log2(1 + 10.0), abs=1e-9)

    def test_equal_signal_noise(self):
        # SNR = 1, C = log2(2) = 1.0
        assert channel_capacity(5.0, 5.0) == pytest.approx(1.0, abs=1e-9)

    def test_non_negative(self):
        assert channel_capacity(1.0, 100.0) >= 0.0

    def test_monotonic_in_snr(self):
        c1 = channel_capacity(1.0, 10.0)
        c2 = channel_capacity(10.0, 10.0)
        c3 = channel_capacity(100.0, 10.0)
        assert c1 < c2 < c3


# ── js_divergence_from_counters ──────────────────────────────────────────────

class TestJSDivergence:
    def test_identical_distributions(self):
        c = Counter({"a": 10, "b": 10})
        assert js_divergence_from_counters(c, c) == pytest.approx(0.0, abs=1e-9)

    def test_disjoint_distributions(self):
        c1 = Counter({"a": 10})
        c2 = Counter({"b": 10})
        jsd = js_divergence_from_counters(c1, c2)
        # Max JS divergence for disjoint distributions is ln(2) ≈ 0.693
        # (scipy uses natural log internally)
        assert jsd == pytest.approx(math.log(2), abs=0.001)

    def test_symmetric(self):
        c1 = Counter({"a": 30, "b": 10, "c": 5})
        c2 = Counter({"a": 5, "b": 20, "d": 15})
        assert js_divergence_from_counters(c1, c2) == pytest.approx(
            js_divergence_from_counters(c2, c1), abs=1e-10
        )

    def test_bounded_zero_one(self):
        c1 = Counter({"a": 50, "b": 30, "c": 20})
        c2 = Counter({"x": 40, "y": 60})
        jsd = js_divergence_from_counters(c1, c2)
        assert 0.0 <= jsd <= 1.0 + 1e-9

    def test_empty_counter_returns_nan(self):
        c1 = Counter({"a": 10})
        c2 = Counter()
        assert math.isnan(js_divergence_from_counters(c1, c2))


# ── js_distance_from_counters ────────────────────────────────────────────────

class TestJSDistance:
    def test_is_sqrt_of_divergence(self):
        c1 = Counter({"a": 30, "b": 20})
        c2 = Counter({"a": 10, "b": 40})
        dist = js_distance_from_counters(c1, c2)
        div = js_divergence_from_counters(c1, c2)
        assert dist == pytest.approx(math.sqrt(div), abs=1e-9)


# ── js_divergence_matrix ─────────────────────────────────────────────────────

class TestJSDivergenceMatrix:
    def test_single_counter(self):
        mat = js_divergence_matrix([Counter({"a": 10})])
        assert mat.shape == (1, 1)
        assert mat[0, 0] == 0.0

    def test_diagonal_is_zero(self):
        counters = [Counter({"a": 10, "b": 5}), Counter({"a": 5, "c": 10})]
        mat = js_divergence_matrix(counters)
        assert mat[0, 0] == 0.0
        assert mat[1, 1] == 0.0

    def test_symmetric(self):
        counters = [
            Counter({"a": 10, "b": 5}),
            Counter({"a": 5, "c": 10}),
            Counter({"b": 3, "d": 7}),
        ]
        mat = js_divergence_matrix(counters)
        np.testing.assert_array_almost_equal(mat, mat.T)

    def test_shape(self):
        counters = [Counter({"a": i + 1}) for i in range(4)]
        mat = js_divergence_matrix(counters)
        assert mat.shape == (4, 4)

    def test_values_bounded(self):
        counters = [
            Counter({"a": 100}),
            Counter({"b": 100}),
            Counter({"a": 50, "b": 50}),
        ]
        mat = js_divergence_matrix(counters)
        assert np.all(mat >= -1e-10)
        assert np.all(mat <= 1.0 + 1e-10)
