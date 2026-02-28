"""
Information-theoretic measures for text analysis.

Mutual information, channel capacity (Shannon–Hartley analogue),
and Jensen–Shannon divergence — with N-text generalizations.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from scipy.spatial.distance import jensenshannon


# ---------------------------------------------------------------------------
# Mutual information
# ---------------------------------------------------------------------------

def mutual_information(
    joint: Counter,
    marginal_x: Counter,
    marginal_y: Counter,
    n: int | None = None,
) -> float:
    """
    Compute mutual information I(X; Y) from joint and marginal counters.

    .. math::
        I(X; Y) = \\sum_{x, y} p(x, y) \\log_2 \\frac{p(x, y)}{p(x)\\,p(y)}

    Parameters
    ----------
    joint : Counter
        Mapping of ``(x, y)`` pairs → counts.
    marginal_x : Counter
        Mapping of ``x`` → counts.
    marginal_y : Counter
        Mapping of ``y`` → counts.
    n : int or None
        Total number of observations.  When ``None``, the sum of *joint*
        values is used.

    Returns
    -------
    float
        Mutual information in bits.  Returns ``0.0`` when *n* is 0.
    """
    if n is None:
        n = sum(joint.values())
    if n == 0:
        return 0.0

    mi = 0.0
    for (x, y), count_xy in joint.items():
        p_xy = count_xy / n
        p_x = marginal_x[x] / n
        p_y = marginal_y[y] / n
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * math.log2(p_xy / (p_x * p_y))

    return mi


# ---------------------------------------------------------------------------
# Channel capacity (Shannon–Hartley analogue)
# ---------------------------------------------------------------------------

def channel_capacity(signal: float, noise: float) -> float:
    """
    Shannon–Hartley channel capacity with unit bandwidth.

    .. math::
        C = \\log_2(1 + S/N)

    Parameters
    ----------
    signal : float
        Signal power (e.g. token frequency).
    noise : float
        Noise power (e.g. total other-token frequency).

    Returns
    -------
    float
        Channel capacity in bits.
    """
    sn_ratio = signal / noise if noise > 0 else signal
    sn_ratio = max(0.0, sn_ratio)
    return math.log2(1 + sn_ratio)


# ---------------------------------------------------------------------------
# Jensen–Shannon divergence
# ---------------------------------------------------------------------------

def js_divergence_from_counters(
    c1: Counter,
    c2: Counter,
) -> float:
    """
    Jensen–Shannon *divergence* between two frequency counters.

    JS divergence is the square of the JS distance returned by
    ``scipy.spatial.distance.jensenshannon``.

    Parameters
    ----------
    c1, c2 : Counter
        Token frequency counters.

    Returns
    -------
    float
        JS divergence.  Ranges from 0 (identical) to ``ln(2) ≈ 0.693``
        for fully disjoint distributions (scipy uses natural log internally).
        Returns ``NaN`` if either counter is empty.
    """
    vocab = sorted(set(c1.keys()) | set(c2.keys()))
    p = np.array([c1.get(t, 0) for t in vocab], dtype=float)
    q = np.array([c2.get(t, 0) for t in vocab], dtype=float)

    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")

    js_dist = jensenshannon(p, q)
    return float(js_dist ** 2)


def js_distance_from_counters(c1: Counter, c2: Counter) -> float:
    """
    Jensen–Shannon *distance* (the square root of JS divergence).

    Parameters
    ----------
    c1, c2 : Counter
        Token frequency counters.

    Returns
    -------
    float
        JS distance in [0, 1].
    """
    vocab = sorted(set(c1.keys()) | set(c2.keys()))
    p = np.array([c1.get(t, 0) for t in vocab], dtype=float)
    q = np.array([c2.get(t, 0) for t in vocab], dtype=float)

    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")

    return float(jensenshannon(p, q))


def js_divergence_matrix(counters: list[Counter]) -> np.ndarray:
    """
    Compute the pairwise JS divergence matrix for *N* frequency counters.

    Parameters
    ----------
    counters : list[Counter]
        One frequency counter per document.

    Returns
    -------
    np.ndarray
        Symmetric N×N matrix where entry ``[i, j]`` is the JS divergence
        between documents *i* and *j*.  Diagonal is 0.
    """
    n = len(counters)
    mat = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            d = js_divergence_from_counters(counters[i], counters[j])
            mat[i, j] = d
            mat[j, i] = d

    return mat
