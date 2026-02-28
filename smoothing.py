"""Smoothing, normalization, and general numeric utilities."""

import numpy as np
from numpy.typing import ArrayLike


def smooth(data: ArrayLike, window: int = 3) -> np.ndarray:
    """
    Apply a simple moving-average convolution to *data*.

    Parameters
    ----------
    data : array-like
        1-D numeric sequence to smooth.
    window : int
        Width of the averaging kernel.  Must be >= 1.

    Returns
    -------
    np.ndarray
        Smoothed array.  Length is ``max(0, len(data) - window + 1)``
        (convolution in ``'valid'`` mode).
    """
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        return np.array([], dtype=float)
    window = max(1, window)
    if arr.size < window:
        return arr.copy()
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def per_1k(count: int | float, total: int | float) -> float:
    """
    Normalize *count* to a per-1 000-token rate.

    Returns ``NaN`` when *total* is zero.
    """
    if total == 0:
        return float("nan")
    return float(count) / float(total) * 1000.0
