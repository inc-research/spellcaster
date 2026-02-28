"""
Entropy-based measures for text analysis.

Provides Shannon entropy computation, windowed entropy collapse (measuring
local redundancy relative to a document), and multiscale collapse curves
that summarize redundancy structure across multiple window sizes.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np


# ---------------------------------------------------------------------------
# Shannon entropy
# ---------------------------------------------------------------------------

def shannon_entropy(counter: Counter) -> float:
    """
    Compute Shannon entropy (in bits) from a frequency counter.

    Parameters
    ----------
    counter : Counter
        Token → count mapping.

    Returns
    -------
    float
        Entropy in bits.  Returns ``0.0`` for an empty counter.

    Notes
    -----
    .. math::
        H = -\\sum_{i} p_i \\log_2 p_i
    """
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = np.array(list(counter.values()), dtype=float) / total
    # Mask zeros to avoid log(0)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


# ---------------------------------------------------------------------------
# Windowed entropy collapse
# ---------------------------------------------------------------------------

def window_collapse(
    tokens: list[str],
    win_size: int = 250,
) -> list[float]:
    """
    Compute per-window *entropy collapse* values for non-overlapping windows.

    Entropy collapse for a window is defined as the normalized deficit
    of the window's entropy relative to the whole-document entropy:

    .. math::
        \\text{collapse}_w = \\frac{H_{\\text{doc}} - H_w}{H_{\\text{doc}}}

    A value near 1 means the window is highly redundant (low local variety);
    a value near 0 means the window is as diverse as the full document.

    Parameters
    ----------
    tokens : list[str]
        Full document token sequence.
    win_size : int
        Non-overlapping window width (in tokens).

    Returns
    -------
    list[float]
        One collapse value per window.  Tail chunks shorter than 2 tokens
        are dropped.
    """
    h_doc = shannon_entropy(Counter(tokens))
    if h_doc == 0:
        return []

    collapses: list[float] = []
    for start in range(0, len(tokens), win_size):
        chunk = tokens[start : start + win_size]
        if len(chunk) < 2:
            continue
        h_chunk = shannon_entropy(Counter(chunk))
        collapses.append((h_doc - h_chunk) / h_doc)

    return collapses


# ---------------------------------------------------------------------------
# Multiscale collapse
# ---------------------------------------------------------------------------

_DEFAULT_WIN_SIZES: tuple[int, ...] = (25, 50, 100, 250, 500)


def multiscale_collapse_curve(
    tokens: list[str],
    win_sizes: tuple[int, ...] = _DEFAULT_WIN_SIZES,
) -> list[dict]:
    """
    Compute mean and max entropy collapse at multiple window sizes.

    Parameters
    ----------
    tokens : list[str]
        Full document token sequence.
    win_sizes : tuple[int, ...]
        Window widths to evaluate.

    Returns
    -------
    list[dict]
        One dict per window size with keys:
        ``win_size``, ``n_windows``, ``mean_collapse``, ``max_collapse``.
    """
    curve: list[dict] = []
    for w in win_sizes:
        cs = window_collapse(tokens, win_size=w)
        curve.append({
            "win_size": int(w),
            "n_windows": len(cs),
            "mean_collapse": float(np.mean(cs)) if cs else float("nan"),
            "max_collapse": float(np.max(cs)) if cs else float("nan"),
        })
    return curve


def summarize_multiscale_collapse(
    curve: list[dict],
    x_scale: str = "log",
) -> dict:
    """
    Summarize a multiscale collapse curve into scalar metrics.

    Parameters
    ----------
    curve : list[dict]
        Output of :func:`multiscale_collapse_curve`.
    x_scale : str
        ``"log"`` for log-scaled x-axis (window size) in AUC integration,
        ``"linear"`` for raw window sizes.

    Returns
    -------
    dict
        ``collapse_auc`` — trapezoidal area under the mean-collapse curve.
        ``collapse_auc_norm`` — AUC divided by x-range (average collapse across scales).
        ``peak_win_size`` — window size with highest mean collapse.
        ``peak_mean_collapse`` — that maximum value.
    """
    pts = [
        (d["win_size"], d["mean_collapse"])
        for d in curve
        if not math.isnan(d["mean_collapse"])
    ]

    nan_result = dict(
        collapse_auc=float("nan"),
        collapse_auc_norm=float("nan"),
        peak_win_size=None,
        peak_mean_collapse=float("nan"),
    )

    if len(pts) == 0:
        return nan_result

    if len(pts) == 1:
        w, m = pts[0]
        return dict(
            collapse_auc=0.0,
            collapse_auc_norm=float(m),
            peak_win_size=int(w),
            peak_mean_collapse=float(m),
        )

    xs = np.array(
        [math.log(w) if x_scale == "log" else float(w) for w, _ in pts],
        dtype=float,
    )
    ys = np.array([m for _, m in pts], dtype=float)

    # np.trapezoid in numpy 2.x; np.trapz in earlier versions
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    auc = float(_trapz(ys, xs))
    x_range = float(xs.max() - xs.min())
    auc_norm = float(auc / x_range) if x_range > 0 else float(np.mean(ys))

    peak_win, peak_mean = max(pts, key=lambda t: t[1])

    return dict(
        collapse_auc=auc,
        collapse_auc_norm=auc_norm,
        peak_win_size=int(peak_win),
        peak_mean_collapse=float(peak_mean),
    )
