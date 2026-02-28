"""Statistical utilities for text analysis."""

import numpy as np


def repetition_intervals(
    tokens: list[str],
    keywords: list[str],
) -> dict[str, list[int]]:
    """
    Compute the token-distance between consecutive occurrences of each keyword.

    Parameters
    ----------
    tokens : list[str]
        Ordered token sequence (e.g. lemmas from a document).
    keywords : list[str]
        Keywords to track.

    Returns
    -------
    dict mapping each keyword to its list of inter-occurrence gaps (in tokens).
    A keyword with fewer than two occurrences will have an empty list.
    """
    intervals: dict[str, list[int]] = {kw: [] for kw in keywords}
    last_seen: dict[str, int] = {}

    keyword_set = set(keywords)
    for i, token in enumerate(tokens):
        if token in keyword_set:
            if token in last_seen:
                intervals[token].append(i - last_seen[token])
            last_seen[token] = i

    return intervals


def interval_summary(intervals: list[int]) -> dict[str, float | None]:
    """
    Compute mean, median, std, and count for a list of repetition intervals.

    Returns a dict with keys ``mean``, ``median``, ``std``, ``count``.
    Values are ``None`` when the list is empty.
    """
    if not intervals:
        return {"mean": None, "median": None, "std": None, "count": 0}

    arr = np.asarray(intervals, dtype=float)
    return {
        "mean": round(float(arr.mean()), 4),
        "median": float(np.median(arr)),
        "std": round(float(arr.std()), 4),
        "count": len(intervals),
    }
