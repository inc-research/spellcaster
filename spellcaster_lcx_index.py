"""Language complexity and flow analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass

import re
import zlib

import Levenshtein
import numpy as np


@dataclass
class TextComplexityResult:
    """Structured results for text complexity analysis."""

    sentences: list[str]
    k_hist: list[int]
    volatility: list[int]
    synergy: list[float]


class TextComplexityAnalyzer:
    """Analyze textual complexity and flow sentence-by-sentence."""

    def __init__(self) -> None:
        self.history_str = ""
        self.last_sentence = ""
        self.last_k_hist = 0

    @staticmethod
    def get_complexity(text: str) -> int:
        """Return compressed byte length for the given text."""
        if not text:
            return 0
        return len(zlib.compress(text.encode("utf-8")))

    def analyze_flow(self, text_content: str) -> TextComplexityResult:
        """Analyze a document and return complexity metrics."""
        clean_text = re.sub(r"<[^>]*>", "", text_content)
        sentences = [s.strip() for s in re.split(r"[.\n•]+", clean_text) if s.strip()]

        data_k_hist: list[int] = []
        data_volatility: list[int] = []
        data_synergy: list[float] = []

        for sentence in sentences:
            self.history_str += f"{sentence} "
            k_curr = self.get_complexity(self.history_str)

            delta_k = max(1, k_curr - self.last_k_hist)

            lev_dist = 0
            if self.last_sentence:
                lev_dist = Levenshtein.distance(self.last_sentence, sentence)

            synergy = lev_dist / delta_k if delta_k > 0 else 0

            data_k_hist.append(k_curr)
            data_volatility.append(lev_dist)
            data_synergy.append(synergy)

            self.last_sentence = sentence
            self.last_k_hist = k_curr

        return TextComplexityResult(
            sentences=sentences,
            k_hist=data_k_hist,
            volatility=data_volatility,
            synergy=data_synergy,
        )


def smooth(data: list[float] | list[int], window: int = 3) -> np.ndarray:
    """Return moving-average smoothing for a numeric series."""
    if len(data) == 0:
        return np.array([])
    if len(data) < window:
        return np.array(data)
    kernel = np.ones(window) / window
    return np.convolve(np.array(data, dtype=float), kernel, mode="valid")
