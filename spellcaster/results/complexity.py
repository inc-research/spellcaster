"""
Result dataclasses for the Complexity Index (LCX) analyzer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SentenceMetrics:
    """Metrics for a single sentence within a complexity flow."""

    text: str
    """Raw sentence text."""

    index: int
    """Zero-based position in the document."""

    k_hist: int
    """Cumulative compressed size of all text up to and including this sentence (bytes)."""

    volatility: int
    """Levenshtein edit distance from the previous sentence (0 for the first)."""

    synergy: float
    """Ratio of volatility to marginal compression cost (volatility / delta_k)."""


@dataclass
class ComplexityFlowResult:
    """
    Result of analysing a single text's complexity flow.

    Returned by :meth:`~spellcaster.analyzers.complexity_index.TextComplexityAnalyzer.analyze_flow`.
    """

    label: str
    """User-supplied label for this text (e.g. file name or condition)."""

    sentences: list[SentenceMetrics] = field(default_factory=list)
    """Per-sentence metrics in document order."""

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flat DataFrame with one row per sentence.

        Columns: ``index``, ``text``, ``k_hist``, ``volatility``, ``synergy``.
        """
        if not self.sentences:
            return pd.DataFrame(
                columns=["index", "text", "k_hist", "volatility", "synergy"]
            )
        return pd.DataFrame(
            {
                "index": [s.index for s in self.sentences],
                "text": [s.text for s in self.sentences],
                "k_hist": [s.k_hist for s in self.sentences],
                "volatility": [s.volatility for s in self.sentences],
                "synergy": [s.synergy for s in self.sentences],
            }
        )

    @property
    def k_hist_array(self) -> np.ndarray:
        """Cumulative complexity as a NumPy array."""
        return np.array([s.k_hist for s in self.sentences])

    @property
    def volatility_array(self) -> np.ndarray:
        """Volatility sequence as a NumPy array."""
        return np.array([s.volatility for s in self.sentences])

    @property
    def synergy_array(self) -> np.ndarray:
        """Synergy sequence as a NumPy array."""
        return np.array([s.synergy for s in self.sentences])


@dataclass
class ComplexityComparisonResult:
    """
    Result of comparing N texts via complexity flow analysis.

    Returned by :meth:`~spellcaster.analyzers.complexity_index.TextComplexityAnalyzer.compare`.
    """

    flows: list[ComplexityFlowResult] = field(default_factory=list)
    """One :class:`ComplexityFlowResult` per input text."""

    def to_dataframe(self) -> pd.DataFrame:
        """
        Combined DataFrame with a ``label`` column distinguishing texts.
        """
        frames = []
        for flow in self.flows:
            df = flow.to_dataframe()
            df["label"] = flow.label
            frames.append(df)
        if not frames:
            return pd.DataFrame(
                columns=["index", "text", "k_hist", "volatility", "synergy", "label"]
            )
        return pd.concat(frames, ignore_index=True)

    @property
    def labels(self) -> list[str]:
        return [f.label for f in self.flows]
