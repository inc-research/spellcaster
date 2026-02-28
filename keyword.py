"""
Result dataclasses for the Keyword ER=EPR (KEPM) analyzer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class KeywordMeasures:
    """
    Within-keyword structural coherence measures for one keyword
    in one document.
    """

    keyword: str
    n_mentions: int
    """Number of sentences containing this keyword."""

    mean_entropy: float | None
    """Mean spectral entropy across mention windows."""

    entropy_trend: float | None
    """Slope of spectral entropy over successive mentions (temporal evolution)."""

    mean_coherence: float | None
    """Mean NCD-similarity between consecutive mention windows."""

    n_epr_pairs: int
    """Number of structurally entangled mention pairs exceeding thresholds."""

    mean_epr_strength: float | None
    """Mean entanglement strength across EPR pairs."""

    max_epr_strength: float | None
    """Maximum entanglement strength observed."""


@dataclass
class CrossKeywordEntanglement:
    """
    Cross-keyword structural entanglement between two keywords.
    """

    keyword_a: str
    keyword_b: str
    mean_cross_coherence: float | None
    """Mean structural similarity between windows of keyword A and keyword B."""

    n_cross_epr_pairs: int
    """Number of cross-keyword EPR pairs."""

    mean_cross_epr_strength: float | None
    """Mean entanglement strength across cross-keyword pairs."""


@dataclass
class FileKeywordResult:
    """
    Full keyword-ERP analysis result for one document.
    """

    file: str
    """Source file path or label."""

    label: str
    """User-friendly label."""

    keyword_measures: list[KeywordMeasures] = field(default_factory=list)
    """Per-keyword structural measures."""

    cross_keyword: list[CrossKeywordEntanglement] = field(default_factory=list)
    """Pairwise cross-keyword entanglement scores."""


@dataclass
class KeywordERPResult:
    """
    Aggregate result from the Keyword ER=EPR analyzer for N files.

    Returned by :meth:`~spellcaster.analyzers.keyword_erp.KeywordERPAnalyzer.analyze`.
    """

    files: list[FileKeywordResult] = field(default_factory=list)
    """One entry per input file."""

    keywords: list[str] = field(default_factory=list)
    """The keywords that were analysed."""

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flat DataFrame with one row per (file, keyword) combination.
        """
        rows = []
        for fkr in self.files:
            for km in fkr.keyword_measures:
                rows.append({
                    "file": fkr.file,
                    "label": fkr.label,
                    "keyword": km.keyword,
                    "n_mentions": km.n_mentions,
                    "mean_entropy": km.mean_entropy,
                    "entropy_trend": km.entropy_trend,
                    "mean_coherence": km.mean_coherence,
                    "n_epr_pairs": km.n_epr_pairs,
                    "mean_epr_strength": km.mean_epr_strength,
                    "max_epr_strength": km.max_epr_strength,
                })
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["file", "label", "keyword", "n_mentions",
                     "mean_entropy", "entropy_trend", "mean_coherence",
                     "n_epr_pairs", "mean_epr_strength", "max_epr_strength"]
        )

    def cross_keyword_dataframe(self) -> pd.DataFrame:
        """
        DataFrame with one row per (file, keyword_a, keyword_b) combination.
        """
        rows = []
        for fkr in self.files:
            for ck in fkr.cross_keyword:
                rows.append({
                    "file": fkr.file,
                    "label": fkr.label,
                    "keyword_a": ck.keyword_a,
                    "keyword_b": ck.keyword_b,
                    "mean_cross_coherence": ck.mean_cross_coherence,
                    "n_cross_epr_pairs": ck.n_cross_epr_pairs,
                    "mean_cross_epr_strength": ck.mean_cross_epr_strength,
                })
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["file", "label", "keyword_a", "keyword_b",
                     "mean_cross_coherence", "n_cross_epr_pairs",
                     "mean_cross_epr_strength"]
        )
