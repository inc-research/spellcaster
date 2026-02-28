"""
Result dataclasses for the Valence Model (LCVM) analyzer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PostMetrics:
    """
    Full metric set for a single document / text, as computed by the
    Valence Model analyzer.
    """

    # ── Identity ─────────────────────────────────────────────────────
    file: str
    """Source path or label."""

    # ── Variation / entropy ──────────────────────────────────────────
    entropy_text: float
    """Shannon entropy of the token distribution (bits)."""

    shannon_entropy_corpus: float | None
    """Normalised entropy deficit relative to the corpus."""

    shannon_entropy_avg: float | None
    """Mean window-collapse at win_size=250."""

    shannon_entropy_max: float | None
    """Max window-collapse at win_size=250."""

    number_of_windows: int
    """Number of non-overlapping 250-token windows."""

    # ── Multiscale redundancy ────────────────────────────────────────
    collapse_curve: list[dict] = field(default_factory=list)
    """Raw multiscale collapse curve (list of dicts from :func:`~spellcaster.core.entropy.multiscale_collapse_curve`)."""

    collapse_auc: float | None = None
    """Trapezoidal AUC of mean-collapse across window sizes."""

    collapse_auc_norm: float | None = None
    """AUC normalised by x-range (average collapse across scales)."""

    peak_win_size: int | None = None
    """Window size with highest mean collapse."""

    peak_mean_collapse: float | None = None
    """That maximum mean-collapse value."""

    # ── Repertoire / action frames ───────────────────────────────────
    token_count: int = 0
    most_common_verb: str | None = None
    most_common_verb_pattern_count: int = 0
    total_frames: int = 0
    unique_verbs: int = 0
    verb_diversity: float | None = None
    frames_per_1k_tokens: float | None = None

    # ── Organisation / information theory on frames ──────────────────
    entropy_frames: float = 0.0
    mi_verb_subject: float | None = None
    mi_verb_object: float | None = None
    coupling_strength: float | None = None
    coupling_orientation: float | None = None

    # ── Semantic breadth (noun deps) ─────────────────────────────────
    total_noun_dependencies: int = 0
    noun_deps_per_1k_tokens: float | None = None
    unique_schema_keywords_in_deps: int = 0
    schema_keywords_per_1k_tokens: float | None = None
    schema_concentration_entropy: float = 0.0
    mean_schema_valence_entropy_topk: float | None = None

    # ── Appendix / detailed nested data ──────────────────────────────
    top_schema_keywords: list[tuple[str, int]] = field(default_factory=list)
    """Top-20 schema keywords with counts."""

    schema_valence_entropy: dict[str, float] = field(default_factory=dict)
    """Valence entropy per top schema keyword."""

    valence_distributions: dict[str, list[str]] = field(default_factory=dict)
    """Valence keyword lists per top schema keyword."""


@dataclass
class ValenceModelResult:
    """
    Aggregate result from the Valence Model analyzer for N texts.

    Returned by :meth:`~spellcaster.analyzers.valence_model.ValenceModelAnalyzer.analyze`.
    """

    posts: list[PostMetrics] = field(default_factory=list)
    """One :class:`PostMetrics` per input text."""

    js_divergence_matrix: np.ndarray | None = None
    """N×N pairwise JS divergence matrix (``None`` when N < 2)."""

    token_capacities: dict[str, pd.DataFrame] = field(default_factory=dict)
    """Per-file DataFrames of token channel capacities."""

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flat DataFrame with one row per text, containing only scalar
        numeric metrics (nested data is excluded).
        """
        _SCALAR_FIELDS = [
            "file", "entropy_text", "shannon_entropy_corpus",
            "shannon_entropy_avg", "shannon_entropy_max", "number_of_windows",
            "collapse_auc", "collapse_auc_norm", "peak_win_size",
            "peak_mean_collapse", "token_count",
            "most_common_verb", "most_common_verb_pattern_count",
            "total_frames", "unique_verbs", "verb_diversity",
            "frames_per_1k_tokens", "entropy_frames",
            "mi_verb_subject", "mi_verb_object",
            "coupling_strength", "coupling_orientation",
            "total_noun_dependencies", "noun_deps_per_1k_tokens",
            "unique_schema_keywords_in_deps", "schema_keywords_per_1k_tokens",
            "schema_concentration_entropy", "mean_schema_valence_entropy_topk",
        ]
        rows = []
        for p in self.posts:
            rows.append({k: getattr(p, k) for k in _SCALAR_FIELDS})
        return pd.DataFrame(rows)

    @property
    def labels(self) -> list[str]:
        return [p.file for p in self.posts]
