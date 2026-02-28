"""
Visualization functions for Spellcaster results.

All functions follow the same pattern:

* Accept a result object from one of the analyzers.
* Return ``(fig, axes)`` so the caller can customise or save.
* Never call ``plt.show()`` — that is left to the caller.
* Use a consistent visual style (``dark_background`` by default).

Dependencies: ``matplotlib`` (required) and ``seaborn`` (optional,
used only by :func:`plot_cross_keyword_heatmap`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from spellcaster.utils.imports import require_matplotlib
from spellcaster.utils.smoothing import smooth

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

    from spellcaster.results.complexity import ComplexityComparisonResult
    from spellcaster.results.valence import ValenceModelResult
    from spellcaster.results.evolution import EvolutionResult
    from spellcaster.results.keyword import KeywordERPResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Palette helpers
# ---------------------------------------------------------------------------

_DEFAULT_STYLE = "dark_background"

# Colour cycles for dark backgrounds
_LINE_COLOURS = [
    "#00FFFF", "#FF00FF", "#00FF00", "#FFA500",
    "#FFFF00", "#FF69B4", "#7DF9FF", "#FF6347",
]
_LINE_STYLES = ["-", "--", "-.", ":"]


def _apply_style(style: str | None = _DEFAULT_STYLE) -> None:
    """Apply matplotlib style, or skip if None."""
    if style is not None:
        plt_ = require_matplotlib()
        plt_.style.use(style)


def _cycle(lst: list, idx: int):
    return lst[idx % len(lst)]


# =====================================================================
# 1. Complexity Index (LCX)
# =====================================================================

def plot_complexity_flow(
    result: ComplexityComparisonResult,
    smoothing_window: int = 4,
    style: str | None = _DEFAULT_STYLE,
    figsize: tuple[float, float] = (12, 12),
) -> tuple[Figure, list[Axes]]:
    """
    Three-panel plot of complexity flow (LCX).

    Panels:
    1. Narrative accumulation (cumulative compressed size).
    2. Lexical volatility (Levenshtein edit distance).
    3. Cognitive synergy (smoothed volatility / marginal info).

    Parameters
    ----------
    result : ComplexityComparisonResult
        Output of :meth:`TextComplexityAnalyzer.compare`.
    smoothing_window : int
        Moving-average window applied to the synergy panel.
    style : str or None
        Matplotlib style name (set ``None`` to keep current style).
    figsize : tuple
        Figure size ``(width, height)`` in inches.

    Returns
    -------
    (Figure, list[Axes])
    """
    plt_ = require_matplotlib()
    _apply_style(style)

    fig, axes = plt_.subplots(3, 1, figsize=figsize)

    for idx, flow in enumerate(result.flows):
        c = _cycle(_LINE_COLOURS, idx)
        ls = _cycle(_LINE_STYLES, idx)
        label = flow.label

        n = len(flow.sentences)
        if n == 0:
            continue

        x = list(range(n))
        k_hist = flow.k_hist_array
        vol = flow.volatility_array
        syn = flow.synergy_array

        # Panel 1: Narrative accumulation
        axes[0].plot(x, k_hist, label=label, color=c, linestyle=ls)

        # Panel 2: Lexical volatility
        axes[1].plot(x, vol, label=label, color=c, linestyle=ls, alpha=0.8)

        # Panel 3: Smoothed synergy
        syn_smooth = smooth(syn.tolist(), window=smoothing_window)
        axes[2].plot(
            range(len(syn_smooth)), syn_smooth,
            label=label, color=c, linestyle=ls, linewidth=2,
        )

    # Titles and labels
    axes[0].set_title(
        "1. Language Complexity Accumulation (LZ77 History)\n"
        "(Steeper slope = More unique info introduced per sentence)"
    )
    axes[0].set_ylabel("Compressed Size (Bytes)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title(
        "2. Language Encoding Volatility (Levenshtein)\n"
        "(Higher = More distinct change between sentences)"
    )
    axes[1].set_ylabel("Edit Distance")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title(
        "3. LBX | Language Behavior Index\n"
        "(Efficiency of Complexity Constraint)"
    )
    axes[2].set_ylabel("Synergy Ratio")
    axes[2].set_xlabel("Sentence Index (Time)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, list(axes)


# =====================================================================
# 2. Valence Model (LCVM)
# =====================================================================

def plot_complexity_plane(
    result: ValenceModelResult,
    x_col: str = "entropy_text",
    y_col: str = "coupling_strength",
    size_col: str = "schema_keywords_per_1k_tokens",
    color_col: str = "collapse_auc_norm",
    label_col: str = "file",
    style: str | None = _DEFAULT_STYLE,
    figsize: tuple[float, float] = (9, 6),
) -> tuple[Figure, Axes]:
    """
    2D complexity plane (bubble chart).

    * x = entropy (variation)
    * y = MI coupling strength (organisation)
    * size = schema breadth
    * colour = multiscale collapse (redundancy)

    Parameters
    ----------
    result : ValenceModelResult
        Output of :meth:`ValenceModelAnalyzer.analyze`.
    x_col, y_col, size_col, color_col, label_col : str
        Column names in the result's flat DataFrame.
    style, figsize : as above.

    Returns
    -------
    (Figure, Axes)
    """
    plt_ = require_matplotlib()
    _apply_style(style)

    df = result.to_dataframe()

    # Handle missing columns gracefully
    for col in [x_col, y_col, size_col, color_col]:
        if col not in df.columns:
            df[col] = 0.0

    # Bubble sizes
    s_raw = df[size_col].to_numpy(dtype=float)
    s_min, s_max = np.nanmin(s_raw), np.nanmax(s_raw)
    denom = (s_max - s_min) if s_max > s_min else 1.0
    sizes = 80.0 + 520.0 * ((s_raw - s_min) / denom)

    fig, ax = plt_.subplots(figsize=figsize)
    sc = ax.scatter(
        df[x_col].to_numpy(dtype=float),
        df[y_col].to_numpy(dtype=float),
        s=sizes,
        c=df[color_col].to_numpy(dtype=float),
        alpha=0.85,
        cmap="viridis",
    )

    # Annotate points
    for _, row in df.iterrows():
        lbl = Path(str(row.get(label_col, ""))).stem
        ax.annotate(
            lbl,
            (row[x_col], row[y_col]),
            xytext=(6, 6),
            textcoords="offset points",
        )

    ax.set_xlabel("Variation: Text entropy (bits)")
    ax.set_ylabel("Organization: Coupling strength (bits)")
    ax.set_title("Complexity Plane")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Redundancy: Multiscale collapse AUC_norm")

    fig.tight_layout()
    return fig, ax


def plot_multiscale_collapse(
    result: ValenceModelResult,
    style: str | None = _DEFAULT_STYLE,
    figsize: tuple[float, float] = (10, 5),
) -> tuple[Figure, Axes]:
    """
    Multiscale collapse curves (mean collapse vs. window size).

    One line per document.

    Parameters
    ----------
    result : ValenceModelResult

    Returns
    -------
    (Figure, Axes)
    """
    plt_ = require_matplotlib()
    _apply_style(style)

    fig, ax = plt_.subplots(figsize=figsize)

    for idx, post in enumerate(result.posts):
        c = _cycle(_LINE_COLOURS, idx)
        ls = _cycle(_LINE_STYLES, idx)
        if not post.collapse_curve:
            continue
        xs = [pt["win_size"] for pt in post.collapse_curve]
        ys = [pt["mean_collapse"] for pt in post.collapse_curve]
        ax.plot(xs, ys, marker="o", label=post.file, color=c, linestyle=ls)

    ax.set_xlabel("Window Size (tokens)")
    ax.set_ylabel("Mean Entropy Collapse")
    ax.set_title("Multiscale Collapse Curves")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax


# =====================================================================
# 3. Adaptive Evolution (APE)
# =====================================================================

def plot_evolutionary_stream(
    result: EvolutionResult,
    top_n: int = 15,
    style: str | None = _DEFAULT_STYLE,
    figsize: tuple[float, float] = (12, 7),
) -> tuple[Figure, Axes]:
    """
    Stacked-area chart showing structural species succession across
    two documents.

    Parameters
    ----------
    result : EvolutionResult
        Output of :meth:`AdaptiveEvolutionAnalyzer.analyze`.
    top_n : int
        Number of top species (by end density) to display.
    style, figsize : as above.

    Returns
    -------
    (Figure, Axes)
    """
    plt_ = require_matplotlib()
    _apply_style(style)

    species = result.species[:top_n]
    if not species or len(result.document_order) < 2:
        fig, ax = plt_.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        return fig, ax

    labels = [f"Group {s.cluster_id}" for s in species]
    y_start = [s.density_start for s in species]
    y_end = [s.density_end for s in species]
    y = np.array([y_start, y_end])

    x = [0, 1]
    cmap = plt_.cm.tab20(np.linspace(0, 1, len(labels)))

    fig, ax = plt_.subplots(figsize=figsize)
    ax.stackplot(x, y.T, labels=labels, colors=cmap, alpha=0.85)

    doc_order = result.document_order
    ax.set_xticks([0, 1])
    ax.set_xticklabels([doc_order[0], doc_order[-1]], fontsize=12)
    ax.set_ylabel("Cognitive Market Share (Density)")
    ax.set_title("Evolutionary Stream: Structural Succession")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), title="Top Structures")

    fig.tight_layout()
    return fig, ax


def plot_survival_slope(
    result: EvolutionResult,
    top_n: int = 5,
    style: str | None = _DEFAULT_STYLE,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[Figure, Axes]:
    """
    Slope chart of the top N movers (biggest absolute density change).

    Green lines = thriving, red lines = declining.

    Parameters
    ----------
    result : EvolutionResult
    top_n : int
        Number of top movers to display.
    style, figsize : as above.

    Returns
    -------
    (Figure, Axes)
    """
    plt_ = require_matplotlib()
    _apply_style(style)

    if not result.species or len(result.document_order) < 2:
        fig, ax = plt_.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        return fig, ax

    # Sort by absolute delta
    ranked = sorted(result.species, key=lambda s: abs(s.delta), reverse=True)[:top_n]

    doc_order = result.document_order
    fig, ax = plt_.subplots(figsize=figsize)

    max_density = 0.0
    for s in ranked:
        d1, d2 = s.density_start, s.density_end
        max_density = max(max_density, d1, d2)
        color = "#2ecc71" if d2 > d1 else "#e74c3c"
        ax.plot([0, 1], [d1, d2], marker="o", markersize=8,
                color=color, linewidth=2)
        ax.text(-0.05, d1, f"G{s.cluster_id} ({d1:.2f})",
                ha="right", color=color)
        ax.text(1.05, d2, f"G{s.cluster_id} ({d2:.2f})",
                ha="left", color=color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([doc_order[0], doc_order[-1]], fontsize=12)
    ax.set_ylabel("Cognitive Market Share (Density)")
    ax.set_title(f"Survival of the Fittest: Top {top_n} Movers")
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(0, max_density * 1.15 if max_density > 0 else 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for spine in ["right", "left", "top"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    return fig, ax


# =====================================================================
# 4. Keyword ERP (KEPM)
# =====================================================================

def plot_keyword_entropy_evolution(
    result: KeywordERPResult,
    file_idx: int = 0,
    style: str | None = _DEFAULT_STYLE,
    figsize_per_kw: tuple[float, float] = (14, 4),
) -> tuple[Figure, list[Axes]]:
    """
    Spectral entropy evolution across keyword mentions (one subplot
    per keyword).

    Parameters
    ----------
    result : KeywordERPResult
        Output of :meth:`KeywordERPAnalyzer.analyze`.
    file_idx : int
        Which file's results to plot.
    style : str or None
    figsize_per_kw : tuple
        Size per keyword subplot.

    Returns
    -------
    (Figure, list[Axes])
    """
    plt_ = require_matplotlib()
    _apply_style(style)

    if file_idx >= len(result.files):
        fig, ax = plt_.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return fig, [ax]

    file_result = result.files[file_idx]
    kw_measures = file_result.keyword_measures

    n_kw = len(kw_measures)
    if n_kw == 0:
        fig, ax = plt_.subplots()
        ax.text(0.5, 0.5, "No keywords", ha="center", va="center",
                transform=ax.transAxes)
        return fig, [ax]

    fig, axes = plt_.subplots(n_kw, 1,
                              figsize=(figsize_per_kw[0], figsize_per_kw[1] * n_kw))
    if n_kw == 1:
        axes = [axes]

    for ax, km in zip(axes, kw_measures):
        if km.n_mentions < 2 or km.mean_entropy is None:
            ax.text(0.5, 0.5, f"Insufficient data for '{km.keyword}'",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        # We don't have the per-mention entropy array in the result
        # dataclass (only aggregates), so show the summary info
        ax.barh(
            [0, 1, 2],
            [km.mean_entropy, km.mean_coherence or 0.0, km.n_epr_pairs],
            color=["#00FFFF", "#FF00FF", "#FFFF00"],
            tick_label=["Mean Entropy", "Mean Coherence", "EPR Pairs"],
        )
        ax.set_title(f'Keyword: "{km.keyword}" ({km.n_mentions} mentions)')

    fig.tight_layout()
    return fig, list(axes)


def plot_cross_keyword_heatmap(
    result: KeywordERPResult,
    file_idx: int = 0,
    style: str | None = _DEFAULT_STYLE,
    figsize: tuple[float, float] = (8, 6),
) -> tuple[Figure, Axes]:
    """
    Heatmap of cross-keyword structural coherence.

    Parameters
    ----------
    result : KeywordERPResult
    file_idx : int
    style, figsize : as above.

    Returns
    -------
    (Figure, Axes)
    """
    plt_ = require_matplotlib()
    from spellcaster.utils.imports import require_seaborn
    sns = require_seaborn()
    _apply_style(style)

    if file_idx >= len(result.files):
        fig, ax = plt_.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return fig, ax

    xdf = result.files[file_idx].cross_keyword

    # Build matrix
    kws = list(result.keywords)
    n = len(kws)
    mat = np.zeros((n, n), dtype=float)
    kw_idx = {kw: i for i, kw in enumerate(kws)}

    for ck in xdf:
        i = kw_idx.get(ck.keyword_a)
        j = kw_idx.get(ck.keyword_b)
        if i is not None and j is not None and ck.mean_cross_coherence is not None:
            mat[i, j] = ck.mean_cross_coherence
            mat[j, i] = ck.mean_cross_coherence
    np.fill_diagonal(mat, 1.0)

    fig, ax = plt_.subplots(figsize=figsize)
    sns.heatmap(
        pd.DataFrame(mat, index=kws, columns=kws),
        cmap="plasma",
        ax=ax,
        vmin=0, vmax=1,
        annot=True, fmt=".2f",
        cbar_kws={"label": "Mean Cross-Coherence"},
    )
    ax.set_title(f"Cross-Keyword Structural Coherence — {result.files[file_idx].label}")

    fig.tight_layout()
    return fig, ax


# =====================================================================
# Convenience: save figure
# =====================================================================

def save_figure(
    fig: Figure,
    path: str,
    dpi: int = 300,
) -> None:
    """
    Save a matplotlib figure to disk.

    Parameters
    ----------
    fig : Figure
    path : str
        Output file path (e.g. ``"plot.png"``).
    dpi : int
        Resolution.
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    logger.info("Saved figure to %s", path)
