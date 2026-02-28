"""
Visualization functions for Spellcaster results.

All functions return ``(fig, axes)`` and never call ``plt.show()``.
"""

from spellcaster.visualization.plots import (
    plot_complexity_flow,
    plot_complexity_plane,
    plot_multiscale_collapse,
    plot_evolutionary_stream,
    plot_survival_slope,
    plot_keyword_entropy_evolution,
    plot_cross_keyword_heatmap,
    save_figure,
)

__all__ = [
    "plot_complexity_flow",
    "plot_complexity_plane",
    "plot_multiscale_collapse",
    "plot_evolutionary_stream",
    "plot_survival_slope",
    "plot_keyword_entropy_evolution",
    "plot_cross_keyword_heatmap",
    "save_figure",
]
