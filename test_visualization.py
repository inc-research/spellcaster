"""Tests for spellcaster.visualization.plots."""

from __future__ import annotations

import os
import tempfile

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
from spellcaster.results.complexity import (
    ComplexityComparisonResult,
    ComplexityFlowResult,
    SentenceMetrics,
)
from spellcaster.results.valence import PostMetrics, ValenceModelResult
from spellcaster.results.evolution import (
    EvolutionaryStatus,
    EvolutionResult,
    SpeciesRecord,
)
from spellcaster.results.keyword import (
    CrossKeywordEntanglement,
    FileKeywordResult,
    KeywordERPResult,
    KeywordMeasures,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_complexity_result():
    def _flow(label, n=20, slope=1.5):
        sents = []
        for i in range(n):
            sents.append(SentenceMetrics(
                text=f"Sentence {i}",
                index=i,
                k_hist=int(100 + i * slope * 10),
                volatility=max(0, int(10 + 5 * np.sin(i * 0.5))),
                synergy=max(0, float(0.5 + 0.3 * np.cos(i * 0.3))),
            ))
        return ComplexityFlowResult(label=label, sentences=sents)

    return ComplexityComparisonResult(flows=[
        _flow("Human", slope=1.5),
        _flow("GPT", slope=1.2),
    ])


def _make_valence_result():
    pm1 = PostMetrics(
        file="human.txt", entropy_text=7.2, shannon_entropy_corpus=0.05,
        shannon_entropy_avg=0.12, shannon_entropy_max=0.25, number_of_windows=4,
        token_count=2000, collapse_auc_norm=0.18, coupling_strength=1.4,
        coupling_orientation=0.2, verb_diversity=0.72, frames_per_1k_tokens=55.0,
        schema_keywords_per_1k_tokens=12.0,
        collapse_curve=[
            {"win_size": 25, "n_windows": 80, "mean_collapse": 0.05, "max_collapse": 0.12},
            {"win_size": 50, "n_windows": 40, "mean_collapse": 0.10, "max_collapse": 0.20},
            {"win_size": 100, "n_windows": 20, "mean_collapse": 0.15, "max_collapse": 0.28},
            {"win_size": 250, "n_windows": 8, "mean_collapse": 0.12, "max_collapse": 0.22},
        ],
    )
    pm2 = PostMetrics(
        file="gpt.txt", entropy_text=6.5, shannon_entropy_corpus=0.12,
        shannon_entropy_avg=0.18, shannon_entropy_max=0.32, number_of_windows=4,
        token_count=1800, collapse_auc_norm=0.25, coupling_strength=0.8,
        coupling_orientation=-0.1, verb_diversity=0.55, frames_per_1k_tokens=42.0,
        schema_keywords_per_1k_tokens=8.0,
        collapse_curve=[
            {"win_size": 25, "n_windows": 72, "mean_collapse": 0.08, "max_collapse": 0.15},
            {"win_size": 50, "n_windows": 36, "mean_collapse": 0.14, "max_collapse": 0.25},
            {"win_size": 100, "n_windows": 18, "mean_collapse": 0.20, "max_collapse": 0.35},
            {"win_size": 250, "n_windows": 7, "mean_collapse": 0.18, "max_collapse": 0.30},
        ],
    )
    return ValenceModelResult(posts=[pm1, pm2])


def _make_evolution_result():
    return EvolutionResult(
        species=[
            SpeciesRecord(0, EvolutionaryStatus.THRIVING, 0.15, 0.30, 0.15, "Cat sat."),
            SpeciesRecord(1, EvolutionaryStatus.DECLINING, 0.25, 0.10, -0.15, "Dog ran."),
            SpeciesRecord(2, EvolutionaryStatus.STABLE, 0.20, 0.22, 0.02, "Fish swam."),
            SpeciesRecord(3, EvolutionaryStatus.EMERGING, 0.0, 0.18, 0.18, "Bird flew."),
            SpeciesRecord(4, EvolutionaryStatus.EXTINCT, 0.15, 0.0, -0.15, "Frog croaked."),
            SpeciesRecord(5, EvolutionaryStatus.STABLE, 0.10, 0.10, 0.0, "Snake slid."),
            SpeciesRecord(6, EvolutionaryStatus.STABLE, 0.15, 0.10, -0.05, "Bear growled."),
        ],
        document_order=["Document 1", "Document 2"],
    )


def _make_keyword_result():
    return KeywordERPResult(
        keywords=["information", "network", "system"],
        files=[
            FileKeywordResult(
                file="essay.txt", label="Human Essay",
                keyword_measures=[
                    KeywordMeasures("information", 12, 1.85, -0.02, 0.72, 5, 0.68, 0.85),
                    KeywordMeasures("network", 8, 1.62, 0.05, 0.65, 3, 0.60, 0.75),
                    KeywordMeasures("system", 6, 1.45, -0.01, 0.58, 2, 0.55, 0.70),
                ],
                cross_keyword=[
                    CrossKeywordEntanglement("information", "network", 0.52, 4, 0.62),
                    CrossKeywordEntanglement("information", "system", 0.48, 2, 0.58),
                    CrossKeywordEntanglement("network", "system", 0.45, 1, 0.55),
                ],
            ),
        ],
    )


# ── LCX plots ────────────────────────────────────────────────────────────────

class TestPlotComplexityFlow:
    def test_returns_fig_axes(self):
        fig, axes = plot_complexity_flow(_make_complexity_result(), style=None)
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 3
        plt.close(fig)

    def test_empty_result(self):
        result = ComplexityComparisonResult(flows=[])
        fig, axes = plot_complexity_flow(result, style=None)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_flow(self):
        result = ComplexityComparisonResult(flows=[
            ComplexityFlowResult("A", [SentenceMetrics("x", 0, 10, 0, 0.0)]),
        ])
        fig, axes = plot_complexity_flow(result, style=None)
        assert len(axes) == 3
        plt.close(fig)


# ── LCVM plots ───────────────────────────────────────────────────────────────

class TestPlotComplexityPlane:
    def test_returns_fig_ax(self):
        fig, ax = plot_complexity_plane(_make_valence_result(), style=None)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_point(self):
        pm = PostMetrics(
            file="one.txt", entropy_text=5.0, shannon_entropy_corpus=0.1,
            shannon_entropy_avg=0.2, shannon_entropy_max=0.3, number_of_windows=2,
            coupling_strength=1.0, collapse_auc_norm=0.2,
            schema_keywords_per_1k_tokens=10.0,
        )
        fig, ax = plot_complexity_plane(ValenceModelResult(posts=[pm]), style=None)
        plt.close(fig)


class TestPlotMultiscaleCollapse:
    def test_returns_fig_ax(self):
        fig, ax = plot_multiscale_collapse(_make_valence_result(), style=None)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_curves(self):
        pm = PostMetrics(
            file="t.txt", entropy_text=5.0, shannon_entropy_corpus=0.1,
            shannon_entropy_avg=0.2, shannon_entropy_max=0.3, number_of_windows=0,
        )
        fig, ax = plot_multiscale_collapse(ValenceModelResult(posts=[pm]), style=None)
        plt.close(fig)


# ── APE plots ────────────────────────────────────────────────────────────────

class TestPlotEvolutionaryStream:
    def test_returns_fig_ax(self):
        fig, ax = plot_evolutionary_stream(_make_evolution_result(), style=None)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty(self):
        fig, ax = plot_evolutionary_stream(EvolutionResult(), style=None)
        plt.close(fig)


class TestPlotSurvivalSlope:
    def test_returns_fig_ax(self):
        fig, ax = plot_survival_slope(_make_evolution_result(), style=None)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty(self):
        fig, ax = plot_survival_slope(EvolutionResult(), style=None)
        plt.close(fig)


# ── KEPM plots ───────────────────────────────────────────────────────────────

class TestPlotKeywordEntropyEvolution:
    def test_returns_fig_axes(self):
        fig, axes = plot_keyword_entropy_evolution(_make_keyword_result(), style=None)
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 3  # 3 keywords
        plt.close(fig)

    def test_empty(self):
        fig, axes = plot_keyword_entropy_evolution(KeywordERPResult(), style=None)
        plt.close(fig)


class TestPlotCrossKeywordHeatmap:
    def test_returns_fig_ax(self):
        fig, ax = plot_cross_keyword_heatmap(_make_keyword_result(), style=None)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty(self):
        fig, ax = plot_cross_keyword_heatmap(KeywordERPResult(), style=None)
        plt.close(fig)


# ── save_figure ──────────────────────────────────────────────────────────────

class TestSaveFigure:
    def test_saves_png(self):
        fig, _ = plot_complexity_flow(_make_complexity_result(), style=None)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.png")
            save_figure(fig, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        plt.close(fig)
