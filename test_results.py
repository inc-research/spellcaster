"""Tests for spellcaster.results (all result dataclasses)."""

from __future__ import annotations

import pandas as pd

from spellcaster.results.complexity import (
    ComplexityComparisonResult,
    ComplexityFlowResult,
    SentenceMetrics,
)
from spellcaster.results.evolution import (
    EvolutionaryStatus,
    EvolutionResult,
    POSComposition,
    SpeciesRecord,
)
from spellcaster.results.keyword import (
    CrossKeywordEntanglement,
    FileKeywordResult,
    KeywordERPResult,
    KeywordMeasures,
)
from spellcaster.results.valence import PostMetrics, ValenceModelResult


# ── Complexity Results ───────────────────────────────────────────────────────

class TestComplexityFlowResult:
    def _make_flow(self):
        return ComplexityFlowResult(
            label="test",
            sentences=[
                SentenceMetrics(text="Hello.", index=0, k_hist=20, volatility=0, synergy=0.0),
                SentenceMetrics(text="World.", index=1, k_hist=30, volatility=5, synergy=0.5),
                SentenceMetrics(text="End.", index=2, k_hist=38, volatility=4, synergy=0.4),
            ],
        )

    def test_to_dataframe_columns(self):
        df = self._make_flow().to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"index", "text", "k_hist", "volatility", "synergy"}
        assert len(df) == 3

    def test_to_dataframe_empty(self):
        df = ComplexityFlowResult(label="empty").to_dataframe()
        assert len(df) == 0
        assert "k_hist" in df.columns

    def test_array_properties(self):
        flow = self._make_flow()
        assert list(flow.k_hist_array) == [20, 30, 38]
        assert list(flow.volatility_array) == [0, 5, 4]
        assert len(flow.synergy_array) == 3


class TestComplexityComparisonResult:
    def test_to_dataframe_combined(self):
        f1 = ComplexityFlowResult(label="A", sentences=[
            SentenceMetrics("a", 0, 10, 0, 0.0),
        ])
        f2 = ComplexityFlowResult(label="B", sentences=[
            SentenceMetrics("b", 0, 15, 0, 0.0),
        ])
        result = ComplexityComparisonResult(flows=[f1, f2])
        df = result.to_dataframe()
        assert len(df) == 2
        assert "label" in df.columns
        assert list(df["label"]) == ["A", "B"]

    def test_labels_property(self):
        result = ComplexityComparisonResult(flows=[
            ComplexityFlowResult(label="X"),
            ComplexityFlowResult(label="Y"),
        ])
        assert result.labels == ["X", "Y"]

    def test_empty(self):
        df = ComplexityComparisonResult().to_dataframe()
        assert len(df) == 0


# ── Valence Results ──────────────────────────────────────────────────────────

class TestValenceModelResult:
    def test_to_dataframe(self):
        pm = PostMetrics(
            file="test.txt",
            entropy_text=5.2,
            shannon_entropy_corpus=0.1,
            shannon_entropy_avg=0.3,
            shannon_entropy_max=0.5,
            number_of_windows=4,
            token_count=1000,
        )
        result = ValenceModelResult(posts=[pm])
        df = result.to_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["entropy_text"] == 5.2
        assert df.iloc[0]["file"] == "test.txt"

    def test_excludes_nested_data(self):
        pm = PostMetrics(
            file="t.txt",
            entropy_text=5.0,
            shannon_entropy_corpus=0.1,
            shannon_entropy_avg=0.2,
            shannon_entropy_max=0.3,
            number_of_windows=2,
            top_schema_keywords=[("cat", 10), ("dog", 5)],
        )
        df = ValenceModelResult(posts=[pm]).to_dataframe()
        # Nested fields should NOT be in the flat DataFrame
        assert "top_schema_keywords" not in df.columns
        assert "schema_valence_entropy" not in df.columns

    def test_labels(self):
        result = ValenceModelResult(posts=[
            PostMetrics(file="a.txt", entropy_text=0, shannon_entropy_corpus=0,
                        shannon_entropy_avg=0, shannon_entropy_max=0, number_of_windows=0),
            PostMetrics(file="b.txt", entropy_text=0, shannon_entropy_corpus=0,
                        shannon_entropy_avg=0, shannon_entropy_max=0, number_of_windows=0),
        ])
        assert result.labels == ["a.txt", "b.txt"]


# ── Evolution Results ────────────────────────────────────────────────────────

class TestEvolutionaryStatus:
    def test_values(self):
        assert EvolutionaryStatus.EMERGING.value == "Emerging"
        assert EvolutionaryStatus.EXTINCT.value == "Extinct"
        assert EvolutionaryStatus.THRIVING.value == "Thriving"

    def test_from_value(self):
        assert EvolutionaryStatus("Stable") == EvolutionaryStatus.STABLE


class TestEvolutionResult:
    def _make_result(self):
        return EvolutionResult(
            species=[
                SpeciesRecord(
                    cluster_id=0,
                    status=EvolutionaryStatus.THRIVING,
                    density_start=0.1,
                    density_end=0.3,
                    delta=0.2,
                    sample_sentence="The cat sat.",
                    pos_composition=[POSComposition("NOUN", 30.0, 15)],
                ),
                SpeciesRecord(
                    cluster_id=1,
                    status=EvolutionaryStatus.EXTINCT,
                    density_start=0.2,
                    density_end=0.0,
                    delta=-0.2,
                    sample_sentence="Dogs ran fast.",
                ),
            ],
            document_order=["doc1", "doc2"],
        )

    def test_to_dataframe(self):
        df = self._make_result().to_dataframe()
        assert len(df) == 2
        assert df.iloc[0]["status"] == "Thriving"
        assert df.iloc[1]["delta"] == -0.2

    def test_to_json(self):
        records = self._make_result().to_json()
        assert len(records) == 2
        assert records[0]["classification"]["status"] == "Thriving"
        assert records[0]["classification"]["trend"] == "Positive"
        assert records[1]["classification"]["trend"] == "Negative"
        assert len(records[0]["pos_composition"]) == 1

    def test_empty(self):
        df = EvolutionResult().to_dataframe()
        assert len(df) == 0


# ── Keyword Results ──────────────────────────────────────────────────────────

class TestKeywordERPResult:
    def _make_result(self):
        return KeywordERPResult(
            keywords=["info", "net"],
            files=[
                FileKeywordResult(
                    file="a.txt",
                    label="A",
                    keyword_measures=[
                        KeywordMeasures("info", 5, 2.1, -0.01, 0.8, 3, 0.7, 0.9),
                        KeywordMeasures("net", 3, 1.8, 0.05, 0.6, 1, 0.5, 0.5),
                    ],
                    cross_keyword=[
                        CrossKeywordEntanglement("info", "net", 0.4, 2, 0.6),
                    ],
                ),
            ],
        )

    def test_to_dataframe(self):
        df = self._make_result().to_dataframe()
        assert len(df) == 2
        assert set(df["keyword"]) == {"info", "net"}
        assert df.iloc[0]["n_mentions"] == 5

    def test_cross_keyword_dataframe(self):
        df = self._make_result().cross_keyword_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["keyword_a"] == "info"
        assert df.iloc[0]["keyword_b"] == "net"

    def test_empty(self):
        df = KeywordERPResult().to_dataframe()
        assert len(df) == 0
        assert "keyword" in df.columns
