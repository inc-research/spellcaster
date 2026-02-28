"""Tests for IO exporters and top-level convenience API."""

from __future__ import annotations

import json
import os
import tempfile

import pandas as pd

from spellcaster.io.exporters import export_csv, export_json
from spellcaster.results.complexity import (
    ComplexityComparisonResult,
    ComplexityFlowResult,
    SentenceMetrics,
)
from spellcaster.results.valence import PostMetrics, ValenceModelResult
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


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _lcx_result():
    return ComplexityComparisonResult(flows=[
        ComplexityFlowResult("A", [
            SentenceMetrics("Hello.", 0, 50, 0, 0.0),
            SentenceMetrics("World.", 1, 80, 30, 1.0),
        ]),
    ])


def _lcvm_result():
    return ValenceModelResult(posts=[
        PostMetrics(file="t.txt", entropy_text=5.0, shannon_entropy_corpus=0.1,
                    shannon_entropy_avg=0.2, shannon_entropy_max=0.3, number_of_windows=2),
    ])


def _ape_result():
    return EvolutionResult(species=[
        SpeciesRecord(0, EvolutionaryStatus.THRIVING, 0.1, 0.3, 0.2, "Cat sat.",
                      pos_composition=[POSComposition("NOUN", 40.0, 8)]),
        SpeciesRecord(1, EvolutionaryStatus.EXTINCT, 0.2, 0.0, -0.2, "Dog ran."),
    ])


def _kepm_result():
    return KeywordERPResult(
        keywords=["info"],
        files=[FileKeywordResult(
            file="a.txt", label="A",
            keyword_measures=[KeywordMeasures("info", 5, 2.0, -0.01, 0.7, 2, 0.6, 0.8)],
            cross_keyword=[],
        )],
    )


# ── export_csv ───────────────────────────────────────────────────────────────

class TestExportCSV:
    def test_lcx(self):
        with tempfile.TemporaryDirectory() as d:
            p = export_csv(_lcx_result(), os.path.join(d, "out.csv"))
            assert p.exists()
            df = pd.read_csv(p)
            assert len(df) == 2

    def test_lcvm(self):
        with tempfile.TemporaryDirectory() as d:
            p = export_csv(_lcvm_result(), os.path.join(d, "out.csv"))
            df = pd.read_csv(p)
            assert len(df) == 1

    def test_ape(self):
        with tempfile.TemporaryDirectory() as d:
            p = export_csv(_ape_result(), os.path.join(d, "out.csv"))
            df = pd.read_csv(p)
            assert len(df) == 2
            assert "status" in df.columns

    def test_kepm(self):
        with tempfile.TemporaryDirectory() as d:
            p = export_csv(_kepm_result(), os.path.join(d, "out.csv"))
            df = pd.read_csv(p)
            assert len(df) == 1


# ── export_json ──────────────────────────────────────────────────────────────

class TestExportJSON:
    def test_lcx(self):
        with tempfile.TemporaryDirectory() as d:
            p = export_json(_lcx_result(), os.path.join(d, "out.json"))
            data = json.loads(p.read_text())
            assert isinstance(data, list)
            assert len(data) == 2

    def test_ape_structured(self):
        """APE results use the structured to_json() format."""
        with tempfile.TemporaryDirectory() as d:
            p = export_json(_ape_result(), os.path.join(d, "out.json"))
            data = json.loads(p.read_text())
            assert isinstance(data, list)
            assert data[0]["classification"]["status"] == "Thriving"
            assert len(data[0]["pos_composition"]) == 1

    def test_lcvm(self):
        with tempfile.TemporaryDirectory() as d:
            p = export_json(_lcvm_result(), os.path.join(d, "out.json"))
            data = json.loads(p.read_text())
            assert isinstance(data, list)

    def test_kepm(self):
        with tempfile.TemporaryDirectory() as d:
            p = export_json(_kepm_result(), os.path.join(d, "out.json"))
            data = json.loads(p.read_text())
            assert isinstance(data, list)


# ── Top-level imports ────────────────────────────────────────────────────────

class TestTopLevelAPI:
    def test_version(self):
        import spellcaster
        assert spellcaster.__version__ == "0.1.0"

    def test_analyzers_importable(self):
        import spellcaster
        assert spellcaster.TextComplexityAnalyzer is not None
        assert spellcaster.ValenceModelAnalyzer is not None
        assert spellcaster.AdaptiveEvolutionAnalyzer is not None
        assert spellcaster.KeywordERPAnalyzer is not None

    def test_convenience_functions_exist(self):
        import spellcaster
        assert callable(spellcaster.analyze_complexity)
        assert callable(spellcaster.analyze_valence)
        assert callable(spellcaster.analyze_evolution)
        assert callable(spellcaster.analyze_keywords)

    def test_io_importable(self):
        import spellcaster
        assert callable(spellcaster.export_csv)
        assert callable(spellcaster.export_json)
        assert callable(spellcaster.load_texts)
        assert callable(spellcaster.texts_from_strings)

    def test_result_types_importable(self):
        import spellcaster
        assert spellcaster.ComplexityComparisonResult is not None
        assert spellcaster.ValenceModelResult is not None
        assert spellcaster.EvolutionResult is not None
        assert spellcaster.KeywordERPResult is not None

    def test_analyze_complexity_from_strings(self):
        import spellcaster
        result = spellcaster.analyze_complexity(
            "The cat sat on the mat. The dog barked loudly.",
            "A fish swam in the pond. Birds flew overhead.",
            from_files=False,
        )
        assert isinstance(result, spellcaster.ComplexityComparisonResult)
        assert len(result.flows) == 2
