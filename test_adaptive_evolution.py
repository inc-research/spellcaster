"""Tests for spellcaster.analyzers.adaptive_evolution."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from spellcaster.analyzers.adaptive_evolution import AdaptiveEvolutionAnalyzer
from spellcaster.extractors.sentence_parser import ParsedSentence
from spellcaster.results.evolution import (
    EvolutionaryStatus,
    EvolutionResult,
    POSComposition,
    SpeciesRecord,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_df(n_per_doc: int = 5, n_docs: int = 2) -> tuple[pd.DataFrame, list[str]]:
    """Synthetic DataFrame with POS_Tags, Sentence, Source_Document, Group_ID."""
    rows = []
    doc_labels = [f"doc_{i}" for i in range(n_docs)]

    for doc_idx, doc_label in enumerate(doc_labels):
        for sent_idx in range(n_per_doc):
            # Alternate group assignment to create variety
            group = (sent_idx + doc_idx) % 3
            rows.append({
                "Sentence": f"Sentence {sent_idx} from {doc_label}.",
                "POS_Tags": ["DET", "NOUN", "VERB", "PUNCT"] if group != 2
                           else ["ADJ", "NOUN", "ADP", "NOUN", "PUNCT"],
                "Source_Document": doc_label,
                "Group_ID": group,
            })

    return pd.DataFrame(rows), doc_labels


# ── _classify_status ─────────────────────────────────────────────────────────

class TestClassifyStatus:
    def setup_method(self):
        self.ape = AdaptiveEvolutionAnalyzer(status_threshold=0.02)

    def test_emerging(self):
        assert self.ape._classify_status(0.0, 0.15, 0.15) == EvolutionaryStatus.EMERGING

    def test_extinct(self):
        assert self.ape._classify_status(0.1, 0.0, -0.1) == EvolutionaryStatus.EXTINCT

    def test_thriving(self):
        assert self.ape._classify_status(0.1, 0.2, 0.1) == EvolutionaryStatus.THRIVING

    def test_declining(self):
        assert self.ape._classify_status(0.2, 0.1, -0.1) == EvolutionaryStatus.DECLINING

    def test_stable(self):
        assert self.ape._classify_status(0.1, 0.11, 0.01) == EvolutionaryStatus.STABLE

    def test_custom_threshold(self):
        ape2 = AdaptiveEvolutionAnalyzer(status_threshold=0.10)
        assert ape2._classify_status(0.1, 0.15, 0.05) == EvolutionaryStatus.STABLE
        assert ape2._classify_status(0.1, 0.25, 0.15) == EvolutionaryStatus.THRIVING

    def test_emerging_takes_priority(self):
        """Even with a large delta, d_start=0 means Emerging."""
        assert self.ape._classify_status(0.0, 0.5, 0.5) == EvolutionaryStatus.EMERGING

    def test_extinct_takes_priority(self):
        """Even with a large negative delta, d_end=0 means Extinct."""
        assert self.ape._classify_status(0.5, 0.0, -0.5) == EvolutionaryStatus.EXTINCT


# ── _compute_pos_composition ─────────────────────────────────────────────────

class TestComputePOSComposition:
    def setup_method(self):
        self.ape = AdaptiveEvolutionAnalyzer(top_k_pos=5)

    def test_basic(self):
        df = pd.DataFrame({
            "POS_Tags": [
                ["DET", "NOUN", "VERB", "PUNCT"],
                ["DET", "NOUN", "ADJ", "NOUN", "PUNCT"],
            ],
        })
        comp = self.ape._compute_pos_composition(df)
        assert isinstance(comp, list)
        assert len(comp) <= 5
        assert all(isinstance(c, POSComposition) for c in comp)

        # NOUN should appear most (3 times)
        noun_entry = next(c for c in comp if c.tag == "NOUN")
        assert noun_entry.count == 3

    def test_percentages_sum(self):
        df = pd.DataFrame({
            "POS_Tags": [["NOUN", "VERB", "NOUN", "VERB"]],
        })
        comp = self.ape._compute_pos_composition(df)
        total = sum(c.percentage for c in comp)
        assert abs(total - 100.0) < 0.1

    def test_empty_df(self):
        df = pd.DataFrame({"POS_Tags": []})
        assert self.ape._compute_pos_composition(df) == []

    def test_non_list_tags_skipped(self):
        df = pd.DataFrame({"POS_Tags": ["not_a_list", ["NOUN"]]})
        comp = self.ape._compute_pos_composition(df)
        assert len(comp) == 1
        assert comp[0].tag == "NOUN"


# ── _compute_structural_similarity ───────────────────────────────────────────

class TestComputeStructuralSimilarity:
    def test_shape(self):
        df = pd.DataFrame({
            "POS_Tags": [["NOUN", "VERB"], ["ADJ", "NOUN"], ["NOUN", "VERB"]],
        })
        mat = AdaptiveEvolutionAnalyzer._compute_structural_similarity(df)
        assert mat.shape == (3, 3)

    def test_diagonal_is_one(self):
        df = pd.DataFrame({"POS_Tags": [["NOUN"], ["VERB"]]})
        mat = AdaptiveEvolutionAnalyzer._compute_structural_similarity(df)
        assert mat[0, 0] == 1.0
        assert mat[1, 1] == 1.0

    def test_symmetric(self):
        df = pd.DataFrame({
            "POS_Tags": [
                ["NOUN", "VERB", "NOUN"],
                ["ADJ", "DET", "NOUN"],
                ["VERB", "NOUN", "ADJ"],
            ],
        })
        mat = AdaptiveEvolutionAnalyzer._compute_structural_similarity(df)
        np.testing.assert_array_almost_equal(mat, mat.T)

    def test_identical_high_similarity(self):
        same = ["NOUN", "VERB", "DET", "NOUN"] * 10
        df = pd.DataFrame({"POS_Tags": [same, same]})
        mat = AdaptiveEvolutionAnalyzer._compute_structural_similarity(df)
        assert mat[0, 1] > 0.7

    def test_bounded(self):
        df = pd.DataFrame({
            "POS_Tags": [["NOUN"] * 20, ["VERB"] * 20, ["ADJ", "NOUN"] * 10],
        })
        mat = AdaptiveEvolutionAnalyzer._compute_structural_similarity(df)
        assert np.all(mat >= 0.0)
        assert np.all(mat <= 1.0)


# ── _compute_evolutionary_dynamics ───────────────────────────────────────────

class TestComputeEvolutionaryDynamics:
    def setup_method(self):
        self.ape = AdaptiveEvolutionAnalyzer(status_threshold=0.02)

    def test_basic_two_docs(self):
        df, doc_order = _make_df(n_per_doc=10, n_docs=2)
        species = self.ape._compute_evolutionary_dynamics(df, doc_order)
        assert isinstance(species, list)
        assert all(isinstance(s, SpeciesRecord) for s in species)
        assert len(species) == df["Group_ID"].nunique()

    def test_sorted_by_density_end(self):
        df, doc_order = _make_df(n_per_doc=20, n_docs=2)
        species = self.ape._compute_evolutionary_dynamics(df, doc_order)
        densities = [s.density_end for s in species]
        assert densities == sorted(densities, reverse=True)

    def test_densities_sum_to_one(self):
        df, doc_order = _make_df(n_per_doc=20, n_docs=2)
        species = self.ape._compute_evolutionary_dynamics(df, doc_order)
        total_start = sum(s.density_start for s in species)
        total_end = sum(s.density_end for s in species)
        assert abs(total_start - 1.0) < 0.001
        assert abs(total_end - 1.0) < 0.001

    def test_emerging_species(self):
        """Species present only in doc_1 should be Emerging."""
        rows = [
            {"Sentence": "A.", "POS_Tags": ["DET"], "Source_Document": "doc_0", "Group_ID": 0},
            {"Sentence": "B.", "POS_Tags": ["NOUN"], "Source_Document": "doc_0", "Group_ID": 0},
            {"Sentence": "C.", "POS_Tags": ["VERB"], "Source_Document": "doc_1", "Group_ID": 0},
            {"Sentence": "D.", "POS_Tags": ["ADJ"], "Source_Document": "doc_1", "Group_ID": 1},
        ]
        df = pd.DataFrame(rows)
        species = self.ape._compute_evolutionary_dynamics(df, ["doc_0", "doc_1"])
        statuses = {s.cluster_id: s.status for s in species}
        assert statuses[1] == EvolutionaryStatus.EMERGING

    def test_extinct_species(self):
        """Species present only in doc_0 should be Extinct."""
        rows = [
            {"Sentence": "A.", "POS_Tags": ["DET"], "Source_Document": "doc_0", "Group_ID": 0},
            {"Sentence": "B.", "POS_Tags": ["NOUN"], "Source_Document": "doc_0", "Group_ID": 1},
            {"Sentence": "C.", "POS_Tags": ["VERB"], "Source_Document": "doc_1", "Group_ID": 0},
            {"Sentence": "D.", "POS_Tags": ["ADJ"], "Source_Document": "doc_1", "Group_ID": 0},
        ]
        df = pd.DataFrame(rows)
        species = self.ape._compute_evolutionary_dynamics(df, ["doc_0", "doc_1"])
        statuses = {s.cluster_id: s.status for s in species}
        assert statuses[1] == EvolutionaryStatus.EXTINCT

    def test_no_group_id_returns_empty(self):
        df = pd.DataFrame({"Sentence": ["A"], "Source_Document": ["d0"]})
        species = self.ape._compute_evolutionary_dynamics(df, ["d0"])
        assert species == []

    def test_n_docs(self):
        """With 3 docs, compares first and last."""
        df, doc_order = _make_df(n_per_doc=10, n_docs=3)
        species = self.ape._compute_evolutionary_dynamics(df, doc_order)
        assert len(species) > 0

    def test_sample_sentence_populated(self):
        df, doc_order = _make_df(n_per_doc=5, n_docs=2)
        species = self.ape._compute_evolutionary_dynamics(df, doc_order)
        for s in species:
            assert len(s.sample_sentence) > 0

    def test_pos_composition_populated(self):
        df, doc_order = _make_df(n_per_doc=5, n_docs=2)
        species = self.ape._compute_evolutionary_dynamics(df, doc_order)
        for s in species:
            assert isinstance(s.pos_composition, list)
            if s.pos_composition:
                assert isinstance(s.pos_composition[0], POSComposition)


# ── prepare_data_from_parsed ─────────────────────────────────────────────────

class TestPrepareDataFromParsed:
    def test_basic(self):
        ape = AdaptiveEvolutionAnalyzer()
        data = ape.prepare_data_from_parsed({
            "doc_a": [
                ParsedSentence("Hello world.", ["NOUN", "NOUN", "PUNCT"]),
                ParsedSentence("Goodbye.", ["NOUN", "PUNCT"]),
            ],
            "doc_b": [
                ParsedSentence("Test sentence.", ["NOUN", "NOUN", "PUNCT"]),
            ],
        })
        assert len(data["df"]) == 3
        assert data["doc_order"] == ["doc_a", "doc_b"]
        assert "POS_Tags" in data["df"].columns

    def test_custom_order(self):
        ape = AdaptiveEvolutionAnalyzer()
        data = ape.prepare_data_from_parsed(
            {"b": [], "a": []},
            doc_order=["a", "b"],
        )
        assert data["doc_order"] == ["a", "b"]


# ── EvolutionResult integration ──────────────────────────────────────────────

class TestEvolutionResultIntegration:
    def test_to_dataframe(self):
        result = EvolutionResult(species=[
            SpeciesRecord(0, EvolutionaryStatus.THRIVING, 0.1, 0.3, 0.2, "Cat sat."),
            SpeciesRecord(1, EvolutionaryStatus.EXTINCT, 0.2, 0.0, -0.2, "Dog ran."),
        ])
        df = result.to_dataframe()
        assert len(df) == 2
        assert df.iloc[0]["status"] == "Thriving"

    def test_to_json(self):
        result = EvolutionResult(species=[
            SpeciesRecord(0, EvolutionaryStatus.EMERGING, 0.0, 0.3, 0.3, "New.",
                          pos_composition=[POSComposition("NOUN", 50.0, 10)]),
        ])
        j = result.to_json()
        assert j[0]["classification"]["status"] == "Emerging"
        assert j[0]["classification"]["trend"] == "Positive"
        assert len(j[0]["pos_composition"]) == 1

    def test_empty(self):
        assert len(EvolutionResult().to_dataframe()) == 0
