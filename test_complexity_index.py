"""Tests for spellcaster.analyzers.complexity_index."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd

from spellcaster.analyzers.complexity_index import (
    TextComplexityAnalyzer,
    _levenshtein_distance,
)
from spellcaster.io.readers import TextDocument
from spellcaster.results.complexity import (
    ComplexityComparisonResult,
    ComplexityFlowResult,
    SentenceMetrics,
)


# ── Levenshtein fallback ─────────────────────────────────────────────────────

class TestLevenshteinDistance:
    def test_identical(self):
        assert _levenshtein_distance("hello", "hello") == 0

    def test_empty_first(self):
        assert _levenshtein_distance("", "abc") == 3

    def test_empty_second(self):
        assert _levenshtein_distance("abc", "") == 3

    def test_both_empty(self):
        assert _levenshtein_distance("", "") == 0

    def test_single_substitution(self):
        assert _levenshtein_distance("cat", "bat") == 1

    def test_single_insertion(self):
        assert _levenshtein_distance("cat", "cats") == 1

    def test_single_deletion(self):
        assert _levenshtein_distance("cats", "cat") == 1

    def test_completely_different(self):
        assert _levenshtein_distance("abc", "xyz") == 3

    def test_symmetric(self):
        assert _levenshtein_distance("kitten", "sitting") == _levenshtein_distance("sitting", "kitten")

    def test_known_value(self):
        # "kitten" → "sitting" = 3 (substitute k→s, e→i, insert g)
        assert _levenshtein_distance("kitten", "sitting") == 3


# ── analyze_flow ─────────────────────────────────────────────────────────────

class TestAnalyzeFlow:
    def setup_method(self):
        self.analyzer = TextComplexityAnalyzer()

    def test_basic_result_type(self):
        result = self.analyzer.analyze_flow("Hello world. How are you.")
        assert isinstance(result, ComplexityFlowResult)

    def test_sentence_count(self):
        text = "First sentence. Second sentence. Third sentence."
        result = self.analyzer.analyze_flow(text)
        assert len(result.sentences) == 3

    def test_label_preserved(self):
        result = self.analyzer.analyze_flow("Hello.", label="test_label")
        assert result.label == "test_label"

    def test_empty_text(self):
        result = self.analyzer.analyze_flow("")
        assert len(result.sentences) == 0

    def test_k_hist_monotonically_increasing(self):
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "A completely different sentence about quantum physics. "
            "Yet another topic about Renaissance art and its influence."
        )
        result = self.analyzer.analyze_flow(text)
        k_hist = [s.k_hist for s in result.sentences]
        for i in range(1, len(k_hist)):
            assert k_hist[i] >= k_hist[i - 1], (
                f"k_hist not monotonic: {k_hist[i]} < {k_hist[i-1]} at index {i}"
            )

    def test_first_sentence_zero_volatility(self):
        result = self.analyzer.analyze_flow("First. Second. Third.")
        assert result.sentences[0].volatility == 0

    def test_subsequent_volatility_positive(self):
        # These sentences are quite different, so volatility should be > 0
        text = "The cat sat on the mat. A rocket launched into deep space."
        result = self.analyzer.analyze_flow(text)
        if len(result.sentences) > 1:
            assert result.sentences[1].volatility > 0

    def test_synergy_non_negative(self):
        text = "Hello world. Goodbye world. Another sentence here."
        result = self.analyzer.analyze_flow(text)
        for s in result.sentences:
            assert s.synergy >= 0, f"Negative synergy: {s.synergy}"

    def test_sentence_indices_sequential(self):
        text = "One. Two. Three. Four."
        result = self.analyzer.analyze_flow(text)
        indices = [s.index for s in result.sentences]
        assert indices == list(range(len(indices)))

    def test_sentence_text_preserved(self):
        text = "Alpha sentence. Beta sentence."
        result = self.analyzer.analyze_flow(text)
        assert result.sentences[0].text == "Alpha sentence"
        assert result.sentences[1].text == "Beta sentence"

    def test_independent_instances(self):
        """Two analyzer instances should produce identical results."""
        text = "The fox. The dog."
        a1 = TextComplexityAnalyzer()
        a2 = TextComplexityAnalyzer()
        r1 = a1.analyze_flow(text, label="A")
        r2 = a2.analyze_flow(text, label="B")
        assert len(r1.sentences) == len(r2.sentences)
        for s1, s2 in zip(r1.sentences, r2.sentences):
            assert s1.k_hist == s2.k_hist
            assert s1.volatility == s2.volatility
            assert s1.synergy == s2.synergy

    def test_html_stripped(self):
        text = "<p>Hello world.</p> <b>Bold text.</b>"
        result = self.analyzer.analyze_flow(text)
        for s in result.sentences:
            assert "<" not in s.text

    def test_to_dataframe(self):
        result = self.analyzer.analyze_flow(
            "One. Two. Three.", label="test"
        )
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert set(df.columns) == {"index", "text", "k_hist", "volatility", "synergy"}

    def test_array_properties(self):
        result = self.analyzer.analyze_flow("A. B. C.")
        assert isinstance(result.k_hist_array, np.ndarray)
        assert len(result.k_hist_array) == 3
        assert len(result.volatility_array) == 3
        assert len(result.synergy_array) == 3


# ── compare ──────────────────────────────────────────────────────────────────

class TestCompare:
    def setup_method(self):
        self.analyzer = TextComplexityAnalyzer()

    def test_from_strings(self):
        result = self.analyzer.compare(
            ["First text. With sentences.", "Second text. Also sentences."],
            labels=["A", "B"],
            from_files=False,
        )
        assert isinstance(result, ComplexityComparisonResult)
        assert len(result.flows) == 2
        assert result.labels == ["A", "B"]

    def test_from_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = os.path.join(tmpdir, "doc_a.txt")
            p2 = os.path.join(tmpdir, "doc_b.txt")
            with open(p1, "w") as f:
                f.write("The quick brown fox. Jumps over the lazy dog.")
            with open(p2, "w") as f:
                f.write("Information flows through networks. Data moves fast.")

            result = self.analyzer.compare([p1, p2])
            assert len(result.flows) == 2
            assert result.flows[0].label == "doc_a"
            assert result.flows[1].label == "doc_b"

    def test_n_texts(self):
        """Generalizes beyond 2 texts."""
        texts = [
            "Alpha. Beta. Gamma.",
            "Delta. Epsilon. Zeta.",
            "Eta. Theta. Iota.",
        ]
        result = self.analyzer.compare(texts, from_files=False)
        assert len(result.flows) == 3

    def test_single_text(self):
        result = self.analyzer.compare(
            ["Just one document. With content."],
            from_files=False,
        )
        assert len(result.flows) == 1

    def test_default_labels_from_strings(self):
        result = self.analyzer.compare(
            ["Text A.", "Text B."],
            from_files=False,
        )
        assert result.labels == ["text_0", "text_1"]

    def test_combined_dataframe(self):
        result = self.analyzer.compare(
            ["One. Two.", "Three. Four."],
            labels=["X", "Y"],
            from_files=False,
        )
        df = result.to_dataframe()
        assert "label" in df.columns
        assert set(df["label"]) == {"X", "Y"}
        assert len(df) == 4  # 2 sentences × 2 texts


# ── analyze_document ─────────────────────────────────────────────────────────

class TestAnalyzeDocument:
    def test_basic(self):
        doc = TextDocument(path="test.txt", label="MyDoc", text="Hello. World.")
        analyzer = TextComplexityAnalyzer()
        result = analyzer.analyze_document(doc)
        assert result.label == "MyDoc"
        assert len(result.sentences) == 2


# ── Behavioral properties ────────────────────────────────────────────────────

class TestBehavioralProperties:
    """
    Higher-level tests verifying that the analyzer's outputs have
    meaningful properties consistent with the intended interpretation.
    """

    def setup_method(self):
        self.analyzer = TextComplexityAnalyzer()

    def test_repetitive_text_lower_k_growth(self):
        """Repetitive text should grow k_hist more slowly than diverse text."""
        repetitive = "The cat sat. " * 20
        diverse = ". ".join(
            f"Sentence {i} about topic {chr(65 + i % 26)}" for i in range(20)
        ) + "."

        r_rep = self.analyzer.analyze_flow(repetitive, "repetitive")
        r_div = self.analyzer.analyze_flow(diverse, "diverse")

        # Compare final k_hist normalised by sentence count
        if r_rep.sentences and r_div.sentences:
            growth_rep = r_rep.sentences[-1].k_hist / len(r_rep.sentences)
            growth_div = r_div.sentences[-1].k_hist / len(r_div.sentences)
            assert growth_rep < growth_div, (
                f"Repetitive growth ({growth_rep:.1f}) >= diverse ({growth_div:.1f})"
            )

    def test_similar_sentences_low_volatility(self):
        """Near-identical sentences should have low volatility."""
        text = "The cat sat on the mat. The cat sat on the rug."
        result = self.analyzer.analyze_flow(text)
        if len(result.sentences) > 1:
            vol = result.sentences[1].volatility
            # "mat" → "rug" is a small edit at the end
            assert vol < 10, f"Expected low volatility, got {vol}"

    def test_very_different_sentences_high_volatility(self):
        """Completely different sentences should have high volatility."""
        text = (
            "The quick brown fox jumps over the lazy sleeping dog in the meadow. "
            "Quantum entanglement governs teleportation of information across spacetime."
        )
        result = self.analyzer.analyze_flow(text)
        if len(result.sentences) > 1:
            vol = result.sentences[1].volatility
            assert vol > 20, f"Expected high volatility, got {vol}"
