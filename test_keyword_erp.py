"""Tests for spellcaster.analyzers.keyword_erp."""

from __future__ import annotations

import numpy as np
import pandas as pd

from spellcaster.analyzers.keyword_erp import (
    KeywordERPAnalyzer,
    _build_pos_cooccurrence,
    _spectral_entropy,
)
from spellcaster.results.keyword import (
    CrossKeywordEntanglement,
    FileKeywordResult,
    KeywordERPResult,
    KeywordMeasures,
)


# ── _build_pos_cooccurrence ──────────────────────────────────────────────────

class TestBuildPOSCooccurrence:
    def test_shape(self):
        vocab = ["NOUN", "VERB", "DET"]
        seqs = [["DET", "NOUN", "VERB"]]
        mat = _build_pos_cooccurrence(seqs, vocab, window=2)
        assert mat.shape == (3, 3)

    def test_symmetric(self):
        vocab = ["NOUN", "VERB", "DET", "ADJ"]
        seqs = [["DET", "NOUN", "VERB", "ADJ", "NOUN"]]
        mat = _build_pos_cooccurrence(seqs, vocab, window=2)
        np.testing.assert_array_almost_equal(mat, mat.T)

    def test_normalised(self):
        vocab = ["NOUN", "VERB"]
        seqs = [["NOUN", "VERB", "NOUN", "VERB"]]
        mat = _build_pos_cooccurrence(seqs, vocab, window=1)
        assert abs(mat.sum() - 1.0) < 1e-10

    def test_empty_sequences(self):
        vocab = ["NOUN", "VERB"]
        mat = _build_pos_cooccurrence([], vocab, window=2)
        assert mat.sum() == 0.0

    def test_unknown_tags_ignored(self):
        vocab = ["NOUN", "VERB"]
        seqs = [["NOUN", "UNKNOWN", "VERB"]]
        mat = _build_pos_cooccurrence(seqs, vocab, window=2)
        # Should still produce a valid matrix
        assert mat.shape == (2, 2)
        assert mat.sum() > 0

    def test_positive_diagonal(self):
        """Diagonal should contain marginal tag frequencies."""
        vocab = ["NOUN", "VERB"]
        seqs = [["NOUN", "VERB", "NOUN"]]
        mat = _build_pos_cooccurrence(seqs, vocab, window=1)
        assert mat[0, 0] > 0  # NOUN appears twice
        assert mat[1, 1] > 0  # VERB appears once

    def test_window_size_affects_density(self):
        vocab = ["A", "B", "C", "D"]
        seqs = [["A", "B", "C", "D"]]
        mat_narrow = _build_pos_cooccurrence(seqs, vocab, window=1)
        mat_wide = _build_pos_cooccurrence(seqs, vocab, window=3)
        # Wider window should capture more off-diagonal co-occurrences
        # Both are normalised, but off-diag proportion should differ
        off_diag_narrow = mat_narrow.sum() - np.trace(mat_narrow)
        off_diag_wide = mat_wide.sum() - np.trace(mat_wide)
        assert off_diag_wide >= off_diag_narrow


# ── _spectral_entropy ────────────────────────────────────────────────────────

class TestSpectralEntropy:
    def test_zero_matrix(self):
        assert _spectral_entropy(np.zeros((4, 4))) == 0.0

    def test_uniform(self):
        """Uniform diagonal → maximum entropy."""
        n = 4
        mat = np.eye(n) / n
        h = _spectral_entropy(mat)
        assert abs(h - 2.0) < 0.01  # log2(4) = 2.0

    def test_concentrated(self):
        """Single non-zero eigenvalue → zero entropy."""
        mat = np.zeros((4, 4))
        mat[0, 0] = 1.0
        h = _spectral_entropy(mat)
        assert abs(h) < 0.01

    def test_positive(self):
        """A real cooccurrence matrix should have positive entropy."""
        vocab = ["NOUN", "VERB", "DET", "ADJ"]
        seqs = [["DET", "NOUN", "VERB", "DET", "NOUN"], ["ADJ", "NOUN", "VERB"]]
        mat = _build_pos_cooccurrence(seqs, vocab, window=2)
        h = _spectral_entropy(mat)
        assert h > 0

    def test_more_diverse_higher_entropy(self):
        """More diverse POS distribution should yield higher entropy."""
        vocab = ["A", "B", "C", "D"]
        # Concentrated: mostly A
        seqs_conc = [["A", "A", "A", "A", "B"]]
        # Diverse: even mix
        seqs_div = [["A", "B", "C", "D", "A", "B", "C", "D"]]
        h_conc = _spectral_entropy(_build_pos_cooccurrence(seqs_conc, vocab, 1))
        h_div = _spectral_entropy(_build_pos_cooccurrence(seqs_div, vocab, 1))
        assert h_div > h_conc


# ── _build_vocab ─────────────────────────────────────────────────────────────

class TestBuildVocab:
    def test_basic(self):
        windows = {
            "info": [{"pos_sequences": [["NOUN", "VERB"], ["DET", "NOUN"]]}],
            "net": [{"pos_sequences": [["ADJ", "NOUN"]]}],
        }
        vocab = KeywordERPAnalyzer._build_vocab(windows)
        assert isinstance(vocab, list)
        assert vocab == sorted(vocab)  # Sorted
        assert set(vocab) == {"NOUN", "VERB", "DET", "ADJ"}

    def test_empty(self):
        assert KeywordERPAnalyzer._build_vocab({}) == []

    def test_no_duplicates(self):
        windows = {
            "kw": [
                {"pos_sequences": [["NOUN", "NOUN", "VERB"]]},
                {"pos_sequences": [["NOUN", "VERB", "VERB"]]},
            ],
        }
        vocab = KeywordERPAnalyzer._build_vocab(windows)
        assert len(vocab) == len(set(vocab))


# ── _within_keyword_measures ─────────────────────────────────────────────────

class TestWithinKeywordMeasures:
    def setup_method(self):
        self.analyzer = KeywordERPAnalyzer(
            keywords=["test"],
            epr_coherence_threshold=0.5,
            min_epr_distance=2,
        )
        self.vocab = ["NOUN", "VERB", "DET", "ADJ", "ADP", "PUNCT"]

    def _make_windows(self, n_mentions: int, vary: bool = False) -> list[dict]:
        base = ["DET", "NOUN", "VERB", "DET", "NOUN", "PUNCT"]
        alt = ["ADJ", "NOUN", "ADP", "NOUN", "VERB", "PUNCT"]
        windows = []
        for i in range(n_mentions):
            seq = alt if (vary and i % 2 == 1) else base
            windows.append({
                "mention_idx": i,
                "sent_idx": i * 10,
                "pos_sequences": [seq],
            })
        return windows

    def test_no_mentions(self):
        matrices = {}
        km = self.analyzer._within_keyword_measures("test", [], matrices, self.vocab)
        assert km.n_mentions == 0
        assert km.mean_entropy is None
        assert km.n_epr_pairs == 0

    def test_single_mention(self):
        windows = self._make_windows(1)
        matrices = {("test", 0): _build_pos_cooccurrence(windows[0]["pos_sequences"], self.vocab, 2)}
        km = self.analyzer._within_keyword_measures("test", windows, matrices, self.vocab)
        assert km.n_mentions == 1
        assert km.mean_entropy is not None
        assert km.entropy_trend is None  # Need >=3 for trend
        assert km.mean_coherence is None  # Need >=2 for coherence

    def test_multiple_mentions(self):
        windows = self._make_windows(5)
        matrices = {}
        for w in windows:
            matrices[("test", w["mention_idx"])] = _build_pos_cooccurrence(
                w["pos_sequences"], self.vocab, 2
            )
        km = self.analyzer._within_keyword_measures("test", windows, matrices, self.vocab)
        assert km.n_mentions == 5
        assert km.mean_entropy is not None and km.mean_entropy > 0
        assert km.entropy_trend is not None
        assert km.mean_coherence is not None

    def test_identical_windows_high_coherence(self):
        windows = self._make_windows(4, vary=False)
        matrices = {}
        for w in windows:
            matrices[("test", w["mention_idx"])] = _build_pos_cooccurrence(
                w["pos_sequences"], self.vocab, 2
            )
        km = self.analyzer._within_keyword_measures("test", windows, matrices, self.vocab)
        # Identical windows → high coherence
        if km.mean_coherence is not None:
            assert km.mean_coherence > 0.5

    def test_epr_pairs_detected(self):
        """Identical windows far apart should produce EPR pairs."""
        windows = self._make_windows(6, vary=False)
        matrices = {}
        for w in windows:
            matrices[("test", w["mention_idx"])] = _build_pos_cooccurrence(
                w["pos_sequences"], self.vocab, 2
            )
        km = self.analyzer._within_keyword_measures("test", windows, matrices, self.vocab)
        # min_epr_distance=2, so pairs like (0,2), (0,3), etc. qualify
        assert km.n_epr_pairs > 0
        assert km.mean_epr_strength is not None
        assert km.max_epr_strength is not None


# ── _cross_keyword ───────────────────────────────────────────────────────────

class TestCrossKeyword:
    def setup_method(self):
        self.analyzer = KeywordERPAnalyzer(
            keywords=["alpha", "beta"],
            epr_coherence_threshold=0.5,
        )
        self.vocab = ["NOUN", "VERB", "DET", "ADJ", "PUNCT"]

    def test_empty_windows(self):
        ck = self.analyzer._cross_keyword(
            "alpha", "beta", {"alpha": [], "beta": []}, {}, self.vocab
        )
        assert isinstance(ck, CrossKeywordEntanglement)
        assert ck.mean_cross_coherence is None
        assert ck.n_cross_epr_pairs == 0

    def test_basic(self):
        wins_a = [{"mention_idx": 0, "sent_idx": 0,
                    "pos_sequences": [["DET", "NOUN", "VERB", "PUNCT"]]}]
        wins_b = [{"mention_idx": 0, "sent_idx": 5,
                    "pos_sequences": [["DET", "NOUN", "VERB", "PUNCT"]]}]
        windows = {"alpha": wins_a, "beta": wins_b}
        matrices = {
            ("alpha", 0): _build_pos_cooccurrence(wins_a[0]["pos_sequences"], self.vocab, 2),
            ("beta", 0): _build_pos_cooccurrence(wins_b[0]["pos_sequences"], self.vocab, 2),
        }
        ck = self.analyzer._cross_keyword("alpha", "beta", windows, matrices, self.vocab)
        assert ck.mean_cross_coherence is not None
        # Identical POS sequences → high similarity
        assert ck.mean_cross_coherence > 0.5


# ── KeywordERPResult integration ─────────────────────────────────────────────

class TestKeywordERPResultIntegration:
    def test_to_dataframe(self):
        result = KeywordERPResult(
            keywords=["info", "net"],
            files=[
                FileKeywordResult(
                    file="a.txt", label="A",
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
        df = result.to_dataframe()
        assert len(df) == 2
        assert set(df["keyword"]) == {"info", "net"}

        xdf = result.cross_keyword_dataframe()
        assert len(xdf) == 1
        assert xdf.iloc[0]["keyword_a"] == "info"

    def test_empty(self):
        result = KeywordERPResult()
        assert len(result.to_dataframe()) == 0
        assert len(result.cross_keyword_dataframe()) == 0
