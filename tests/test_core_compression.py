"""Tests for spellcaster.core.compression."""

import pytest

from spellcaster.core.compression import (
    compressed_size,
    ncd_similarity,
    normalized_compression_distance,
)


# ── compressed_size ──────────────────────────────────────────────────────────

class TestCompressedSize:
    def test_empty_string(self):
        assert compressed_size("") == 0

    def test_returns_positive_for_nonempty(self):
        assert compressed_size("hello world") > 0

    def test_repetitive_smaller_than_random(self):
        repetitive = "abcabc" * 100
        diverse = "".join(chr(65 + (i * 7 + 3) % 26) for i in range(600))
        assert compressed_size(repetitive) < compressed_size(diverse)

    def test_longer_text_generally_larger(self):
        short = "The quick brown fox."
        long = short * 10
        # Compression + overhead means longer text compressed is >= short compressed
        assert compressed_size(long) >= compressed_size(short)


# ── normalized_compression_distance ──────────────────────────────────────────

class TestNCD:
    def test_identical_sequences(self):
        seq = ["NOUN", "VERB", "NOUN", "ADJ"]
        ncd = normalized_compression_distance(seq, seq)
        # Identical sequences should have very low NCD
        assert ncd < 0.3

    def test_empty_sequences(self):
        assert normalized_compression_distance([], []) == 0.0

    def test_bounded_zero_one(self):
        s1 = ["NOUN", "VERB"] * 20
        s2 = ["ADJ", "ADV", "DET"] * 15
        ncd = normalized_compression_distance(s1, s2)
        assert 0.0 <= ncd <= 1.0

    def test_similar_lower_than_dissimilar(self):
        base = ["NOUN", "VERB", "NOUN", "ADJ", "NOUN"]
        similar = ["NOUN", "VERB", "NOUN", "ADJ", "DET"]
        different = ["ADV", "CONJ", "INTJ", "PART", "SCONJ"]

        ncd_similar = normalized_compression_distance(base, similar)
        ncd_different = normalized_compression_distance(base, different)
        # Similar sequences should have lower NCD
        assert ncd_similar <= ncd_different

    def test_symmetry(self):
        s1 = ["NOUN", "VERB", "ADJ"]
        s2 = ["DET", "NOUN", "PUNCT"]
        assert normalized_compression_distance(s1, s2) == pytest.approx(
            normalized_compression_distance(s2, s1), abs=1e-10
        )


# ── ncd_similarity ───────────────────────────────────────────────────────────

class TestNCDSimilarity:
    def test_identical_high_similarity(self):
        seq = ["NOUN", "VERB"] * 20
        assert ncd_similarity(seq, seq) > 0.7

    def test_inverse_of_ncd(self):
        s1 = ["NOUN", "VERB"] * 10
        s2 = ["ADJ", "ADV"] * 10
        ncd = normalized_compression_distance(s1, s2)
        sim = ncd_similarity(s1, s2)
        assert sim == pytest.approx(1.0 - ncd, abs=1e-10)

    def test_bounded_zero_one(self):
        s1 = ["A"] * 50
        s2 = ["B"] * 50
        sim = ncd_similarity(s1, s2)
        assert 0.0 <= sim <= 1.0
