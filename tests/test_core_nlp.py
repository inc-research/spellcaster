"""Tests for spellcaster.core.nlp."""

import pytest

from spellcaster.core.nlp import clear_model_cache, get_nlp, tokenize
from spellcaster.exceptions import ModelNotLoadedError


# ── get_nlp ──────────────────────────────────────────────────────────────────

class TestGetNlp:
    def setup_method(self):
        clear_model_cache()

    def test_loads_default_model(self):
        nlp = get_nlp()
        assert nlp is not None
        assert hasattr(nlp, "pipe_names")

    def test_caches_same_config(self):
        nlp1 = get_nlp("en_core_web_sm", disable=["ner"])
        nlp2 = get_nlp("en_core_web_sm", disable=["ner"])
        assert nlp1 is nlp2

    def test_different_disable_different_cache(self):
        nlp1 = get_nlp("en_core_web_sm", disable=[])
        nlp2 = get_nlp("en_core_web_sm", disable=["parser", "ner"])
        assert nlp1 is not nlp2

    def test_invalid_model_raises(self):
        with pytest.raises(ModelNotLoadedError, match="not installed"):
            get_nlp("nonexistent_model_xyz")

    def test_clear_cache(self):
        nlp1 = get_nlp()
        clear_model_cache()
        nlp2 = get_nlp()
        assert nlp1 is not nlp2


# ── tokenize ─────────────────────────────────────────────────────────────────

class TestTokenize:
    def test_basic_tokenization(self):
        tokens = tokenize("The quick brown fox jumps over the lazy dog.")
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        # All tokens should be lowercase strings
        for t in tokens:
            assert isinstance(t, str)
            assert t == t.lower()

    def test_removes_stop_words(self):
        tokens = tokenize("The the the is is are was were a an.")
        # Most/all of these are stop words — list should be short or empty
        assert len(tokens) < 5

    def test_only_alphabetic(self):
        tokens = tokenize("Hello world! 42 items cost $3.50 each.")
        for t in tokens:
            assert t.isalpha(), f"Non-alphabetic token: {t!r}"

    def test_empty_string(self):
        assert tokenize("") == []

    def test_accepts_custom_nlp(self):
        nlp = get_nlp("en_core_web_sm", disable=["parser", "ner"])
        tokens = tokenize("Testing custom model injection.", nlp=nlp)
        assert len(tokens) > 0

    def test_returns_lemmas(self):
        tokens = tokenize("The dogs were running quickly through the forests.")
        # "dogs" → "dog", "running" → "run", "forests" → "forest"
        assert "dog" in tokens or "run" in tokens  # At least one lemmatization

    def test_deterministic(self):
        text = "Complex systems exhibit emergent properties."
        t1 = tokenize(text)
        t2 = tokenize(text)
        assert t1 == t2
