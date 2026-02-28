"""Tests for spellcaster.extractors (action_frames, noun_dependencies, sentence_parser)."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# spaCy-free tests (no NLP model required)
# ---------------------------------------------------------------------------

from spellcaster.extractors.sentence_parser import (
    DEFAULT_ABBREVIATIONS,
    split_sentences_simple,
)
from spellcaster.extractors.action_frames import make_hashable_frame


class TestSplitSentencesSimple:
    def test_basic(self):
        text = "Hello world. How are you. Fine thanks."
        result = split_sentences_simple(text)
        assert len(result) == 3
        assert result[0] == "Hello world"

    def test_newlines(self):
        text = "Line one\nLine two\nLine three"
        result = split_sentences_simple(text)
        assert len(result) == 3

    def test_bullet_points(self):
        text = "Item one•Item two•Item three"
        result = split_sentences_simple(text)
        assert len(result) == 3

    def test_strips_html(self):
        text = "<p>Hello world.</p> <b>Bold text.</b>"
        result = split_sentences_simple(text)
        assert all("<" not in s for s in result)

    def test_empty_string(self):
        assert split_sentences_simple("") == []

    def test_whitespace_only(self):
        assert split_sentences_simple("   \n  ") == []

    def test_no_delimiters(self):
        result = split_sentences_simple("Just one sentence without ending punctuation")
        assert len(result) == 1

    def test_consecutive_periods(self):
        result = split_sentences_simple("Hello... World")
        # "..." splits into multiple empty strings, filtered out
        non_empty = [s for s in result if s]
        assert len(non_empty) >= 1


class TestMakeHashableFrame:
    def test_basic(self):
        frame = {
            "verb": "chase",
            "subjects": ["cat"],
            "objects": ["mouse"],
            "other_deps": [("advmod", "quickly")],
        }
        h = make_hashable_frame(frame)
        assert isinstance(h, tuple)
        assert h[0] == "chase"
        # Should be hashable
        assert hash(h) is not None

    def test_sorted_deterministic(self):
        frame1 = {
            "verb": "eat",
            "subjects": ["b", "a"],
            "objects": ["d", "c"],
            "other_deps": [("x", "z"), ("x", "y")],
        }
        frame2 = {
            "verb": "eat",
            "subjects": ["a", "b"],
            "objects": ["c", "d"],
            "other_deps": [("x", "y"), ("x", "z")],
        }
        assert make_hashable_frame(frame1) == make_hashable_frame(frame2)

    def test_different_verbs_differ(self):
        f1 = {"verb": "run", "subjects": [], "objects": [], "other_deps": []}
        f2 = {"verb": "walk", "subjects": [], "objects": [], "other_deps": []}
        assert make_hashable_frame(f1) != make_hashable_frame(f2)

    def test_countable(self):
        from collections import Counter
        frame = {"verb": "go", "subjects": [], "objects": [], "other_deps": []}
        h = make_hashable_frame(frame)
        c = Counter([h, h, h])
        assert c[h] == 3


class TestDefaultAbbreviations:
    def test_is_frozenset(self):
        assert isinstance(DEFAULT_ABBREVIATIONS, frozenset)

    def test_contains_common(self):
        assert "dr." in DEFAULT_ABBREVIATIONS
        assert "mr." in DEFAULT_ABBREVIATIONS
        assert "e.g." in DEFAULT_ABBREVIATIONS
