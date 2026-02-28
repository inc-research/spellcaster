"""Tests for spellcaster.io.readers."""

from __future__ import annotations

import tempfile
import os

from spellcaster.io.readers import TextDocument, load_texts, texts_from_strings


class TestTextsFromStrings:
    def test_basic(self):
        docs = texts_from_strings(["Hello world", "Goodbye world"])
        assert len(docs) == 2
        assert docs[0].text == "Hello world"
        assert docs[0].path == "<inline>"
        assert docs[0].label == "text_0"
        assert docs[1].label == "text_1"

    def test_custom_labels(self):
        docs = texts_from_strings(["A", "B"], labels=["Alpha", "Beta"])
        assert docs[0].label == "Alpha"
        assert docs[1].label == "Beta"

    def test_label_mismatch_raises(self):
        try:
            texts_from_strings(["A", "B"], labels=["only_one"])
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "length" in str(e).lower()

    def test_empty(self):
        assert texts_from_strings([]) == []


class TestLoadTexts:
    def test_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = os.path.join(tmpdir, "doc1.txt")
            path2 = os.path.join(tmpdir, "doc2.txt")
            with open(path1, "w") as f:
                f.write("Content of doc one.")
            with open(path2, "w") as f:
                f.write("Content of doc two.")

            docs = load_texts([path1, path2])
            assert len(docs) == 2
            assert docs[0].text == "Content of doc one."
            assert docs[0].label == "doc1"  # stem
            assert docs[1].label == "doc2"

    def test_custom_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.txt")
            with open(path, "w") as f:
                f.write("hello")

            docs = load_texts([path], labels=["MyDoc"])
            assert docs[0].label == "MyDoc"

    def test_file_not_found(self):
        try:
            load_texts(["/nonexistent/path/xyz.txt"])
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass

    def test_label_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.txt")
            with open(path, "w") as f:
                f.write("hello")
            try:
                load_texts([path], labels=["a", "b"])
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "length" in str(e).lower()


class TestTextDocument:
    def test_dataclass(self):
        doc = TextDocument(path="/tmp/x.txt", label="X", text="content")
        assert doc.path == "/tmp/x.txt"
        assert doc.label == "X"
        assert doc.text == "content"
