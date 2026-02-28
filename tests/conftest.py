"""Shared fixtures for the spellcaster test suite."""

import pytest
from collections import Counter


# ---------------------------------------------------------------------------
# Sample texts for integration tests
# ---------------------------------------------------------------------------

SAMPLE_TEXT_A = (
    "The quick brown fox jumps over the lazy dog. "
    "A swift red fox leaps across the sleeping hound. "
    "The agile creature bounded through the meadow with grace."
)

SAMPLE_TEXT_B = (
    "Information flows through networks like water through pipes. "
    "Networks carry data from node to node efficiently. "
    "The system processes information at remarkable speed."
)

SAMPLE_TEXT_SHORT = "Hello world."

SAMPLE_TEXT_EMPTY = ""


@pytest.fixture
def sample_text_a():
    return SAMPLE_TEXT_A


@pytest.fixture
def sample_text_b():
    return SAMPLE_TEXT_B


@pytest.fixture
def sample_text_short():
    return SAMPLE_TEXT_SHORT


# ---------------------------------------------------------------------------
# Pre-built counters for unit tests (no spaCy needed)
# ---------------------------------------------------------------------------

@pytest.fixture
def uniform_counter():
    """Uniform distribution over 4 tokens — entropy should be 2.0 bits."""
    return Counter({"a": 10, "b": 10, "c": 10, "d": 10})


@pytest.fixture
def skewed_counter():
    """Heavily skewed — lower entropy than uniform."""
    return Counter({"a": 100, "b": 1, "c": 1, "d": 1})


@pytest.fixture
def single_token_counter():
    """Only one token type — entropy should be 0.0."""
    return Counter({"a": 50})


@pytest.fixture
def empty_counter():
    return Counter()


@pytest.fixture
def token_list_repetitive():
    """Highly repetitive token list — high collapse expected."""
    return ["the"] * 200 + ["cat"] * 50


@pytest.fixture
def token_list_diverse():
    """Diverse token list — low collapse expected."""
    return [f"word{i}" for i in range(250)]
