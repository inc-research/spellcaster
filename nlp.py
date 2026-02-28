"""
Shared NLP infrastructure: spaCy model management and tokenization.

Models are loaded lazily (never at import time) and cached so that repeated
calls with the same configuration reuse the same ``spacy.Language`` instance.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import spacy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------
_model_cache: dict[str, spacy.Language] = {}


def get_nlp(
    model_name: str = "en_core_web_sm",
    disable: list[str] | None = None,
) -> spacy.Language:
    """
    Load (and cache) a spaCy model.

    Parameters
    ----------
    model_name : str
        Any installed spaCy model name (e.g. ``"en_core_web_sm"``).
    disable : list[str] or None
        Pipeline components to disable (e.g. ``["ner"]``).

    Returns
    -------
    spacy.Language
        The loaded (or cached) pipeline.

    Raises
    ------
    spellcaster.exceptions.ModelNotLoadedError
        If the requested model is not installed.
    """
    import spacy
    from spellcaster.exceptions import ModelNotLoadedError

    disable = disable or []
    cache_key = f"{model_name}|{','.join(sorted(disable))}"

    if cache_key not in _model_cache:
        try:
            logger.debug("Loading spaCy model %r (disable=%s)", model_name, disable)
            _model_cache[cache_key] = spacy.load(model_name, disable=disable)
        except OSError as exc:
            raise ModelNotLoadedError(
                f"spaCy model '{model_name}' is not installed. "
                f"Run: python -m spacy download {model_name}"
            ) from exc

    return _model_cache[cache_key]


def clear_model_cache() -> None:
    """Remove all cached models (useful in tests or to reclaim memory)."""
    _model_cache.clear()


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize(
    text: str,
    model_name: str = "en_core_web_sm",
    nlp: spacy.Language | None = None,
) -> list[str]:
    """
    Tokenize *text* into lowercase lemmas, keeping only alphabetic tokens
    and discarding stop-words.

    This is the canonical tokenizer used across Spellcaster for
    entropy, information-theoretic, and frequency-based analyses.

    Parameters
    ----------
    text : str
        Raw input text.
    model_name : str
        spaCy model to use (ignored when *nlp* is provided).
    nlp : spacy.Language or None
        Pre-loaded pipeline.  When ``None``, a model is loaded via
        :func:`get_nlp` with the parser and NER disabled for speed.

    Returns
    -------
    list[str]
        Ordered list of lowercase lemma strings.
    """
    if nlp is None:
        nlp = get_nlp(model_name, disable=["parser", "ner"])

    doc = nlp(text.lower())
    return [t.lemma_ for t in doc if t.is_alpha and not t.is_stop]
