"""
Sentence parsing with POS-tag extraction.

Provides sentence segmentation (with a custom abbreviation-aware
boundary detector) and per-sentence POS-tag sequences, which feed
into the Adaptive Evolution (APE) and Keyword ERP (KEPM) analyzers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import spacy

from spellcaster.core.nlp import get_nlp


# ---------------------------------------------------------------------------
# Default abbreviation set for sentence boundary detection
# ---------------------------------------------------------------------------

DEFAULT_ABBREVIATIONS: frozenset[str] = frozenset({
    "mr.", "mrs.", "ms.", "dr.", "prof.", "rev.", "col.",
    "gen.", "maj.", "capt.", "lt.", "sgt.", "pvt.",
    "jr.", "sr.", "etc.", "e.g.", "i.e.",
})


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedSentence:
    """A single sentence with its POS tag sequence."""

    text: str
    """The raw sentence text (whitespace-stripped)."""

    pos_tags: list[str] = field(default_factory=list)
    """POS tags for every token in the sentence (e.g. ``['DET', 'NOUN', 'VERB', ...]``)."""


# ---------------------------------------------------------------------------
# Core parsing functions
# ---------------------------------------------------------------------------

def _register_boundary_detector(
    nlp: spacy.Language,
    abbreviations: frozenset[str],
) -> spacy.Language:
    """
    Register (once) a custom sentence-boundary component on *nlp* that
    respects abbreviations and newline boundaries.

    The component is inserted **before** the parser so that spaCy's
    built-in sentenciser can still override when needed.
    """
    from spacy.language import Language as SpacyLanguage

    component_name = "spellcaster_sent_boundary"

    if component_name in nlp.pipe_names:
        return nlp

    @SpacyLanguage.component(component_name)
    def _boundary_detector(doc):
        if len(doc) == 0:
            return doc
        doc[0].is_sent_start = True

        for i, token in enumerate(doc[:-1]):
            nxt = doc[i + 1]
            # Period followed by capitalised word (not after abbreviation)
            if (
                token.text == "."
                and nxt.text
                and nxt.text[0].isupper()
                and not nxt.is_space
                and not nxt.is_punct
                and token.text.lower() not in abbreviations
            ):
                nxt.is_sent_start = True
            # Newline triggers new sentence
            if token.text in ("\n", "\r", "\r\n") and i + 1 < len(doc):
                nxt.is_sent_start = True

        return doc

    nlp.add_pipe(component_name, before="parser")
    return nlp


def parse_sentences(
    text: str,
    *,
    nlp: spacy.Language | None = None,
    model_name: str = "en_core_web_sm",
    abbreviations: frozenset[str] | None = None,
    use_custom_boundaries: bool = True,
) -> list[ParsedSentence]:
    """
    Segment *text* into sentences with per-token POS tags.

    Parameters
    ----------
    text : str
        Raw input text.
    nlp : spacy.Language or None
        Pre-loaded pipeline. When ``None``, loaded via
        :func:`~spellcaster.core.nlp.get_nlp`.
    model_name : str
        spaCy model name (used only when *nlp* is ``None``).
    abbreviations : frozenset[str] or None
        Abbreviations that should **not** trigger sentence boundaries
        (e.g. ``"dr."``).  Defaults to :data:`DEFAULT_ABBREVIATIONS`.
    use_custom_boundaries : bool
        If ``True``, register a custom boundary detector component
        on the pipeline.

    Returns
    -------
    list[ParsedSentence]
        Ordered list of non-empty sentences with POS tags.
    """
    if nlp is None:
        nlp = get_nlp(model_name)

    if use_custom_boundaries:
        abbrevs = abbreviations if abbreviations is not None else DEFAULT_ABBREVIATIONS
        nlp = _register_boundary_detector(nlp, abbrevs)

    doc = nlp(text)
    results: list[ParsedSentence] = []

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if sent_text:
            pos_tags = [token.pos_ for token in sent]
            results.append(ParsedSentence(text=sent_text, pos_tags=pos_tags))

    return results


def split_sentences_simple(text: str) -> list[str]:
    """
    Lightweight sentence splitter using regex (no spaCy required).

    Splits on periods, newlines, and bullet characters.  Used by the
    :class:`~spellcaster.analyzers.complexity_index.TextComplexityAnalyzer`
    which does not need POS tags.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    list[str]
        Non-empty, whitespace-stripped sentence strings.
    """
    clean = re.sub(r"<[^>]*>", "", text)
    parts = re.split(r"[.\n•]+", clean)
    return [s.strip() for s in parts if s.strip()]
