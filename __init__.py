"""
NLP extraction modules for Spellcaster.

* :mod:`.action_frames` — Verb-centred action frame extraction.
* :mod:`.noun_dependencies` — Schema–valence noun dependency triples.
* :mod:`.sentence_parser` — Sentence segmentation with POS tags.
"""

from spellcaster.extractors.action_frames import (
    ActionFrame,
    extract_action_frames,
    make_hashable_frame,
)
from spellcaster.extractors.noun_dependencies import (
    NounDependency,
    extract_noun_dependencies,
)
from spellcaster.extractors.sentence_parser import (
    DEFAULT_ABBREVIATIONS,
    ParsedSentence,
    parse_sentences,
    split_sentences_simple,
)

__all__ = [
    "ActionFrame",
    "extract_action_frames",
    "make_hashable_frame",
    "NounDependency",
    "extract_noun_dependencies",
    "ParsedSentence",
    "parse_sentences",
    "split_sentences_simple",
    "DEFAULT_ABBREVIATIONS",
]