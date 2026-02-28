"""
Action frame extraction from text.

An *action frame* is a verb-centred structure capturing who did what
to whom, extracted from spaCy dependency parses.  Each frame records
the verb lemma together with its nominal subjects, objects, and other
syntactic dependents.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import spacy

from spellcaster.core.nlp import get_nlp


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

# An action frame is a plain dict for now; typed as TypedDict for clarity.
# Using a dict keeps it JSON-serializable and easy to work with in pandas.
ActionFrame = dict[str, Any]
"""
Keys
----
verb : str
    Verb lemma.
subjects : list[str]
    Nominal-subject lemmas (``nsubj``, ``nsubjpass``).
objects : list[str]
    Object / complement lemmas (``dobj``, ``pobj``, ``attr``,
    ``ccomp``, ``xcomp``).
other_deps : list[tuple[str, str]]
    ``(dep_label, lemma)`` for all other children.
"""

# Dependency labels grouped by role
_SUBJECT_DEPS = frozenset({"nsubj", "nsubjpass"})
_OBJECT_DEPS = frozenset({"dobj", "pobj", "attr", "ccomp", "xcomp"})


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_action_frames(
    text: str,
    *,
    nlp: spacy.Language | None = None,
    model_name: str = "en_core_web_sm",
) -> list[ActionFrame]:
    """
    Extract verb-centred action frames from *text*.

    Parameters
    ----------
    text : str
        Raw input text.
    nlp : spacy.Language or None
        Pre-loaded pipeline (must include the ``parser`` component).
        When ``None``, a model is loaded via :func:`~spellcaster.core.nlp.get_nlp`
        with NER disabled for speed.
    model_name : str
        spaCy model name (used only when *nlp* is ``None``).

    Returns
    -------
    list[ActionFrame]
        One dict per verb token found in the text.

    Examples
    --------
    >>> frames = extract_action_frames("The cat chased the mouse.")
    >>> frames[0]["verb"]
    'chase'
    >>> frames[0]["subjects"]
    ['cat']
    >>> frames[0]["objects"]
    ['mouse']
    """
    if nlp is None:
        nlp = get_nlp(model_name, disable=["ner"])

    doc = nlp(text)
    frames: list[ActionFrame] = []

    for tok in doc:
        if tok.pos_ != "VERB":
            continue

        subjects = [
            c.lemma_ for c in tok.children
            if c.dep_.startswith("nsubj")
        ]
        objects = [
            c.lemma_ for c in tok.children
            if c.dep_ in _OBJECT_DEPS
        ]
        other_deps = [
            (c.dep_, c.lemma_) for c in tok.children
            if c.dep_ not in _SUBJECT_DEPS and c.dep_ not in _OBJECT_DEPS
        ]

        frames.append({
            "verb": tok.lemma_,
            "subjects": subjects,
            "objects": objects,
            "other_deps": other_deps,
        })

    return frames


def make_hashable_frame(frame: ActionFrame) -> tuple:
    """
    Convert an action frame dict into a hashable tuple suitable for
    counting in a :class:`~collections.Counter`.

    Returns
    -------
    tuple
        ``(verb, sorted_subjects, sorted_objects, sorted_other_deps)``
    """
    return (
        frame["verb"],
        tuple(sorted(frame["subjects"])),
        tuple(sorted(frame["objects"])),
        tuple(sorted(frame["other_deps"])),
    )
