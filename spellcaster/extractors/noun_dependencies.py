"""
Noun-dependency extraction (schema–valence pairs).

Extracts structured relationships between nouns and their modifiers
or governing verbs.  In Spellcaster's terminology:

* **Schema keyword** — the noun lemma (the concept being modified).
* **Valence keyword** — the adjective or verb lemma that colours the
  noun's meaning in context.
* **Dependency type** — the syntactic relation (``amod``, ``nsubj``,
  ``dobj``).

These triples power the *valence entropy* and *semantic breadth*
metrics in the Valence Model analyzer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import spacy

from spellcaster.core.nlp import get_nlp


# Type alias for a single dependency triple
NounDependency = tuple[str, str, str]
"""``(schema_keyword, valence_keyword, dep_type)``"""


def extract_noun_dependencies(
    text: str,
    *,
    nlp: spacy.Language | None = None,
    model_name: str = "en_core_web_sm",
) -> list[NounDependency]:
    """
    Extract noun–adjective and noun–verb dependency triples from *text*.

    Three dependency patterns are captured:

    1. **amod** — An adjective modifying a noun
       (e.g. *"quick fox"* → ``("fox", "quick", "amod")``).
    2. **nsubj** — A noun serving as subject of a verb
       (e.g. *"The fox jumps"* → ``("fox", "jump", "nsubj")``).
    3. **dobj** — A noun serving as direct object of a verb
       (e.g. *"chase the mouse"* → ``("mouse", "chase", "dobj")``).

    Stop-word valence keywords are filtered out.

    Parameters
    ----------
    text : str
        Raw input text.
    nlp : spacy.Language or None
        Pre-loaded pipeline (must include ``parser``).
        When ``None``, loaded via :func:`~spellcaster.core.nlp.get_nlp`.
    model_name : str
        spaCy model name (used only when *nlp* is ``None``).

    Returns
    -------
    list[NounDependency]
        List of ``(schema_keyword, valence_keyword, dep_type)`` tuples.
    """
    if nlp is None:
        nlp = get_nlp(model_name, disable=["ner"])

    doc = nlp(text)
    dependencies: list[NounDependency] = []

    for tok in doc:
        # --- Pattern 1: adjective modifier of a noun ---
        if tok.pos_ in {"NOUN", "PROPN"}:
            schema = tok.lemma_
            for child in tok.children:
                if child.dep_ == "amod":
                    valence = child.lemma_
                    if not nlp.vocab[valence].is_stop:
                        dependencies.append((schema, valence, "amod"))

        # --- Patterns 2 & 3: noun as subject or object of a verb ---
        if tok.pos_ == "VERB":
            valence = tok.lemma_
            if nlp.vocab[valence].is_stop:
                continue
            for child in tok.children:
                if child.pos_ in {"NOUN", "PROPN"}:
                    schema = child.lemma_
                    if child.dep_ == "nsubj":
                        dependencies.append((schema, valence, "nsubj"))
                    elif child.dep_ == "dobj":
                        dependencies.append((schema, valence, "dobj"))

    return dependencies
