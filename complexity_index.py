"""
Language Complexity Index (LCX) Analyzer.

Measures how text complexity accumulates sentence-by-sentence using
three complementary signals:

1. **Narrative accumulation** (``k_hist``) — cumulative compressed size
   as each sentence is appended, approximating the growth of unique
   information via LZ77 (zlib).

2. **Lexical volatility** (``volatility``) — Levenshtein edit distance
   between consecutive sentences, capturing how much the surface form
   shifts from one sentence to the next.

3. **Cognitive synergy** (``synergy``) — the ratio of volatility to
   marginal compression cost (``volatility / delta_k``).  A high
   synergy score means a sentence introduced a large surface-level
   change while adding relatively little *new* information to the
   narrative history — it stayed "on pattern".

Example
-------
>>> from spellcaster.analyzers.complexity_index import TextComplexityAnalyzer
>>> analyzer = TextComplexityAnalyzer()
>>> result = analyzer.compare(
...     ["First text content...", "Second text content..."],
...     labels=["Human", "GPT"],
...     from_files=False,
... )
>>> result.to_dataframe().head()
"""

from __future__ import annotations

import logging

from spellcaster.core.compression import compressed_size
from spellcaster.extractors.sentence_parser import split_sentences_simple
from spellcaster.io.readers import TextDocument, load_texts, texts_from_strings
from spellcaster.results.complexity import (
    ComplexityComparisonResult,
    ComplexityFlowResult,
    SentenceMetrics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Levenshtein with graceful fallback
# ---------------------------------------------------------------------------

def _levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein edit distance between two strings.

    Uses the fast C-extension ``python-Levenshtein`` package when
    available, otherwise falls back to a pure-Python implementation.
    """
    try:
        import Levenshtein
        return Levenshtein.distance(s1, s2)
    except ImportError:
        pass

    # Pure-Python fallback (standard DP algorithm)
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Insertion, deletion, substitution
            insert = prev_row[j + 1] + 1
            delete = curr_row[j] + 1
            subst = prev_row[j] + (0 if c1 == c2 else 1)
            curr_row.append(min(insert, delete, subst))
        prev_row = curr_row

    return prev_row[-1]


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class TextComplexityAnalyzer:
    """
    Language Complexity Index (LCX) analyzer.

    Processes text sentence-by-sentence to produce a complexity flow —
    a time-series of compression growth, lexical volatility, and
    cognitive synergy.

    Parameters
    ----------
    smoothing_window : int
        Default smoothing window used by convenience plotting (not
        applied to the raw results, which are always unsmoothed).
    """

    def __init__(self, smoothing_window: int = 3):
        self.smoothing_window = smoothing_window

    # ── Single-text analysis ─────────────────────────────────────────

    def analyze_flow(
        self,
        text: str,
        label: str = "",
    ) -> ComplexityFlowResult:
        """
        Analyse a single text's complexity flow.

        Parameters
        ----------
        text : str
            Raw text content.
        label : str
            Human-readable identifier for this text.

        Returns
        -------
        ComplexityFlowResult
            Per-sentence metrics in document order.
        """
        sentences = split_sentences_simple(text)

        if not sentences:
            logger.warning("No sentences found in text (label=%r)", label)
            return ComplexityFlowResult(label=label)

        metrics: list[SentenceMetrics] = []
        history = ""
        last_sentence = ""
        last_k_hist = 0

        for i, sentence in enumerate(sentences):
            # Narrative accumulation
            history += sentence + " "
            k_curr = compressed_size(history)

            # Marginal information (floor of 1 to avoid division by zero)
            delta_k = max(1, k_curr - last_k_hist)

            # Lexical volatility
            lev_dist = (
                _levenshtein_distance(last_sentence, sentence)
                if last_sentence
                else 0
            )

            # Cognitive synergy
            synergy = lev_dist / delta_k if delta_k > 0 else 0.0

            metrics.append(SentenceMetrics(
                text=sentence,
                index=i,
                k_hist=k_curr,
                volatility=lev_dist,
                synergy=synergy,
            ))

            last_sentence = sentence
            last_k_hist = k_curr

        logger.info(
            "LCX analysis complete: label=%r, sentences=%d",
            label, len(metrics),
        )
        return ComplexityFlowResult(label=label, sentences=metrics)

    # ── N-text comparison ────────────────────────────────────────────

    def compare(
        self,
        texts_or_paths: list[str],
        labels: list[str] | None = None,
        from_files: bool = True,
    ) -> ComplexityComparisonResult:
        """
        Compare N texts via complexity flow analysis.

        Parameters
        ----------
        texts_or_paths : list[str]
            File paths (when *from_files* is ``True``) or raw text
            strings (when ``False``).
        labels : list[str] or None
            Human-readable labels, one per text.  When ``None``,
            defaults to file stems or ``"text_0"``, ``"text_1"``, etc.
        from_files : bool
            If ``True``, treat *texts_or_paths* as file paths and
            read them from disk.  If ``False``, treat them as raw
            text content.

        Returns
        -------
        ComplexityComparisonResult
            One :class:`ComplexityFlowResult` per input text.
        """
        if from_files:
            documents = load_texts(texts_or_paths, labels=labels)
        else:
            documents = texts_from_strings(texts_or_paths, labels=labels)

        flows = [
            self.analyze_flow(doc.text, label=doc.label)
            for doc in documents
        ]

        return ComplexityComparisonResult(flows=flows)

    # ── Convenience: analyse a single TextDocument ───────────────────

    def analyze_document(self, doc: TextDocument) -> ComplexityFlowResult:
        """
        Analyse a pre-loaded :class:`~spellcaster.io.readers.TextDocument`.
        """
        return self.analyze_flow(doc.text, label=doc.label)
