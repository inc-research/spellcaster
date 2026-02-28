"""
Keyword ER=EPR Model (KEPM) Analyzer.

Analyses structural coherence of keyword usage across text, using
concepts from quantum information (spectral entropy of POS
co-occurrence density matrices) and information geometry (NCD-based
structural similarity between keyword context windows).

Pipeline:

1. **Find keyword positions** — Locate sentences containing each keyword.
2. **Extract context windows** — Collect ±N sentences around each mention.
3. **Build POS density matrices** — Co-occurrence matrices of POS tags
   within each keyword window.
4. **Within-keyword measures** — Spectral entropy evolution, structural
   coherence (NCD similarity) between consecutive windows, and EPR-pair
   detection (structurally entangled distant mentions).
5. **Cross-keyword entanglement** — Structural similarity between
   windows of different keywords.

Example
-------
>>> from spellcaster.analyzers.keyword_erp import KeywordERPAnalyzer
>>> kw = KeywordERPAnalyzer(keywords=["information", "network"])
>>> result = kw.analyze(["essay_a.txt", "essay_b.txt"])
>>> result.to_dataframe()
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh

if TYPE_CHECKING:
    import spacy

from spellcaster.core.compression import ncd_similarity
from spellcaster.core.nlp import get_nlp
from spellcaster.io.readers import TextDocument, load_texts, texts_from_strings
from spellcaster.results.keyword import (
    CrossKeywordEntanglement,
    FileKeywordResult,
    KeywordERPResult,
    KeywordMeasures,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal: POS density matrix construction
# ---------------------------------------------------------------------------

def _build_pos_cooccurrence(
    pos_sequences: list[list[str]],
    vocab: list[str],
    window: int = 2,
) -> np.ndarray:
    """
    Build a normalised POS co-occurrence matrix from POS-tag sequences.

    Parameters
    ----------
    pos_sequences : list[list[str]]
        One POS-tag list per sentence in the window.
    vocab : list[str]
        Ordered POS vocabulary.
    window : int
        Maximum token distance for co-occurrence.

    Returns
    -------
    np.ndarray
        Symmetric ``|V|×|V|`` normalised co-occurrence matrix.
    """
    n = len(vocab)
    idx = {tag: i for i, tag in enumerate(vocab)}
    mat = np.zeros((n, n), dtype=float)

    for seq in pos_sequences:
        for i, tag_i in enumerate(seq):
            if tag_i not in idx:
                continue
            # Diagonal: tag frequency (marginal)
            mat[idx[tag_i], idx[tag_i]] += 1
            # Off-diagonal: co-occurrence within window
            for j in range(max(0, i - window), min(len(seq), i + window + 1)):
                if i == j:
                    continue
                tag_j = seq[j]
                if tag_j not in idx:
                    continue
                mat[idx[tag_i], idx[tag_j]] += 1

    total = mat.sum()
    if total > 0:
        mat /= total
    # Ensure symmetry
    mat = (mat + mat.T) / 2.0
    return mat


def _spectral_entropy(matrix: np.ndarray) -> float:
    """
    Von-Neumann-like spectral entropy of a density matrix.

    Computes ``-sum(λ_i * log2(λ_i))`` over the non-negative eigenvalues
    of *matrix*, treating it as a normalised density.
    """
    trace = np.trace(matrix)
    if trace <= 0:
        return 0.0
    rho = matrix / trace  # Normalise to trace-1

    eigvals = eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-12]

    if len(eigvals) == 0:
        return 0.0

    return float(-(eigvals * np.log2(eigvals)).sum())


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class KeywordERPAnalyzer:
    """
    Keyword ER=EPR Model (KEPM) analyzer.

    Parameters
    ----------
    keywords : list[str]
        Keywords to analyse (case-insensitive, matched by lemma).
    context_window : int
        ±N sentences around each keyword mention.
    pos_window : int
        POS co-occurrence window size (tokens).
    min_epr_distance : int
        Minimum mentions apart to qualify as an EPR pair.
    epr_coherence_threshold : float
        NCD similarity threshold above which a pair is considered
        structurally entangled (default 0.6).
    model_name : str
        spaCy model for POS tagging.
    nlp : spacy.Language or None
        Pre-loaded spaCy pipeline.
    """

    def __init__(
        self,
        keywords: list[str],
        context_window: int = 25,
        pos_window: int = 2,
        min_epr_distance: int = 3,
        epr_coherence_threshold: float = 0.6,
        model_name: str = "en_core_web_sm",
        nlp: spacy.Language | None = None,
    ):
        self.keywords = [kw.lower() for kw in keywords]
        self.context_window = context_window
        self.pos_window = pos_window
        self.min_epr_distance = min_epr_distance
        self.epr_coherence_threshold = epr_coherence_threshold
        self._model_name = model_name
        self._nlp = nlp

    # ── Public API ───────────────────────────────────────────────────

    def analyze(
        self,
        texts_or_paths: list[str],
        labels: list[str] | None = None,
        from_files: bool = True,
    ) -> KeywordERPResult:
        """
        Run the full KEPM pipeline on N texts.

        Parameters
        ----------
        texts_or_paths : list[str]
            File paths or raw text strings.
        labels : list[str] or None
            Human-readable labels.
        from_files : bool
            Whether to read from files or treat as raw strings.

        Returns
        -------
        KeywordERPResult
        """
        if from_files:
            documents = load_texts(texts_or_paths, labels=labels)
        else:
            documents = texts_from_strings(texts_or_paths, labels=labels)

        return self.analyze_documents(documents)

    def analyze_documents(
        self,
        documents: list[TextDocument],
    ) -> KeywordERPResult:
        """Analyse pre-loaded :class:`TextDocument` objects."""
        nlp = self._get_nlp()
        file_results: list[FileKeywordResult] = []

        for doc in documents:
            logger.info("Analyzing keywords in %s", doc.label)
            fr = self._analyze_single_file(doc, nlp)
            file_results.append(fr)

        return KeywordERPResult(
            files=file_results,
            keywords=list(self.keywords),
        )

    # ── Internal: per-file analysis ──────────────────────────────────

    def _get_nlp(self) -> spacy.Language:
        if self._nlp is not None:
            return self._nlp
        return get_nlp(self._model_name)

    def _analyze_single_file(
        self,
        doc: TextDocument,
        nlp: spacy.Language,
    ) -> FileKeywordResult:
        """Full KEPM pipeline for one document."""

        # 1. Parse into sentences with POS tags
        spacy_doc = nlp(doc.text)
        sentences = list(spacy_doc.sents)

        # 2. Find keyword positions
        positions = self._find_keyword_positions(sentences, nlp)

        # 3. Extract context windows with POS sequences
        windows = self._extract_windows(sentences, positions)

        # 4. Build POS vocabulary across all windows
        vocab = self._build_vocab(windows)

        # 5. Build density matrices per (keyword, mention)
        matrices = self._build_matrices(windows, vocab)

        # 6. Within-keyword measures
        kw_measures = []
        for kw in self.keywords:
            km = self._within_keyword_measures(kw, windows.get(kw, []), matrices, vocab)
            kw_measures.append(km)

        # 7. Cross-keyword entanglement
        cross = []
        for i, kw_a in enumerate(self.keywords):
            for kw_b in self.keywords[i + 1:]:
                ck = self._cross_keyword(kw_a, kw_b, windows, matrices, vocab)
                cross.append(ck)

        return FileKeywordResult(
            file=doc.path,
            label=doc.label,
            keyword_measures=kw_measures,
            cross_keyword=cross,
        )

    # ── Stage: find keyword positions ────────────────────────────────

    def _find_keyword_positions(
        self,
        sentences: list,
        nlp: spacy.Language,
    ) -> dict[str, list[int]]:
        """Map each keyword to a list of sentence indices where it appears."""
        positions: dict[str, list[int]] = {kw: [] for kw in self.keywords}
        kw_set = set(self.keywords)

        for sent_idx, sent in enumerate(sentences):
            found_in_sent: set[str] = set()
            for token in sent:
                lemma = token.lemma_.lower()
                if lemma in kw_set and lemma not in found_in_sent:
                    positions[lemma].append(sent_idx)
                    found_in_sent.add(lemma)

        return positions

    # ── Stage: extract context windows ───────────────────────────────

    def _extract_windows(
        self,
        sentences: list,
        positions: dict[str, list[int]],
    ) -> dict[str, list[dict]]:
        """Extract ±context_window sentences around each keyword mention."""
        n_sents = len(sentences)
        result: dict[str, list[dict]] = {}

        for kw, idxs in positions.items():
            wins = []
            for mention_idx, sent_idx in enumerate(idxs):
                start = max(0, sent_idx - self.context_window)
                end = min(n_sents, sent_idx + self.context_window + 1)
                window_sents = sentences[start:end]

                pos_sequences = [
                    [token.pos_ for token in sent]
                    for sent in window_sents
                ]

                wins.append({
                    "mention_idx": mention_idx,
                    "sent_idx": sent_idx,
                    "pos_sequences": pos_sequences,
                })
            result[kw] = wins

        return result

    # ── Stage: vocabulary + matrices ─────────────────────────────────

    @staticmethod
    def _build_vocab(windows: dict[str, list[dict]]) -> list[str]:
        """Build a sorted POS vocabulary from all windows."""
        tags: set[str] = set()
        for kw_windows in windows.values():
            for win in kw_windows:
                for seq in win["pos_sequences"]:
                    tags.update(seq)
        return sorted(tags)

    def _build_matrices(
        self,
        windows: dict[str, list[dict]],
        vocab: list[str],
    ) -> dict[tuple[str, int], np.ndarray]:
        """Build POS co-occurrence density matrices per (keyword, mention)."""
        matrices: dict[tuple[str, int], np.ndarray] = {}
        for kw, wins in windows.items():
            for win in wins:
                mat = _build_pos_cooccurrence(
                    win["pos_sequences"], vocab, window=self.pos_window,
                )
                matrices[(kw, win["mention_idx"])] = mat
        return matrices

    # ── Stage: within-keyword measures ───────────────────────────────

    def _within_keyword_measures(
        self,
        keyword: str,
        windows: list[dict],
        matrices: dict[tuple[str, int], np.ndarray],
        vocab: list[str],
    ) -> KeywordMeasures:
        """Compute spectral entropy, coherence, and EPR pairs for one keyword."""
        n = len(windows)

        if n == 0:
            return KeywordMeasures(
                keyword=keyword, n_mentions=0,
                mean_entropy=None, entropy_trend=None,
                mean_coherence=None, n_epr_pairs=0,
                mean_epr_strength=None, max_epr_strength=None,
            )

        # Spectral entropies
        entropies = [
            _spectral_entropy(matrices[(keyword, i)])
            for i in range(n)
        ]

        mean_ent = float(np.mean(entropies)) if entropies else None

        # Entropy trend (linear slope)
        trend = None
        if n >= 3:
            x = np.arange(n, dtype=float)
            coeffs = np.polyfit(x, entropies, 1)
            trend = float(coeffs[0])

        # Consecutive coherence (NCD similarity between adjacent windows)
        coherences: list[float] = []
        for i in range(n - 1):
            pos_flat_i = [tag for seq in windows[i]["pos_sequences"] for tag in seq]
            pos_flat_j = [tag for seq in windows[i + 1]["pos_sequences"] for tag in seq]
            if pos_flat_i and pos_flat_j:
                coherences.append(ncd_similarity(pos_flat_i, pos_flat_j))

        mean_coh = float(np.mean(coherences)) if coherences else None

        # EPR pairs: structurally similar distant mentions
        epr_strengths: list[float] = []
        for i in range(n):
            for j in range(i + self.min_epr_distance, n):
                pos_i = [tag for seq in windows[i]["pos_sequences"] for tag in seq]
                pos_j = [tag for seq in windows[j]["pos_sequences"] for tag in seq]
                if not pos_i or not pos_j:
                    continue
                sim = ncd_similarity(pos_i, pos_j)
                if sim >= self.epr_coherence_threshold:
                    epr_strengths.append(sim)

        return KeywordMeasures(
            keyword=keyword,
            n_mentions=n,
            mean_entropy=round(mean_ent, 6) if mean_ent is not None else None,
            entropy_trend=round(trend, 6) if trend is not None else None,
            mean_coherence=round(mean_coh, 6) if mean_coh is not None else None,
            n_epr_pairs=len(epr_strengths),
            mean_epr_strength=(
                round(float(np.mean(epr_strengths)), 6)
                if epr_strengths else None
            ),
            max_epr_strength=(
                round(float(np.max(epr_strengths)), 6)
                if epr_strengths else None
            ),
        )

    # ── Stage: cross-keyword entanglement ────────────────────────────

    def _cross_keyword(
        self,
        kw_a: str,
        kw_b: str,
        windows: dict[str, list[dict]],
        matrices: dict[tuple[str, int], np.ndarray],
        vocab: list[str],
    ) -> CrossKeywordEntanglement:
        """Structural similarity between windows of two different keywords."""
        wins_a = windows.get(kw_a, [])
        wins_b = windows.get(kw_b, [])

        if not wins_a or not wins_b:
            return CrossKeywordEntanglement(
                keyword_a=kw_a, keyword_b=kw_b,
                mean_cross_coherence=None,
                n_cross_epr_pairs=0,
                mean_cross_epr_strength=None,
            )

        sims: list[float] = []
        epr_strengths: list[float] = []

        for wa in wins_a:
            pos_a = [tag for seq in wa["pos_sequences"] for tag in seq]
            if not pos_a:
                continue
            for wb in wins_b:
                pos_b = [tag for seq in wb["pos_sequences"] for tag in seq]
                if not pos_b:
                    continue
                sim = ncd_similarity(pos_a, pos_b)
                sims.append(sim)
                if sim >= self.epr_coherence_threshold:
                    epr_strengths.append(sim)

        return CrossKeywordEntanglement(
            keyword_a=kw_a,
            keyword_b=kw_b,
            mean_cross_coherence=(
                round(float(np.mean(sims)), 6) if sims else None
            ),
            n_cross_epr_pairs=len(epr_strengths),
            mean_cross_epr_strength=(
                round(float(np.mean(epr_strengths)), 6)
                if epr_strengths else None
            ),
        )
