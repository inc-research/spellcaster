"""
Adaptive POS Evolution (APE) Analyzer.

Treats syntactic structures as biological *species* competing for
"cognitive market share" across documents.  The pipeline:

1. **Parse** — Segment each document into sentences with POS tags.
2. **Structural similarity** — Compute NCD-based structural similarity
   between every pair of sentences (compression distance on POS-tag
   sequences).
3. **Embed** — Optionally compute semantic embeddings (sentence-transformers).
4. **Hybrid distance** — Blend structural and semantic distances via a
   configurable weight ``alpha``.
5. **Cluster** — Agglomerative clustering on the hybrid distance matrix
   with a data-driven distance threshold.
6. **Evolutionary dynamics** — Track cluster (species) density across
   documents to classify each as Emerging, Extinct, Thriving, Declining,
   or Stable.
7. **POS composition** — Profile the syntactic makeup of each species.

Example
-------
>>> from spellcaster.analyzers.adaptive_evolution import AdaptiveEvolutionAnalyzer
>>> ape = AdaptiveEvolutionAnalyzer()
>>> result = ape.analyze(["early_draft.txt", "final_draft.txt"])
>>> result.to_dataframe()
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import spacy

from spellcaster.core.compression import ncd_similarity
from spellcaster.extractors.sentence_parser import parse_sentences, ParsedSentence
from spellcaster.io.readers import TextDocument, load_texts, texts_from_strings
from spellcaster.results.evolution import (
    EvolutionaryStatus,
    EvolutionResult,
    POSComposition,
    SpeciesRecord,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class AdaptiveEvolutionAnalyzer:
    """
    Adaptive POS Evolution (APE) analyzer.

    Parameters
    ----------
    alpha_semantic : float
        Weight for semantic (embedding) distance vs. structural (NCD)
        distance.  0.0 = pure structural, 1.0 = pure semantic,
        0.5 = equal blend.  Ignored when *use_embeddings* is ``False``.
    status_threshold : float
        Minimum absolute density delta to classify a species as
        Thriving or Declining (default ``0.02`` = 2%).
    top_k_pos : int
        Number of POS tags to report per species cluster.
    embedding_model : str
        Sentence-transformer model name for semantic embeddings.
    use_embeddings : bool
        If ``False``, skip semantic embeddings entirely and cluster
        on structural (NCD) distance alone.  This avoids the
        ``sentence-transformers`` dependency.
    model_name : str
        spaCy model for sentence parsing and POS tagging.
    nlp : spacy.Language or None
        Pre-loaded spaCy pipeline (overrides *model_name*).
    """

    def __init__(
        self,
        alpha_semantic: float = 0.5,
        status_threshold: float = 0.02,
        top_k_pos: int = 10,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_embeddings: bool = True,
        model_name: str = "en_core_web_sm",
        nlp: spacy.Language | None = None,
    ):
        self.alpha_semantic = alpha_semantic
        self.status_threshold = status_threshold
        self.top_k_pos = top_k_pos
        self.embedding_model = embedding_model
        self.use_embeddings = use_embeddings
        self._model_name = model_name
        self._nlp = nlp

    # ── Public API ───────────────────────────────────────────────────

    def analyze(
        self,
        texts_or_paths: list[str],
        labels: list[str] | None = None,
        from_files: bool = True,
    ) -> EvolutionResult:
        """
        Run the full APE pipeline on N ordered documents.

        Documents should be in chronological order (earliest first).

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
        EvolutionResult
        """
        if from_files:
            documents = load_texts(texts_or_paths, labels=labels)
        else:
            documents = texts_from_strings(texts_or_paths, labels=labels)

        return self.analyze_documents(documents)

    def analyze_documents(
        self,
        documents: list[TextDocument],
    ) -> EvolutionResult:
        """Analyse pre-loaded :class:`TextDocument` objects."""

        # 1. Parse sentences with POS tags
        parsed = self._prepare_data(documents)
        logger.info("Parsed %d sentences from %d documents", len(parsed["df"]), len(documents))

        df = parsed["df"]
        if len(df) < 2:
            logger.warning("Fewer than 2 sentences — cannot cluster.")
            return EvolutionResult(document_order=parsed["doc_order"])

        # 2. Structural similarity matrix (NCD on POS tags)
        struct_sim = self._compute_structural_similarity(df)

        # 3. Build distance matrix (hybrid or pure structural)
        dist_matrix = self._build_distance_matrix(df, struct_sim)

        # 4. Cluster
        df = self._cluster(df, dist_matrix)

        # 5. Evolutionary dynamics
        species = self._compute_evolutionary_dynamics(df, parsed["doc_order"])

        return EvolutionResult(
            species=species,
            structural_similarity_matrix=struct_sim,
            cluster_assignments=df,
            document_order=parsed["doc_order"],
        )

    # ── Stage 1: Parse ───────────────────────────────────────────────

    def _prepare_data(
        self,
        documents: list[TextDocument],
    ) -> dict:
        """Parse all documents into a combined DataFrame of sentences."""
        all_rows: list[dict] = []

        for doc in documents:
            try:
                sents = parse_sentences(
                    doc.text,
                    nlp=self._nlp,
                    model_name=self._model_name,
                )
            except Exception:
                # Fallback: basic splitting (no POS tags)
                logger.warning("spaCy parse failed for %s; skipping", doc.label)
                continue

            for s in sents:
                all_rows.append({
                    "Sentence": s.text,
                    "POS_Tags": s.pos_tags,
                    "Source_Document": doc.label,
                })

        df = pd.DataFrame(all_rows)
        doc_order = [d.label for d in documents]

        return {"df": df, "doc_order": doc_order}

    def prepare_data_from_parsed(
        self,
        sentences_per_doc: dict[str, list[ParsedSentence]],
        doc_order: list[str] | None = None,
    ) -> dict:
        """
        Build the internal DataFrame from pre-parsed sentences.

        Useful for testing or when sentences have already been parsed.

        Parameters
        ----------
        sentences_per_doc : dict
            Mapping of document label → list of :class:`ParsedSentence`.
        doc_order : list[str] or None
            Chronological order.  Defaults to dict key order.

        Returns
        -------
        dict with ``"df"`` and ``"doc_order"`` keys.
        """
        rows = []
        for doc_label, sents in sentences_per_doc.items():
            for s in sents:
                rows.append({
                    "Sentence": s.text,
                    "POS_Tags": s.pos_tags,
                    "Source_Document": doc_label,
                })
        df = pd.DataFrame(rows)
        order = doc_order or list(sentences_per_doc.keys())
        return {"df": df, "doc_order": order}

    # ── Stage 2: Structural similarity ───────────────────────────────

    @staticmethod
    def _compute_structural_similarity(df: pd.DataFrame) -> np.ndarray:
        """NCD-based structural similarity matrix for all sentence pairs."""
        n = len(df)
        mat = np.eye(n, dtype=float)

        pos_lists = df["POS_Tags"].tolist()
        for i in range(n):
            for j in range(i + 1, n):
                sim = ncd_similarity(pos_lists[i], pos_lists[j])
                mat[i, j] = sim
                mat[j, i] = sim

        return mat

    # ── Stage 3: Distance matrix ─────────────────────────────────────

    def _build_distance_matrix(
        self,
        df: pd.DataFrame,
        structural_similarity: np.ndarray,
    ) -> np.ndarray:
        """
        Build a hybrid or pure-structural distance matrix.

        When *use_embeddings* is True, blends structural and semantic
        distances using *alpha_semantic*.
        """
        structural_distance = 1.0 - structural_similarity

        if not self.use_embeddings or self.alpha_semantic == 0.0:
            dist = structural_distance.copy()
        else:
            semantic_sim = self._compute_semantic_similarity(df)
            semantic_distance = 1.0 - semantic_sim
            dist = (
                self.alpha_semantic * semantic_distance
                + (1.0 - self.alpha_semantic) * structural_distance
            )

        # Ensure symmetry and zero diagonal
        dist = (dist + dist.T) / 2.0
        np.fill_diagonal(dist, 0.0)
        return dist

    def _compute_semantic_similarity(self, df: pd.DataFrame) -> np.ndarray:
        """Cosine similarity of sentence embeddings."""
        from spellcaster.utils.imports import require_sentence_transformers, require_sklearn

        SentenceTransformer = require_sentence_transformers()
        sklearn = require_sklearn()
        from sklearn.metrics.pairwise import cosine_similarity

        logger.info("Computing sentence embeddings with %s", self.embedding_model)
        model = SentenceTransformer(self.embedding_model)
        embeddings = model.encode(df["Sentence"].tolist())
        return cosine_similarity(embeddings)

    # ── Stage 4: Cluster ─────────────────────────────────────────────

    @staticmethod
    def _cluster(
        df: pd.DataFrame,
        distance_matrix: np.ndarray,
        threshold_offset: float = 0.5,
    ) -> pd.DataFrame:
        """
        Agglomerative clustering with a data-driven distance threshold.

        Parameters
        ----------
        df : DataFrame
            Must include at least ``Sentence`` and ``Source_Document``.
        distance_matrix : np.ndarray
            Precomputed distance matrix.
        threshold_offset : float
            Number of standard deviations below the mean distance to
            set as the clustering threshold (tighter = more clusters).

        Returns
        -------
        DataFrame with an added ``Group_ID`` column.
        """
        from spellcaster.utils.imports import require_sklearn
        require_sklearn()
        from sklearn.cluster import AgglomerativeClustering

        auto_threshold = float(
            np.mean(distance_matrix) - np.std(distance_matrix) * threshold_offset
        )
        auto_threshold = max(auto_threshold, 0.01)  # Floor

        logger.info("Clustering with auto threshold=%.4f", auto_threshold)

        agg = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=auto_threshold,
            metric="precomputed",
            linkage="average",
        )
        df = df.copy()
        df["Group_ID"] = agg.fit_predict(distance_matrix)

        logger.info("Found %d clusters", df["Group_ID"].nunique())
        return df

    # ── Stage 5: Evolutionary dynamics ───────────────────────────────

    def _compute_evolutionary_dynamics(
        self,
        df: pd.DataFrame,
        doc_order: list[str],
    ) -> list[SpeciesRecord]:
        """
        Compute density, delta, status, and POS composition per cluster.

        Generalizes to N documents by comparing the earliest and latest
        in *doc_order*.
        """
        if "Group_ID" not in df.columns:
            return []

        # Density per (document, cluster)
        counts = (
            df.groupby(["Source_Document", "Group_ID"])
            .size()
            .unstack(fill_value=0)
        )
        density = counts.div(counts.sum(axis=1), axis=0)

        first_doc = doc_order[0]
        last_doc = doc_order[-1]

        # Ensure both docs exist in density index
        for d in [first_doc, last_doc]:
            if d not in density.index:
                density.loc[d] = 0.0

        species: list[SpeciesRecord] = []

        for cluster_id in sorted(density.columns):
            d_start = float(density.loc[first_doc, cluster_id])
            d_end = float(density.loc[last_doc, cluster_id])
            delta = d_end - d_start

            status = self._classify_status(d_start, d_end, delta)

            # Sample sentence
            cluster_rows = df[df["Group_ID"] == cluster_id]
            sample = (
                cluster_rows["Sentence"].iloc[0]
                if not cluster_rows.empty
                else ""
            )

            # POS composition
            pos_comp = self._compute_pos_composition(cluster_rows)

            species.append(SpeciesRecord(
                cluster_id=int(cluster_id),
                status=status,
                density_start=round(d_start, 6),
                density_end=round(d_end, 6),
                delta=round(delta, 6),
                sample_sentence=sample,
                pos_composition=pos_comp,
            ))

        # Sort by end-state density (descending)
        species.sort(key=lambda s: s.density_end, reverse=True)
        return species

    def _classify_status(
        self,
        d_start: float,
        d_end: float,
        delta: float,
    ) -> EvolutionaryStatus:
        """Classify a species' evolutionary trajectory."""
        if d_start == 0:
            return EvolutionaryStatus.EMERGING
        if d_end == 0:
            return EvolutionaryStatus.EXTINCT
        if delta > self.status_threshold:
            return EvolutionaryStatus.THRIVING
        if delta < -self.status_threshold:
            return EvolutionaryStatus.DECLINING
        return EvolutionaryStatus.STABLE

    def _compute_pos_composition(
        self,
        cluster_df: pd.DataFrame,
    ) -> list[POSComposition]:
        """Top-K POS tag frequencies for a cluster."""
        if cluster_df.empty or "POS_Tags" not in cluster_df.columns:
            return []

        all_tags = [
            tag
            for tags in cluster_df["POS_Tags"]
            for tag in (tags if isinstance(tags, list) else [])
        ]
        if not all_tags:
            return []

        counter = Counter(all_tags)
        total = sum(counter.values())

        return [
            POSComposition(
                tag=tag,
                percentage=round(count / total * 100, 2),
                count=count,
            )
            for tag, count in counter.most_common(self.top_k_pos)
        ]
