"""
Language Complexity Valence Model (LCVM) Analyzer.

The most comprehensive Spellcaster analyzer, computing ~30 metrics per
document across five analytical dimensions:

1. **Variation** — Shannon entropy of token distributions.
2. **Redundancy** — Multiscale entropy-collapse curves measuring local
   repetition at different window sizes.
3. **Organisation** — Mutual information between verbs and their
   subjects/objects, capturing how tightly the action repertoire is
   coupled.
4. **Repertoire** — Action-frame density and verb diversity.
5. **Semantic breadth** — Noun-dependency richness, schema-keyword
   concentration, and valence entropy.

Additionally computes pairwise Jensen–Shannon divergence across N texts
and per-token channel capacities (Shannon–Hartley analogue).

Example
-------
>>> from spellcaster.analyzers.valence_model import ValenceModelAnalyzer
>>> vm = ValenceModelAnalyzer()
>>> result = vm.analyze(["essay_a.txt", "essay_b.txt"])
>>> result.to_dataframe()
"""

from __future__ import annotations

import collections
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import spacy

from spellcaster.core.compression import compressed_size
from spellcaster.core.entropy import (
    multiscale_collapse_curve,
    shannon_entropy,
    summarize_multiscale_collapse,
    window_collapse,
)
from spellcaster.core.information import (
    channel_capacity,
    js_divergence_from_counters,
    mutual_information,
)
from spellcaster.core.nlp import get_nlp, tokenize
from spellcaster.extractors.action_frames import extract_action_frames, make_hashable_frame
from spellcaster.extractors.noun_dependencies import extract_noun_dependencies
from spellcaster.io.readers import TextDocument, load_texts, texts_from_strings
from spellcaster.results.valence import PostMetrics, ValenceModelResult
from spellcaster.utils.smoothing import per_1k

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal: parsed document bundle
# ---------------------------------------------------------------------------

class _ParsedPost:
    """Intermediate representation of a single parsed document."""

    __slots__ = ("path", "label", "text", "tokens", "frames", "noun_deps")

    def __init__(
        self,
        path: str,
        label: str,
        text: str,
        tokens: list[str],
        frames: list[dict],
        noun_deps: list[tuple[str, str, str]],
    ):
        self.path = path
        self.label = label
        self.text = text
        self.tokens = tokens
        self.frames = frames
        self.noun_deps = noun_deps


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class ValenceModelAnalyzer:
    """
    Language Complexity Valence Model (LCVM) analyzer.

    Parameters
    ----------
    window_sizes : tuple[int, ...]
        Window sizes for multiscale collapse analysis.
    top_k_schemas : int
        Number of top schema keywords to track in detail.
    model_name : str
        spaCy model for tokenisation and dependency parsing.
    nlp : spacy.Language or None
        Pre-loaded pipeline (overrides *model_name*).
    """

    def __init__(
        self,
        window_sizes: tuple[int, ...] = (25, 50, 100, 250, 500),
        top_k_schemas: int = 20,
        model_name: str = "en_core_web_sm",
        nlp: spacy.Language | None = None,
    ):
        self.window_sizes = window_sizes
        self.top_k_schemas = top_k_schemas
        self._model_name = model_name
        self._nlp = nlp

    # ── Public API ───────────────────────────────────────────────────

    def analyze(
        self,
        texts_or_paths: list[str],
        labels: list[str] | None = None,
        from_files: bool = True,
    ) -> ValenceModelResult:
        """
        Run the full LCVM pipeline on N texts.

        Parameters
        ----------
        texts_or_paths : list[str]
            File paths (when *from_files* is ``True``) or raw text
            strings (when ``False``).
        labels : list[str] or None
            Human-readable labels.
        from_files : bool
            Whether to read from files or treat as raw strings.

        Returns
        -------
        ValenceModelResult
        """
        # 1. Load
        if from_files:
            documents = load_texts(texts_or_paths, labels=labels)
        else:
            documents = texts_from_strings(texts_or_paths, labels=labels)

        # 2. Parse all documents
        posts = self._parse_all(documents)
        logger.info("Parsed %d documents", len(posts))

        # 3. Corpus-level entropy (for relative collapse)
        corpus_tokens = [t for p in posts for t in p.tokens]
        corpus_counter = Counter(corpus_tokens)
        h_corpus = shannon_entropy(corpus_counter)

        # 4. Per-document metrics
        post_metrics: list[PostMetrics] = []
        token_capacities: dict[str, pd.DataFrame] = {}

        for p in posts:
            pm, tc = self._compute_post_metrics(p, h_corpus)
            post_metrics.append(pm)
            token_capacities[p.label] = tc

        # 5. Cross-document JS divergence matrix
        js_matrix = self._compute_js_matrix(posts)

        return ValenceModelResult(
            posts=post_metrics,
            js_divergence_matrix=js_matrix,
            token_capacities=token_capacities,
        )

    def analyze_documents(
        self,
        documents: list[TextDocument],
    ) -> ValenceModelResult:
        """
        Analyse pre-loaded :class:`~spellcaster.io.readers.TextDocument` objects.
        """
        posts = self._parse_all(documents)
        corpus_tokens = [t for p in posts for t in p.tokens]
        h_corpus = shannon_entropy(Counter(corpus_tokens))

        post_metrics = []
        token_capacities = {}
        for p in posts:
            pm, tc = self._compute_post_metrics(p, h_corpus)
            post_metrics.append(pm)
            token_capacities[p.label] = tc

        js_matrix = self._compute_js_matrix(posts)
        return ValenceModelResult(
            posts=post_metrics,
            js_divergence_matrix=js_matrix,
            token_capacities=token_capacities,
        )

    def build_complexity_profile(
        self,
        result: ValenceModelResult,
    ) -> pd.DataFrame:
        """
        Extract a concise complexity profile from a full result.

        Returns a DataFrame with one row per text, containing the key
        metrics across all five dimensions plus (for N=2) the pairwise
        JS divergence.  Column names use descriptive ``"Section: Metric"``
        labels.
        """
        df = result.to_dataframe()
        wanted = [
            "file",
            "entropy_text",
            "collapse_auc_norm",
            "peak_win_size",
            "mi_verb_subject",
            "mi_verb_object",
            "coupling_strength",
            "coupling_orientation",
            "frames_per_1k_tokens",
            "verb_diversity",
            "schema_keywords_per_1k_tokens",
            "noun_deps_per_1k_tokens",
            "schema_concentration_entropy",
            "mean_schema_valence_entropy_topk",
        ]
        cols = [c for c in wanted if c in df.columns]
        prof = df[cols].copy()

        # Add JS divergence for the N=2 case
        if (
            result.js_divergence_matrix is not None
            and result.js_divergence_matrix.shape == (2, 2)
        ):
            jsd = float(result.js_divergence_matrix[0, 1])
            prof["js_divergence"] = jsd

        rename = {
            "entropy_text": "Variation: Text entropy (bits)",
            "collapse_auc_norm": "Redundancy: Multiscale collapse AUC_norm",
            "peak_win_size": "Redundancy: Peak scale (win size)",
            "coupling_strength": "Organization: Coupling strength (bits)",
            "coupling_orientation": "Organization: Orientation MI(V;O)-MI(V;S)",
            "frames_per_1k_tokens": "Repertoire: Action density (frames/1k tokens)",
            "verb_diversity": "Repertoire: Verb diversity (unique_verbs/frames)",
            "schema_keywords_per_1k_tokens": "Semantic breadth: Schema keywords/1k tokens",
            "js_divergence": "Distance: JS divergence (distance²)",
        }
        prof = prof.rename(columns=rename)
        return prof

    def profile_for_print(
        self,
        profile_df: pd.DataFrame,
        label: str = "stem",
        add_delta: bool = True,
        group_sections: bool = True,
    ) -> pd.DataFrame:
        """
        Transpose a complexity profile into a tall, print-friendly table.

        Parameters
        ----------
        profile_df : DataFrame
            Output of :meth:`build_complexity_profile`.
        label : str
            ``"stem"`` to use filename stems as column headers.
        add_delta : bool
            If ``True`` and exactly 2 texts, add Δ and %Δ columns.
        group_sections : bool
            If ``True``, split ``"Section: Metric"`` names into a MultiIndex.
        """
        df = profile_df.copy()

        if "file" in df.columns:
            if label == "stem":
                df["Text"] = df["file"].map(lambda p: Path(str(p)).stem)
            else:
                df["Text"] = df["file"].astype(str)
            df = df.drop(columns=["file"]).set_index("Text")

        t = df.T

        if group_sections:
            parts = t.index.to_series().str.split(":", n=1, expand=True)
            section = parts[0].fillna("Other").str.strip()
            metric = parts[1].fillna(parts[0]).str.strip()
            t.index = pd.MultiIndex.from_arrays(
                [section, metric], names=["Section", "Metric"]
            )

        if add_delta and t.shape[1] == 2:
            a, b = t.columns[0], t.columns[1]
            va = pd.to_numeric(t[a], errors="coerce")
            vb = pd.to_numeric(t[b], errors="coerce")
            t["Δ (B − A)"] = vb - va
            t["%Δ (vs A)"] = np.where(va != 0, (vb - va) / va * 100.0, np.nan)

        return t.reset_index()

    # ── Internal: parsing ────────────────────────────────────────────

    def _get_nlp(self) -> spacy.Language:
        if self._nlp is not None:
            return self._nlp
        return get_nlp(self._model_name, disable=["ner"])

    def _get_nlp_tok(self) -> spacy.Language:
        return get_nlp(self._model_name, disable=["parser", "ner"])

    def _parse_all(self, documents: list[TextDocument]) -> list[_ParsedPost]:
        nlp = self._get_nlp()
        posts = []
        for doc in documents:
            frames = extract_action_frames(doc.text, nlp=nlp)
            noun_deps = extract_noun_dependencies(doc.text, nlp=nlp)
            tok_list = tokenize(doc.text, nlp=self._get_nlp_tok())
            posts.append(_ParsedPost(
                path=doc.path,
                label=doc.label,
                text=doc.text,
                tokens=tok_list,
                frames=frames,
                noun_deps=noun_deps,
            ))
        return posts

    # ── Internal: per-post metrics ───────────────────────────────────

    def _compute_post_metrics(
        self,
        p: _ParsedPost,
        h_corpus: float,
    ) -> tuple[PostMetrics, pd.DataFrame]:
        """Compute all metrics for a single document. Returns (PostMetrics, token_capacity_df)."""

        tok_list = p.tokens
        token_count = len(tok_list)
        token_counter = Counter(tok_list)
        h_text = shannon_entropy(token_counter)

        # --- Variation: entropy deficit vs corpus ---
        rel_entropy_deficit = (
            (h_corpus - h_text) / h_corpus if h_corpus > 0 else None
        )

        # --- Redundancy: multiscale collapse ---
        curve = multiscale_collapse_curve(tok_list, win_sizes=self.window_sizes)
        red = summarize_multiscale_collapse(curve, x_scale="log")

        collapses_250 = window_collapse(tok_list, win_size=250)
        collapse_mean_250 = float(np.mean(collapses_250)) if collapses_250 else None
        collapse_max_250 = float(np.max(collapses_250)) if collapses_250 else None
        n_windows_250 = len(collapses_250)

        # --- Repertoire: action frames ---
        total_frames = len(p.frames)
        verb_counter = Counter(f["verb"] for f in p.frames)
        unique_verbs = len(verb_counter)
        verb_diversity = (unique_verbs / total_frames) if total_frames > 0 else None
        frames_1k = per_1k(total_frames, token_count)

        top_verb, top_verb_count = (None, 0)
        if verb_counter:
            top_verb, top_verb_count = verb_counter.most_common(1)[0]

        # --- Organisation: MI + frame entropy ---
        hashable_frames = [make_hashable_frame(f) for f in p.frames]
        h_frames = shannon_entropy(Counter(hashable_frames))

        mi_vs = self._compute_mi_verb_role(p.frames, "subjects")
        mi_vo = self._compute_mi_verb_role(p.frames, "objects")

        coupling_strength = (
            (mi_vs + mi_vo) / 2.0
            if mi_vs is not None and mi_vo is not None
            else None
        )
        coupling_orientation = (
            (mi_vo - mi_vs)
            if mi_vs is not None and mi_vo is not None
            else None
        )

        # --- Semantic breadth: noun dependencies ---
        noun_deps = p.noun_deps
        total_noun_deps = len(noun_deps)
        noun_deps_1k = per_1k(total_noun_deps, token_count)

        schema_keywords = [dep[0] for dep in noun_deps]
        schema_counter = Counter(schema_keywords)
        unique_schemas = len(schema_counter)
        schemas_1k = per_1k(unique_schemas, token_count)
        schema_conc_entropy = shannon_entropy(schema_counter)

        top_schemas = schema_counter.most_common(self.top_k_schemas)
        top_schema_set = {k for k, _ in top_schemas}

        valence_distrib: dict[str, list[str]] = defaultdict(list)
        for sk, vk, _ in noun_deps:
            if sk in top_schema_set:
                valence_distrib[sk].append(vk)

        schema_val_entropy: dict[str, float] = {}
        for sk, valences in valence_distrib.items():
            schema_val_entropy[sk] = round(shannon_entropy(Counter(valences)), 4)

        mean_sv_entropy = (
            float(np.mean(list(schema_val_entropy.values())))
            if schema_val_entropy
            else None
        )

        # --- Token channel capacities ---
        tc_df = self._compute_token_capacities(p.frames)

        # --- Assemble PostMetrics ---
        def _r(v, d=4):
            """Round if not None."""
            return round(v, d) if v is not None and not (isinstance(v, float) and math.isnan(v)) else v

        pm = PostMetrics(
            file=p.label,
            entropy_text=_r(h_text),
            shannon_entropy_corpus=_r(rel_entropy_deficit),
            shannon_entropy_avg=_r(collapse_mean_250),
            shannon_entropy_max=_r(collapse_max_250),
            number_of_windows=n_windows_250,
            collapse_curve=curve,
            collapse_auc=_r(red["collapse_auc"], 6),
            collapse_auc_norm=_r(red["collapse_auc_norm"], 6),
            peak_win_size=red["peak_win_size"],
            peak_mean_collapse=_r(red["peak_mean_collapse"], 6),
            token_count=token_count,
            most_common_verb=top_verb,
            most_common_verb_pattern_count=int(top_verb_count),
            total_frames=total_frames,
            unique_verbs=unique_verbs,
            verb_diversity=_r(verb_diversity, 6),
            frames_per_1k_tokens=_r(frames_1k, 6),
            entropy_frames=_r(h_frames),
            mi_verb_subject=_r(mi_vs, 6),
            mi_verb_object=_r(mi_vo, 6),
            coupling_strength=_r(coupling_strength, 6),
            coupling_orientation=_r(coupling_orientation, 6),
            total_noun_dependencies=total_noun_deps,
            noun_deps_per_1k_tokens=_r(noun_deps_1k, 6),
            unique_schema_keywords_in_deps=unique_schemas,
            schema_keywords_per_1k_tokens=_r(schemas_1k, 6),
            schema_concentration_entropy=_r(schema_conc_entropy, 6),
            mean_schema_valence_entropy_topk=_r(mean_sv_entropy, 6),
            top_schema_keywords=top_schemas,
            schema_valence_entropy=schema_val_entropy,
            valence_distributions=dict(valence_distrib),
        )
        return pm, tc_df

    # ── Internal: mutual information for verb–role pairs ─────────────

    @staticmethod
    def _compute_mi_verb_role(
        frames: list[dict],
        role_key: str,
    ) -> float | None:
        """MI(Verb; Role) where role_key is 'subjects' or 'objects'."""
        pairs = [(f["verb"], r) for f in frames for r in f.get(role_key, [])]
        if not pairs:
            return None
        joint = Counter(pairs)
        verb_marginal = Counter(v for v, _ in pairs)
        role_marginal = Counter(r for _, r in pairs)
        return mutual_information(joint, verb_marginal, role_marginal, len(pairs))

    # ── Internal: token channel capacities ───────────────────────────

    @staticmethod
    def _compute_token_capacities(frames: list[dict]) -> pd.DataFrame:
        """Shannon–Hartley channel capacity per unique token in action frames."""
        frame_tokens: list[str] = []
        for f in frames:
            frame_tokens.append(f["verb"])
            frame_tokens.extend(f.get("subjects", []))
            frame_tokens.extend(f.get("objects", []))
            frame_tokens.extend(dep[1] for dep in f.get("other_deps", []))

        if not frame_tokens:
            return pd.DataFrame(columns=["token", "channel_capacity"])

        counter = Counter(frame_tokens)
        total = len(frame_tokens)
        rows = []
        for token in sorted(set(frame_tokens)):
            s = counter[token]
            n = total - s
            rows.append({"token": token, "channel_capacity": channel_capacity(s, max(n, 0))})

        return pd.DataFrame(rows)

    # ── Internal: JS divergence matrix ───────────────────────────────

    def _compute_js_matrix(self, posts: list[_ParsedPost]) -> np.ndarray | None:
        """N×N pairwise JS divergence matrix.  Returns None for N < 2."""
        n = len(posts)
        if n < 2:
            return None

        counters = [Counter(p.tokens) for p in posts]
        mat = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = js_divergence_from_counters(counters[i], counters[j])
                mat[i, j] = d
                mat[j, i] = d
        return mat
