"""Tests for spellcaster.analyzers.valence_model."""

from __future__ import annotations

import math
from collections import Counter
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from spellcaster.analyzers.valence_model import ValenceModelAnalyzer, _ParsedPost
from spellcaster.results.valence import PostMetrics, ValenceModelResult


# ── Helper to build a _ParsedPost without spaCy ─────────────────────────────

def _make_post(
    label: str = "test",
    tokens: list[str] | None = None,
    frames: list[dict] | None = None,
    noun_deps: list[tuple[str, str, str]] | None = None,
) -> _ParsedPost:
    if tokens is None:
        tokens = ["the", "cat", "sit", "mat", "dog", "run", "park"] * 40
    if frames is None:
        frames = [
            {"verb": "sit", "subjects": ["cat"], "objects": ["mat"], "other_deps": []},
            {"verb": "run", "subjects": ["dog"], "objects": ["park"], "other_deps": []},
            {"verb": "sit", "subjects": ["cat"], "objects": ["rug"], "other_deps": []},
            {"verb": "chase", "subjects": ["dog"], "objects": ["cat"], "other_deps": []},
        ]
    if noun_deps is None:
        noun_deps = [
            ("cat", "quick", "amod"),
            ("dog", "lazy", "amod"),
            ("cat", "sit", "nsubj"),
            ("park", "run", "dobj"),
        ]
    return _ParsedPost(
        path=f"{label}.txt",
        label=label,
        text="placeholder text",
        tokens=tokens,
        frames=frames,
        noun_deps=noun_deps,
    )


# ── _compute_mi_verb_role ────────────────────────────────────────────────────

class TestComputeMIVerbRole:
    def test_basic_subjects(self):
        frames = [
            {"verb": "eat", "subjects": ["cat"], "objects": []},
            {"verb": "eat", "subjects": ["dog"], "objects": []},
            {"verb": "run", "subjects": ["cat"], "objects": []},
        ]
        mi = ValenceModelAnalyzer._compute_mi_verb_role(frames, "subjects")
        assert mi is not None
        assert mi >= 0.0

    def test_perfectly_correlated(self):
        frames = [
            {"verb": "eat", "subjects": ["cat"], "objects": []},
            {"verb": "eat", "subjects": ["cat"], "objects": []},
            {"verb": "run", "subjects": ["dog"], "objects": []},
            {"verb": "run", "subjects": ["dog"], "objects": []},
        ]
        mi = ValenceModelAnalyzer._compute_mi_verb_role(frames, "subjects")
        assert mi is not None
        assert mi > 0.5  # Strongly correlated

    def test_empty_role(self):
        frames = [{"verb": "go", "subjects": [], "objects": []}]
        mi = ValenceModelAnalyzer._compute_mi_verb_role(frames, "subjects")
        assert mi is None

    def test_no_frames(self):
        mi = ValenceModelAnalyzer._compute_mi_verb_role([], "subjects")
        assert mi is None


# ── _compute_token_capacities ────────────────────────────────────────────────

class TestComputeTokenCapacities:
    def test_basic(self):
        frames = [
            {"verb": "eat", "subjects": ["cat"], "objects": ["fish"], "other_deps": []},
            {"verb": "run", "subjects": ["dog"], "objects": [], "other_deps": []},
        ]
        df = ValenceModelAnalyzer._compute_token_capacities(frames)
        assert isinstance(df, pd.DataFrame)
        assert "token" in df.columns
        assert "channel_capacity" in df.columns
        assert len(df) == 5  # eat, cat, fish, run, dog

    def test_empty_frames(self):
        df = ValenceModelAnalyzer._compute_token_capacities([])
        assert len(df) == 0

    def test_capacity_positive(self):
        frames = [
            {"verb": "go", "subjects": ["I"], "objects": ["home"], "other_deps": []},
        ]
        df = ValenceModelAnalyzer._compute_token_capacities(frames)
        assert (df["channel_capacity"] > 0).all()

    def test_frequent_token_higher_capacity(self):
        frames = [
            {"verb": "go", "subjects": [], "objects": [], "other_deps": []},
            {"verb": "go", "subjects": [], "objects": [], "other_deps": []},
            {"verb": "go", "subjects": [], "objects": [], "other_deps": []},
            {"verb": "stop", "subjects": [], "objects": [], "other_deps": []},
        ]
        df = ValenceModelAnalyzer._compute_token_capacities(frames)
        go_cap = df[df["token"] == "go"]["channel_capacity"].iloc[0]
        stop_cap = df[df["token"] == "stop"]["channel_capacity"].iloc[0]
        assert go_cap > stop_cap


# ── _compute_post_metrics (using synthetic _ParsedPost) ──────────────────────

class TestComputePostMetrics:
    def setup_method(self):
        self.analyzer = ValenceModelAnalyzer()
        self.post = _make_post("doc1")
        self.h_corpus = 4.0  # Plausible corpus entropy

    def test_returns_post_metrics_and_df(self):
        pm, tc = self.analyzer._compute_post_metrics(self.post, self.h_corpus)
        assert isinstance(pm, PostMetrics)
        assert isinstance(tc, pd.DataFrame)

    def test_file_label(self):
        pm, _ = self.analyzer._compute_post_metrics(self.post, self.h_corpus)
        assert pm.file == "doc1"

    def test_entropy_text_positive(self):
        pm, _ = self.analyzer._compute_post_metrics(self.post, self.h_corpus)
        assert pm.entropy_text > 0

    def test_token_count(self):
        pm, _ = self.analyzer._compute_post_metrics(self.post, self.h_corpus)
        assert pm.token_count == len(self.post.tokens)

    def test_total_frames(self):
        pm, _ = self.analyzer._compute_post_metrics(self.post, self.h_corpus)
        assert pm.total_frames == 4

    def test_verb_diversity(self):
        pm, _ = self.analyzer._compute_post_metrics(self.post, self.h_corpus)
        # 3 unique verbs / 4 frames = 0.75
        assert pm.verb_diversity is not None
        assert 0 < pm.verb_diversity <= 1.0

    def test_most_common_verb(self):
        pm, _ = self.analyzer._compute_post_metrics(self.post, self.h_corpus)
        assert pm.most_common_verb == "sit"  # Appears twice
        assert pm.most_common_verb_pattern_count == 2

    def test_mi_not_none(self):
        pm, _ = self.analyzer._compute_post_metrics(self.post, self.h_corpus)
        assert pm.mi_verb_subject is not None
        assert pm.mi_verb_object is not None

    def test_coupling_strength(self):
        pm, _ = self.analyzer._compute_post_metrics(self.post, self.h_corpus)
        assert pm.coupling_strength is not None
        assert pm.coupling_strength >= 0

    def test_noun_dependencies(self):
        pm, _ = self.analyzer._compute_post_metrics(self.post, self.h_corpus)
        assert pm.total_noun_dependencies == 4
        assert pm.unique_schema_keywords_in_deps > 0

    def test_schema_valence_entropy(self):
        pm, _ = self.analyzer._compute_post_metrics(self.post, self.h_corpus)
        # With 4 noun deps, we should have some schema-valence data
        assert isinstance(pm.schema_valence_entropy, dict)

    def test_collapse_metrics(self):
        pm, _ = self.analyzer._compute_post_metrics(self.post, self.h_corpus)
        assert isinstance(pm.collapse_curve, list)

    def test_corpus_relative_entropy(self):
        pm, _ = self.analyzer._compute_post_metrics(self.post, self.h_corpus)
        # Post entropy < corpus entropy → positive deficit
        assert pm.shannon_entropy_corpus is not None

    def test_no_frames_graceful(self):
        post = _make_post("empty_frames", frames=[], noun_deps=[])
        pm, tc = self.analyzer._compute_post_metrics(post, self.h_corpus)
        assert pm.total_frames == 0
        assert pm.verb_diversity is None
        assert pm.mi_verb_subject is None
        assert len(tc) == 0


# ── _compute_js_matrix ───────────────────────────────────────────────────────

class TestComputeJSMatrix:
    def setup_method(self):
        self.analyzer = ValenceModelAnalyzer()

    def test_two_posts(self):
        posts = [
            _make_post("a", tokens=["cat", "dog", "cat"] * 20),
            _make_post("b", tokens=["fish", "bird", "fish"] * 20),
        ]
        mat = self.analyzer._compute_js_matrix(posts)
        assert mat is not None
        assert mat.shape == (2, 2)
        assert mat[0, 0] == 0.0
        assert mat[0, 1] > 0  # Different distributions

    def test_single_post_returns_none(self):
        posts = [_make_post("a")]
        assert self.analyzer._compute_js_matrix(posts) is None

    def test_symmetric(self):
        posts = [
            _make_post("a", tokens=["x", "y"] * 30),
            _make_post("b", tokens=["y", "z"] * 30),
            _make_post("c", tokens=["x", "z"] * 30),
        ]
        mat = self.analyzer._compute_js_matrix(posts)
        assert mat.shape == (3, 3)
        np.testing.assert_array_almost_equal(mat, mat.T)

    def test_identical_zero_divergence(self):
        same_tokens = ["word", "other"] * 50
        posts = [_make_post("a", tokens=same_tokens), _make_post("b", tokens=same_tokens)]
        mat = self.analyzer._compute_js_matrix(posts)
        assert mat[0, 1] < 1e-10


# ── build_complexity_profile ─────────────────────────────────────────────────

class TestBuildComplexityProfile:
    def test_basic(self):
        analyzer = ValenceModelAnalyzer()
        pm = PostMetrics(
            file="test.txt", entropy_text=5.0, shannon_entropy_corpus=0.1,
            shannon_entropy_avg=0.3, shannon_entropy_max=0.5, number_of_windows=4,
            token_count=1000, collapse_auc_norm=0.25, coupling_strength=1.2,
            verb_diversity=0.75, frames_per_1k_tokens=50.0,
            schema_keywords_per_1k_tokens=10.0,
        )
        result = ValenceModelResult(posts=[pm])
        profile = analyzer.build_complexity_profile(result)
        assert isinstance(profile, pd.DataFrame)
        assert len(profile) == 1
        # Should have renamed columns
        assert any("Variation" in c for c in profile.columns)

    def test_with_js_divergence_n2(self):
        analyzer = ValenceModelAnalyzer()
        pm1 = PostMetrics(file="a.txt", entropy_text=5.0, shannon_entropy_corpus=0.1,
                          shannon_entropy_avg=0.2, shannon_entropy_max=0.3, number_of_windows=2)
        pm2 = PostMetrics(file="b.txt", entropy_text=4.0, shannon_entropy_corpus=0.2,
                          shannon_entropy_avg=0.3, shannon_entropy_max=0.4, number_of_windows=3)
        js_mat = np.array([[0.0, 0.15], [0.15, 0.0]])
        result = ValenceModelResult(posts=[pm1, pm2], js_divergence_matrix=js_mat)
        profile = analyzer.build_complexity_profile(result)
        assert any("Distance" in c for c in profile.columns)


# ── profile_for_print ────────────────────────────────────────────────────────

class TestProfileForPrint:
    def test_basic(self):
        analyzer = ValenceModelAnalyzer()
        profile = pd.DataFrame({
            "file": ["a.txt", "b.txt"],
            "Variation: Text entropy (bits)": [5.0, 4.0],
            "Redundancy: Multiscale collapse AUC_norm": [0.2, 0.3],
        })
        printed = analyzer.profile_for_print(profile)
        assert isinstance(printed, pd.DataFrame)
        assert "Section" in printed.columns
        assert "Metric" in printed.columns

    def test_delta_columns(self):
        analyzer = ValenceModelAnalyzer()
        profile = pd.DataFrame({
            "file": ["a.txt", "b.txt"],
            "Variation: Text entropy (bits)": [5.0, 4.0],
        })
        printed = analyzer.profile_for_print(profile, add_delta=True)
        assert "Δ (B − A)" in printed.columns


# ── ValenceModelResult integration ───────────────────────────────────────────

class TestValenceModelResultIntegration:
    def test_to_dataframe_excludes_nested(self):
        pm = PostMetrics(
            file="t.txt", entropy_text=5.0, shannon_entropy_corpus=0.1,
            shannon_entropy_avg=0.2, shannon_entropy_max=0.3, number_of_windows=2,
            top_schema_keywords=[("cat", 10), ("dog", 5)],
            schema_valence_entropy={"cat": 1.5},
        )
        df = ValenceModelResult(posts=[pm]).to_dataframe()
        assert "top_schema_keywords" not in df.columns
        assert "schema_valence_entropy" not in df.columns

    def test_nested_data_accessible(self):
        pm = PostMetrics(
            file="t.txt", entropy_text=5.0, shannon_entropy_corpus=0.1,
            shannon_entropy_avg=0.2, shannon_entropy_max=0.3, number_of_windows=2,
            top_schema_keywords=[("cat", 10)],
        )
        result = ValenceModelResult(posts=[pm])
        assert result.posts[0].top_schema_keywords == [("cat", 10)]
