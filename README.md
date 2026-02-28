# Spellcaster

Analyze natural language text through complex systems science, information theory, information-theoretic physics analogues, and evolutionary game theory.

Spellcaster provides four complementary analyzers that reveal the hidden structural, informational, and evolutionary properties of text — from sentence-level compression dynamics to document-scale syntactic evolution.

## Installation

```bash
pip install spellcaster
```

**Required dependencies** (installed automatically): `numpy`, `pandas`, `spacy`, `scipy`, `python-Levenshtein`

**Optional dependencies:**

```bash
# Sentence embeddings for APE hybrid clustering
pip install spellcaster[ml]      # sentence-transformers, scikit-learn

# Visualization (convenience plotting)
pip install spellcaster[viz]     # matplotlib, seaborn

# Everything
pip install spellcaster[all]
```

**spaCy model** (required):

```bash
python -m spacy download en_core_web_sm
```

## Quick Start

### One-liner API

```python
import spellcaster

# Complexity flow analysis (LCX)
result = spellcaster.analyze_complexity("essay_a.txt", "essay_b.txt")
df = result.to_dataframe()

# Valence model (LCVM) — entropy, MI, action frames, multiscale collapse
result = spellcaster.analyze_valence("essay_a.txt", "essay_b.txt")
profile = spellcaster.ValenceModelAnalyzer().build_complexity_profile(result)

# Adaptive evolution (APE) — syntactic species dynamics
result = spellcaster.analyze_evolution("early_draft.txt", "final_draft.txt")
print(result.to_dataframe())

# Keyword structural coherence (KEPM)
result = spellcaster.analyze_keywords(
    "essay.txt",
    keywords=["information", "network"],
)
```

### Full-control API

```python
from spellcaster.analyzers import TextComplexityAnalyzer

analyzer = TextComplexityAnalyzer()
result = analyzer.compare(
    ["Human text here.", "AI text here."],
    labels=["Human", "AI"],
    from_files=False,
)
for flow in result.flows:
    print(f"{flow.label}: {len(flow.sentences)} sentences")
    print(f"  Final k_hist: {flow.sentences[-1].k_hist}")
```

## Analyzers

### TextComplexityAnalyzer (LCX)

Sentence-by-sentence complexity flow via compression (zlib), Levenshtein volatility, and synergy ratios.

```python
from spellcaster.analyzers import TextComplexityAnalyzer

lcx = TextComplexityAnalyzer()
result = lcx.compare(["file_a.txt", "file_b.txt"])
```

**Key metrics:** cumulative compressed size (`k_hist`), edit distance (`volatility`), volatility/marginal-info ratio (`synergy`)

### ValenceModelAnalyzer (LCVM)

The most comprehensive analyzer — ~30 metrics per document across five dimensions:

| Dimension | Metrics |
|-----------|---------|
| **Variation** | Shannon entropy of token distributions |
| **Redundancy** | Multiscale entropy-collapse curves |
| **Organisation** | MI(Verb; Subject), MI(Verb; Object), coupling strength |
| **Repertoire** | Action-frame density, verb diversity |
| **Semantic breadth** | Schema-keyword concentration, valence entropy |

```python
from spellcaster.analyzers import ValenceModelAnalyzer

vm = ValenceModelAnalyzer()
result = vm.analyze(["essay_a.txt", "essay_b.txt"])
profile = vm.build_complexity_profile(result)
print(vm.profile_for_print(profile))
```

### AdaptiveEvolutionAnalyzer (APE)

Treats syntactic structures as biological species competing for "cognitive market share" across document revisions.

```python
from spellcaster.analyzers import AdaptiveEvolutionAnalyzer

ape = AdaptiveEvolutionAnalyzer(use_embeddings=False)  # NCD-only mode
result = ape.analyze(["draft_v1.txt", "draft_v2.txt"])

for species in result.species[:5]:
    print(f"  Group {species.cluster_id}: {species.status.value} "
          f"(Δ={species.delta:+.3f})")
```

### KeywordERPAnalyzer (KEPM)

Analyses structural coherence of keyword usage through POS co-occurrence spectral entropy and NCD-based structural similarity.

```python
from spellcaster.analyzers import KeywordERPAnalyzer

kw = KeywordERPAnalyzer(keywords=["information", "network"])
result = kw.analyze(["essay.txt"])
df = result.to_dataframe()
```

## Export

All results can be exported to CSV or JSON:

```python
from spellcaster.io import export_csv, export_json

export_csv(result, "output.csv")
export_json(result, "output.json")
```

## Result Objects

Every analyzer returns a structured result with:

- **`.to_dataframe()`** — flat pandas DataFrame for analysis
- **Direct attribute access** — full nested data (e.g., `result.posts[0].schema_valence_entropy`)
- **NumPy array properties** — e.g., `flow.k_hist_array`, `flow.volatility_array`

## Architecture

```
spellcaster/
├── analyzers/          # 4 analyzer classes (LCX, LCVM, APE, KEPM)
├── core/               # Shared math: entropy, compression, MI, JS divergence
├── extractors/         # NLP extraction: action frames, noun deps, sentences
├── results/            # Structured dataclass results with .to_dataframe()
├── io/                 # Text loading and result export (CSV, JSON)
├── utils/              # Smoothing, statistics, lazy imports
└── visualization/      # Optional convenience plotting (requires matplotlib)
```

## Requirements

- Python ≥ 3.10
- spaCy with `en_core_web_sm` (or another English model)

## License

MIT
