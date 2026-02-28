# Changelog

## 0.1.0 (2026-02-28)

Initial release.

### Analyzers

- **TextComplexityAnalyzer (LCX)** — Sentence-level compression dynamics,
  Levenshtein volatility, and cognitive synergy ratios.
- **ValenceModelAnalyzer (LCVM)** — ~30 metrics across variation, redundancy,
  organisation, repertoire, and semantic breadth. Includes multiscale
  entropy-collapse curves, mutual information coupling, and JS divergence.
- **AdaptiveEvolutionAnalyzer (APE)** — NCD-based structural clustering with
  evolutionary dynamics tracking (Emerging/Extinct/Thriving/Declining/Stable).
- **KeywordERPAnalyzer (KEPM)** — Keyword structural coherence via POS
  co-occurrence spectral entropy and EPR-pair detection.

### Core

- Shannon entropy, multiscale window collapse, zlib compression metrics.
- Mutual information, channel capacity, Jensen–Shannon divergence.
- Normalized compression distance and NCD similarity.
- spaCy model management with lazy loading and caching.

### I/O

- Unified text loading from files or strings.
- CSV and JSON export for all result types.

### Results

- Structured dataclass results with `.to_dataframe()` for all analyzers.
- NumPy array properties for efficient numerical access.
- JSON serialisation for the APE evolutionary report format.
