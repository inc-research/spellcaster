# Spellcaster LCX Index

This library exposes text complexity and flow analysis utilities that were
previously authored as a notebook. The primary entry points are
`TextComplexityAnalyzer` and `smooth`.

## Installation

```bash
pip install spellcaster-lcx-index
```

## Usage

```python
from spellcaster_lcx_index import TextComplexityAnalyzer, smooth

analyzer = TextComplexityAnalyzer()
results = analyzer.analyze_flow("First sentence. Second sentence.")

print(results.k_hist)
print(results.volatility)
print(smooth(results.synergy, window=4))
```

## Notes

* The analyzer strips simple HTML tags before processing.
* Sentences are split on periods, newlines, or bullet points.
