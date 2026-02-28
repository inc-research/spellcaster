"""
Spellcaster
===========

Analyze natural language text through complex systems science,
information theory, information-theoretic physics analogues,
and evolutionary game theory.

Quick Start
-----------
>>> import spellcaster
>>> result = spellcaster.analyze_complexity("draft_a.txt", "draft_b.txt")
>>> result.to_dataframe()

Analyzers
---------
For full control, import the analyzer classes directly::

    from spellcaster.analyzers import (
        TextComplexityAnalyzer,   # LCX  — compression, volatility, synergy
        ValenceModelAnalyzer,     # LCVM — entropy, MI, action frames, collapse
        AdaptiveEvolutionAnalyzer,# APE  — POS clustering, evolutionary dynamics
        KeywordERPAnalyzer,       # KEPM — keyword structural coherence
    )

Results
-------
All analyzers return structured dataclass results with ``.to_dataframe()``
methods for easy integration with pandas, matplotlib, or any other tooling.

Export
------
>>> from spellcaster.io import export_csv, export_json
>>> export_csv(result, "output.csv")
>>> export_json(result, "output.json")
"""

from spellcaster._version import __version__

from spellcaster.analyzers.complexity_index import TextComplexityAnalyzer
from spellcaster.analyzers.valence_model import ValenceModelAnalyzer
from spellcaster.analyzers.adaptive_evolution import AdaptiveEvolutionAnalyzer
from spellcaster.analyzers.keyword_erp import KeywordERPAnalyzer

from spellcaster.io.readers import load_texts, texts_from_strings
from spellcaster.io.exporters import export_csv, export_json

from spellcaster.results.complexity import (
    ComplexityComparisonResult,
    ComplexityFlowResult,
)
from spellcaster.results.valence import ValenceModelResult
from spellcaster.results.evolution import EvolutionResult
from spellcaster.results.keyword import KeywordERPResult


# ── Convenience functions ────────────────────────────────────────────────────

def analyze_complexity(
    *texts_or_paths: str,
    labels: list[str] | None = None,
    from_files: bool = True,
) -> ComplexityComparisonResult:
    """
    One-liner complexity analysis.

    Parameters
    ----------
    *texts_or_paths : str
        File paths (when *from_files* is True) or raw text strings.
    labels : list[str] or None
        Human-readable labels.
    from_files : bool
        Whether to read from files or treat as raw strings.

    Returns
    -------
    ComplexityComparisonResult
    """
    return TextComplexityAnalyzer().compare(
        list(texts_or_paths), labels=labels, from_files=from_files,
    )


def analyze_valence(
    *texts_or_paths: str,
    labels: list[str] | None = None,
    from_files: bool = True,
    window_sizes: tuple[int, ...] = (25, 50, 100, 250, 500),
) -> ValenceModelResult:
    """
    One-liner valence model analysis.

    Parameters
    ----------
    *texts_or_paths : str
        File paths or raw text strings.
    labels : list[str] or None
        Human-readable labels.
    from_files : bool
        Whether to read from files.
    window_sizes : tuple[int, ...]
        Window sizes for multiscale collapse.

    Returns
    -------
    ValenceModelResult
    """
    return ValenceModelAnalyzer(window_sizes=window_sizes).analyze(
        list(texts_or_paths), labels=labels, from_files=from_files,
    )


def analyze_evolution(
    *texts_or_paths: str,
    labels: list[str] | None = None,
    from_files: bool = True,
    use_embeddings: bool = True,
    alpha_semantic: float = 0.5,
) -> EvolutionResult:
    """
    One-liner adaptive evolution analysis.

    Documents should be in chronological order (earliest first).

    Parameters
    ----------
    *texts_or_paths : str
        File paths or raw text strings.
    labels : list[str] or None
        Human-readable labels.
    from_files : bool
        Whether to read from files.
    use_embeddings : bool
        Whether to use sentence-transformer embeddings.
    alpha_semantic : float
        Blend weight for semantic vs. structural distance.

    Returns
    -------
    EvolutionResult
    """
    return AdaptiveEvolutionAnalyzer(
        use_embeddings=use_embeddings,
        alpha_semantic=alpha_semantic,
    ).analyze(
        list(texts_or_paths), labels=labels, from_files=from_files,
    )


def analyze_keywords(
    *texts_or_paths: str,
    keywords: list[str],
    labels: list[str] | None = None,
    from_files: bool = True,
    context_window: int = 25,
) -> KeywordERPResult:
    """
    One-liner keyword ERP analysis.

    Parameters
    ----------
    *texts_or_paths : str
        File paths or raw text strings.
    keywords : list[str]
        Keywords to analyse.
    labels : list[str] or None
        Human-readable labels.
    from_files : bool
        Whether to read from files.
    context_window : int
        ±N sentences around each keyword mention.

    Returns
    -------
    KeywordERPResult
    """
    return KeywordERPAnalyzer(
        keywords=keywords,
        context_window=context_window,
    ).analyze(
        list(texts_or_paths), labels=labels, from_files=from_files,
    )


__all__ = [
    "__version__",
    "analyze_complexity",
    "analyze_valence",
    "analyze_evolution",
    "analyze_keywords",
    "TextComplexityAnalyzer",
    "ValenceModelAnalyzer",
    "AdaptiveEvolutionAnalyzer",
    "KeywordERPAnalyzer",
    "load_texts",
    "texts_from_strings",
    "export_csv",
    "export_json",
    "ComplexityComparisonResult",
    "ComplexityFlowResult",
    "ValenceModelResult",
    "EvolutionResult",
    "KeywordERPResult",
]
