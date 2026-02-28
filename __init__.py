"""
Structured result objects for all Spellcaster analyzers.

Every result class provides a ``.to_dataframe()`` method that produces
a flat :class:`pandas.DataFrame` for exploratory analysis.
"""

from spellcaster.results.complexity import (
    ComplexityComparisonResult,
    ComplexityFlowResult,
    SentenceMetrics,
)
from spellcaster.results.evolution import (
    EvolutionaryStatus,
    EvolutionResult,
    POSComposition,
    SpeciesRecord,
)
from spellcaster.results.keyword import (
    CrossKeywordEntanglement,
    FileKeywordResult,
    KeywordERPResult,
    KeywordMeasures,
)
from spellcaster.results.valence import (
    PostMetrics,
    ValenceModelResult,
)

__all__ = [
    "SentenceMetrics",
    "ComplexityFlowResult",
    "ComplexityComparisonResult",
    "PostMetrics",
    "ValenceModelResult",
    "EvolutionaryStatus",
    "SpeciesRecord",
    "POSComposition",
    "EvolutionResult",
    "KeywordMeasures",
    "CrossKeywordEntanglement",
    "FileKeywordResult",
    "KeywordERPResult",
]
