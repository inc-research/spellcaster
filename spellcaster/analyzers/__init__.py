"""
Spellcaster analyzers.

Each analyzer encapsulates a complete analytical pipeline, from text
input to structured result objects.
"""

from spellcaster.analyzers.complexity_index import TextComplexityAnalyzer
from spellcaster.analyzers.valence_model import ValenceModelAnalyzer
from spellcaster.analyzers.adaptive_evolution import AdaptiveEvolutionAnalyzer
from spellcaster.analyzers.keyword_erp import KeywordERPAnalyzer

__all__ = [
    "TextComplexityAnalyzer",
    "ValenceModelAnalyzer",
    "AdaptiveEvolutionAnalyzer",
    "KeywordERPAnalyzer",
]
