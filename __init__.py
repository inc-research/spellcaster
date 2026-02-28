"""
Core mathematical and NLP primitives used across all Spellcaster analyzers.
"""

from spellcaster.core.compression import (
    compressed_size,
    ncd_similarity,
    normalized_compression_distance,
)
from spellcaster.core.entropy import (
    multiscale_collapse_curve,
    shannon_entropy,
    summarize_multiscale_collapse,
    window_collapse,
)
from spellcaster.core.information import (
    channel_capacity,
    js_distance_from_counters,
    js_divergence_from_counters,
    js_divergence_matrix,
    mutual_information,
)
from spellcaster.core.nlp import clear_model_cache, get_nlp, tokenize

__all__ = [
    # entropy
    "shannon_entropy",
    "window_collapse",
    "multiscale_collapse_curve",
    "summarize_multiscale_collapse",
    # compression
    "compressed_size",
    "normalized_compression_distance",
    "ncd_similarity",
    # information
    "mutual_information",
    "channel_capacity",
    "js_divergence_from_counters",
    "js_distance_from_counters",
    "js_divergence_matrix",
    # nlp
    "get_nlp",
    "clear_model_cache",
    "tokenize",
]
