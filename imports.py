"""Lazy import helpers for optional dependencies."""

from spellcaster.exceptions import OptionalDependencyError


def require_matplotlib():
    """Import and return matplotlib.pyplot, or raise with install instructions."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise OptionalDependencyError("matplotlib", "viz")


def require_seaborn():
    """Import and return seaborn, or raise with install instructions."""
    try:
        import seaborn as sns
        return sns
    except ImportError:
        raise OptionalDependencyError("seaborn", "viz")


def require_networkx():
    """Import and return networkx, or raise with install instructions."""
    try:
        import networkx as nx
        return nx
    except ImportError:
        raise OptionalDependencyError("networkx", "graphs")


def require_sentence_transformers():
    """Import and return SentenceTransformer, or raise with install instructions."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError:
        raise OptionalDependencyError("sentence-transformers", "ml")


def require_sklearn():
    """Import and return sklearn, or raise with install instructions."""
    try:
        import sklearn
        return sklearn
    except ImportError:
        raise OptionalDependencyError("scikit-learn", "ml")
