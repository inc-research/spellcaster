"""
Export utilities for Spellcaster results.

Provides generic serialisation of any result object to JSON or CSV,
plus a specialised APE evolutionary-dynamics JSON report.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from spellcaster.results.evolution import EvolutionResult

logger = logging.getLogger(__name__)


def export_csv(
    result,
    path: str,
    **kwargs,
) -> Path:
    """
    Export any result with a ``.to_dataframe()`` method to CSV.

    Parameters
    ----------
    result
        Any Spellcaster result object (``ComplexityComparisonResult``,
        ``ValenceModelResult``, ``EvolutionResult``, ``KeywordERPResult``).
    path : str
        Output file path.
    **kwargs
        Forwarded to :meth:`pandas.DataFrame.to_csv`.

    Returns
    -------
    Path
        The written file path.
    """
    df = result.to_dataframe()
    p = Path(path)
    df.to_csv(p, index=kwargs.pop("index", False), **kwargs)
    logger.info("Exported %d rows to %s", len(df), p)
    return p


def export_json(
    result,
    path: str,
    indent: int = 2,
) -> Path:
    """
    Export a result to JSON.

    For :class:`~spellcaster.results.evolution.EvolutionResult`, uses the
    structured ``to_json()`` format.  For all other result types, converts
    the ``.to_dataframe()`` output to JSON records.

    Parameters
    ----------
    result
        Any Spellcaster result object.
    path : str
        Output file path.
    indent : int
        JSON indentation.

    Returns
    -------
    Path
        The written file path.
    """
    p = Path(path)

    if isinstance(result, EvolutionResult):
        data = result.to_json()
    elif hasattr(result, "to_dataframe"):
        df = result.to_dataframe()
        data = json.loads(df.to_json(orient="records", default_handler=str))
    else:
        raise TypeError(
            f"Cannot export {type(result).__name__}: "
            "expected a Spellcaster result with .to_dataframe() or .to_json()"
        )

    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)

    logger.info("Exported JSON to %s", p)
    return p
