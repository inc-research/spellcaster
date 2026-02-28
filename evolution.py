"""
Result dataclasses for the Adaptive Evolution (APE) analyzer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


class EvolutionaryStatus(Enum):
    """Classification of a structural species' evolutionary trajectory."""

    EMERGING = "Emerging"
    """New strategy that appeared in a later document."""

    EXTINCT = "Extinct"
    """Strategy present earlier but absent in the latest document."""

    THRIVING = "Thriving"
    """Strategy gaining dominance (positive delta above threshold)."""

    DECLINING = "Declining"
    """Strategy losing dominance (negative delta below threshold)."""

    STABLE = "Stable"
    """Strategy persisting without significant density change."""


@dataclass
class POSComposition:
    """Frequency of a single POS tag within a species cluster."""

    tag: str
    percentage: float
    count: int


@dataclass
class SpeciesRecord:
    """
    A single structural *species* — a cluster of sentences sharing
    similar syntactic structure, tracked across documents.
    """

    cluster_id: int
    status: EvolutionaryStatus
    density_start: float
    """Proportion of sentences in the earliest document."""

    density_end: float
    """Proportion of sentences in the latest document."""

    delta: float
    """``density_end - density_start``."""

    sample_sentence: str
    """A representative sentence from this cluster."""

    pos_composition: list[POSComposition] = field(default_factory=list)
    """Top POS tags in this cluster, with frequencies."""


@dataclass
class EvolutionResult:
    """
    Aggregate result from the Adaptive Evolution analyzer.

    Returned by :meth:`~spellcaster.analyzers.adaptive_evolution.AdaptiveEvolutionAnalyzer.analyze`.
    """

    species: list[SpeciesRecord] = field(default_factory=list)
    """One record per structural cluster."""

    structural_similarity_matrix: np.ndarray | None = None
    """Sentence-level NCD similarity matrix."""

    cluster_assignments: pd.DataFrame | None = None
    """DataFrame with columns ``Sentence``, ``POS_Tags``, ``Source_Document``, ``Group_ID``."""

    document_order: list[str] = field(default_factory=list)
    """Ordered list of source document labels (earliest → latest)."""

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flat DataFrame with one row per species.

        Columns: ``cluster_id``, ``status``, ``density_start``,
        ``density_end``, ``delta``, ``sample_sentence``.
        """
        if not self.species:
            return pd.DataFrame(
                columns=["cluster_id", "status", "density_start",
                         "density_end", "delta", "sample_sentence"]
            )
        return pd.DataFrame(
            {
                "cluster_id": [s.cluster_id for s in self.species],
                "status": [s.status.value for s in self.species],
                "density_start": [s.density_start for s in self.species],
                "density_end": [s.density_end for s in self.species],
                "delta": [s.delta for s in self.species],
                "sample_sentence": [s.sample_sentence for s in self.species],
            }
        )

    def to_json(self) -> list[dict]:
        """
        JSON-serializable list of species records, mirroring the report
        format from the original APE notebook.
        """
        records = []
        for s in self.species:
            records.append({
                "cluster_id": s.cluster_id,
                "classification": {
                    "status": s.status.value,
                    "trend": "Positive" if s.delta > 0 else "Negative",
                },
                "metrics": {
                    "density_start": round(s.density_start, 4),
                    "density_end": round(s.density_end, 4),
                    "selection_pressure_delta": round(s.delta, 4),
                },
                "phenotype": {
                    "sample_structure": s.sample_sentence[:200],
                },
                "pos_composition": [
                    {"tag": pc.tag, "percentage": pc.percentage, "count": pc.count}
                    for pc in s.pos_composition
                ],
            })
        return records
