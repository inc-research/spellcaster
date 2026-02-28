"""
Text loading utilities.

Provides a uniform way to load documents from file paths or raw strings,
producing :class:`TextDocument` instances that all analyzers accept.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass


@dataclass
class TextDocument:
    """
    A loaded text document with provenance metadata.

    Attributes
    ----------
    path : str
        Original file path, or ``"<inline>"`` for text supplied directly.
    label : str
        Human-readable label (defaults to the filename stem).
    text : str
        Full text content.
    """

    path: str
    label: str
    text: str


def load_texts(
    file_paths: list[str],
    labels: list[str] | None = None,
    encoding: str = "utf-8",
) -> list[TextDocument]:
    """
    Load text files from disk and wrap them as :class:`TextDocument` objects.

    Parameters
    ----------
    file_paths : list[str]
        Paths to ``.txt`` files.
    labels : list[str] or None
        Human-readable labels, one per file.  When ``None``, labels
        default to the file-name stems (e.g. ``"essay1"`` for
        ``"/data/essay1.txt"``).
    encoding : str
        Text encoding to use when reading files.

    Returns
    -------
    list[TextDocument]

    Raises
    ------
    FileNotFoundError
        If any file does not exist.
    ValueError
        If *labels* is provided but its length differs from *file_paths*.
    """
    if labels is not None and len(labels) != len(file_paths):
        raise ValueError(
            f"labels length ({len(labels)}) must match "
            f"file_paths length ({len(file_paths)})"
        )

    documents: list[TextDocument] = []
    for i, fp in enumerate(file_paths):
        p = pathlib.Path(fp)
        if not p.exists():
            raise FileNotFoundError(f"Text file not found: {fp}")

        text = p.read_text(encoding=encoding, errors="ignore")
        label = labels[i] if labels is not None else p.stem

        documents.append(TextDocument(path=str(fp), label=label, text=text))

    return documents


def texts_from_strings(
    texts: list[str],
    labels: list[str] | None = None,
) -> list[TextDocument]:
    """
    Wrap raw strings as :class:`TextDocument` objects (no file I/O).

    Useful when text is already in memory — for example, from a
    database or an API response.

    Parameters
    ----------
    texts : list[str]
        Raw text strings.
    labels : list[str] or None
        Human-readable labels.  When ``None``, defaults to
        ``"text_0"``, ``"text_1"``, etc.

    Returns
    -------
    list[TextDocument]
    """
    if labels is not None and len(labels) != len(texts):
        raise ValueError(
            f"labels length ({len(labels)}) must match "
            f"texts length ({len(texts)})"
        )

    documents: list[TextDocument] = []
    for i, text in enumerate(texts):
        label = labels[i] if labels is not None else f"text_{i}"
        documents.append(TextDocument(path="<inline>", label=label, text=text))

    return documents
