"""
Compression-based complexity measures.

Uses ``zlib`` (LZ77-family) as an approximation of Kolmogorov complexity.
Provides raw compressed size and the Normalized Compression Distance (NCD)
for comparing structural similarity between two sequences.
"""

from __future__ import annotations

import zlib


def compressed_size(text: str) -> int:
    """
    Return the byte-length of *text* after zlib compression.

    This serves as a practical upper-bound proxy for Kolmogorov complexity:
    more compressible text → lower complexity.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    int
        Size in bytes of the zlib-compressed UTF-8 encoding.
        Returns ``0`` for empty input.
    """
    if not text:
        return 0
    return len(zlib.compress(text.encode("utf-8")))


def normalized_compression_distance(
    seq1: list[str],
    seq2: list[str],
) -> float:
    """
    Compute the Normalized Compression Distance (NCD) between two token sequences.

    NCD is an approximation of normalized information distance based on
    Kolmogorov complexity.  Lower NCD means the two sequences share more
    structural patterns.

    .. math::
        \\text{NCD}(x, y) = \\frac{C(xy) - \\min(C(x), C(y))}{\\max(C(x), C(y))}

    Parameters
    ----------
    seq1, seq2 : list[str]
        Token sequences (e.g. POS tags).

    Returns
    -------
    float
        NCD value in [0, 1].  0 = identical structure, 1 = maximally distinct.
    """
    if not seq1 and not seq2:
        return 0.0

    s1 = " ".join(seq1).encode("utf-8")
    s2 = " ".join(seq2).encode("utf-8")

    if not s1 or not s2:
        return 1.0  # One empty, one not → maximally distinct

    c_x = len(zlib.compress(s1))
    c_y = len(zlib.compress(s2))
    # Average both concatenation orders to ensure symmetry
    # (zlib's LZ77 window introduces order-dependent bias on short inputs)
    c_xy = len(zlib.compress(s1 + b" " + s2))
    c_yx = len(zlib.compress(s2 + b" " + s1))
    c_concat = (c_xy + c_yx) / 2.0

    max_c = max(c_x, c_y)
    if max_c == 0:
        return 0.0

    ncd = (c_concat - min(c_x, c_y)) / max_c
    return max(0.0, min(1.0, ncd))


def ncd_similarity(seq1: list[str], seq2: list[str]) -> float:
    """
    Structural similarity score: ``1 - NCD``.

    A convenience wrapper that returns 1.0 for identical structure and
    0.0 for maximally distinct structure.

    Parameters
    ----------
    seq1, seq2 : list[str]
        Token sequences (e.g. POS tags).

    Returns
    -------
    float
        Similarity in [0, 1].
    """
    return 1.0 - normalized_compression_distance(seq1, seq2)
