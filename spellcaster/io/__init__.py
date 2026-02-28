"""
I/O utilities: text loading and result export.
"""

from spellcaster.io.readers import TextDocument, load_texts, texts_from_strings
from spellcaster.io.exporters import export_csv, export_json

__all__ = [
    "TextDocument",
    "load_texts",
    "texts_from_strings",
    "export_csv",
    "export_json",
]
