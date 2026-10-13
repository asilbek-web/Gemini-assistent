"""Rough language hint for Uzbek / Russian / English.

This is only a hint for Claude; the system prompt asks it to answer in the
language (and script) the correspondent used. Returns "uz", "ru", "en" or
None when the text has no letters (emoji, numbers).
"""
from __future__ import annotations

import re
from typing import Optional

_CYRILLIC = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюяё")
_LATIN = set("abcdefghijklmnopqrstuvwxyz")

# Letters that exist only in Uzbek Cyrillic.
_UZ_CYRILLIC_LETTERS = set("ўқғҳ")

# Uzbek Cyrillic words that are hard to mistake for Russian.
_UZ_CYRILLIC_WORDS = (
    "салом",
    "рахмат",
    "яхши",
    "нима",
    "қалесан",
    "йўқ",
    "бугун",
    "эртага",
    "керак",
    "ассалому",
)

# Apostrophe variants used for the Uzbek o'/g' sounds.
_APOSTROPHES = "'ʻ’‘`"

# English contractions that must not count as the Uzbek o'/g' apostrophe.
_ENGLISH_CONTRACTIONS = ("dog's", "o'clock")

_UZ_APOSTROPHE_RE = re.compile(r"\b[og][" + _APOSTROPHES + r"]", re.IGNORECASE)

_UZ_WORDS = (
    "salom",
    "rahmat",
    "yaxshi",
    "nima",
    "qalesan",
    "qayerda",
    "bugun",
    "ertaga",
    "kerak",
    "xop",
    "bopti",
    "zor",
    "hozir",
    "keyin",
    "men",
    "sen",
    "siz",
)

_UZ_SUFFIXES = (
    "misan",
    "misiz",
    "mizmi",
    "yapsan",
    "yapman",
    "yapti",
    "yapmiz",
    "asizmi",
    "ganman",
    "dingmi",
)


def _strip_english_contractions(text: str) -> str:
    lowered = text.lower()
    for contraction in _ENGLISH_CONTRACTIONS:
        lowered = lowered.replace(contraction, " ")
    return lowered


def detect_language(text: str) -> Optional[str]:
    """Return "uz", "ru", "en" or None (no letters, e.g. emoji)."""
    if not text:
        return None

    letters = [ch.lower() for ch in text if ch.isalpha()]
    if not letters:
        return None

    cyrillic = sum(1 for ch in letters if ch in _CYRILLIC or ch in _UZ_CYRILLIC_LETTERS)
    latin = sum(1 for ch in letters if ch in _LATIN)

    if cyrillic > latin:
        lowered = text.lower()
        if any(ch in _UZ_CYRILLIC_LETTERS for ch in lowered):
            return "uz"
        if any(word in lowered for word in _UZ_CYRILLIC_WORDS):
            return "uz"
        return "ru"

    if latin:
        cleaned = _strip_english_contractions(text)
        if _UZ_APOSTROPHE_RE.search(cleaned):
            return "uz"
        words = re.findall(r"[a-zA-Z']+", cleaned)
        for word in words:
            bare = word.lower().strip(_APOSTROPHES)
            if bare in _UZ_WORDS:
                return "uz"
            if bare.endswith(_UZ_SUFFIXES):
                return "uz"
        return "en"

    return None
