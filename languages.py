"""
Language Registry & Constants for AI4Bharat Pralekha Dataset
Covers 12 Indic languages + English with full Unicode metadata.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import unicodedata


# ─────────────────────────────────────────────
# Script Unicode Block Definitions
# ─────────────────────────────────────────────
SCRIPT_UNICODE_BLOCKS: Dict[str, List[Tuple[int, int]]] = {
    "Devanagari":  [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "Bengali":     [(0x0980, 0x09FF)],
    "Gurmukhi":    [(0x0A00, 0x0A7F)],
    "Gujarati":    [(0x0A80, 0x0AFF)],
    "Odia":        [(0x0B00, 0x0B7F)],
    "Tamil":       [(0x0B80, 0x0BFF)],
    "Telugu":      [(0x0C00, 0x0C7F)],
    "Kannada":     [(0x0C80, 0x0CFF)],
    "Malayalam":   [(0x0D00, 0x0D7F)],
    "Sinhala":     [(0x0D80, 0x0DFF)],
    "Arabic":      [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "Latin":       [(0x0000, 0x007F), (0x0080, 0x00FF), (0x0100, 0x017F)],
    "Cyrillic":    [(0x0400, 0x04FF)],
    "CJK":         [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
}

SCRIPT_TO_DIRECTION: Dict[str, str] = {
    "Devanagari": "LTR",
    "Bengali":    "LTR",
    "Gurmukhi":   "LTR",
    "Gujarati":   "LTR",
    "Odia":       "LTR",
    "Tamil":      "LTR",
    "Telugu":     "LTR",
    "Kannada":    "LTR",
    "Malayalam":  "LTR",
    "Arabic":     "RTL",
    "Latin":      "LTR",
}


@dataclass
class LanguageInfo:
    code: str                    # Pralekha subset code (ben, hin, ...)
    name: str                    # Full English name
    script: str                  # Primary script name
    iso639_1: Optional[str]      # ISO 639-1 two-letter code
    iso639_3: str                # ISO 639-3 three-letter code
    label_id: int                # Numeric label for ML models
    unicode_ranges: List[Tuple[int, int]] = field(default_factory=list)
    direction: str = "LTR"
    family: str = "Indo-Aryan"
    native_name: str = ""
    romanization_scheme: Optional[str] = None  # IAST, ISO 15919, etc.
    shares_script_with: List[str] = field(default_factory=list)

    def char_in_script(self, char: str) -> bool:
        cp = ord(char)
        return any(lo <= cp <= hi for lo, hi in self.unicode_ranges)


# ─────────────────────────────────────────────
# Language Registry (12 Indic + English)
# ─────────────────────────────────────────────
LANGUAGE_REGISTRY: Dict[str, LanguageInfo] = {
    "ben": LanguageInfo(
        code="ben", name="Bengali", script="Bengali",
        iso639_1="bn", iso639_3="ben", label_id=0,
        unicode_ranges=SCRIPT_UNICODE_BLOCKS["Bengali"],
        direction="LTR", family="Indo-Aryan",
        native_name="বাংলা",
        romanization_scheme="ISO 15919",
    ),
    "eng": LanguageInfo(
        code="eng", name="English", script="Latin",
        iso639_1="en", iso639_3="eng", label_id=1,
        unicode_ranges=SCRIPT_UNICODE_BLOCKS["Latin"],
        direction="LTR", family="Germanic",
        native_name="English",
    ),
    "guj": LanguageInfo(
        code="guj", name="Gujarati", script="Gujarati",
        iso639_1="gu", iso639_3="guj", label_id=2,
        unicode_ranges=SCRIPT_UNICODE_BLOCKS["Gujarati"],
        direction="LTR", family="Indo-Aryan",
        native_name="ગુજરાતી",
        romanization_scheme="ISO 15919",
    ),
    "hin": LanguageInfo(
        code="hin", name="Hindi", script="Devanagari",
        iso639_1="hi", iso639_3="hin", label_id=3,
        unicode_ranges=SCRIPT_UNICODE_BLOCKS["Devanagari"],
        direction="LTR", family="Indo-Aryan",
        native_name="हिन्दी",
        romanization_scheme="IAST",
        shares_script_with=["mar"],
    ),
    "kan": LanguageInfo(
        code="kan", name="Kannada", script="Kannada",
        iso639_1="kn", iso639_3="kan", label_id=4,
        unicode_ranges=SCRIPT_UNICODE_BLOCKS["Kannada"],
        direction="LTR", family="Dravidian",
        native_name="ಕನ್ನಡ",
    ),
    "mal": LanguageInfo(
        code="mal", name="Malayalam", script="Malayalam",
        iso639_1="ml", iso639_3="mal", label_id=5,
        unicode_ranges=SCRIPT_UNICODE_BLOCKS["Malayalam"],
        direction="LTR", family="Dravidian",
        native_name="മലയാളം",
    ),
    "mar": LanguageInfo(
        code="mar", name="Marathi", script="Devanagari",
        iso639_1="mr", iso639_3="mar", label_id=6,
        unicode_ranges=SCRIPT_UNICODE_BLOCKS["Devanagari"],
        direction="LTR", family="Indo-Aryan",
        native_name="मराठी",
        romanization_scheme="IAST",
        shares_script_with=["hin"],
    ),
    "ori": LanguageInfo(
        code="ori", name="Odia", script="Odia",
        iso639_1="or", iso639_3="ori", label_id=7,
        unicode_ranges=SCRIPT_UNICODE_BLOCKS["Odia"],
        direction="LTR", family="Indo-Aryan",
        native_name="ଓଡ଼ିଆ",
    ),
    "pan": LanguageInfo(
        code="pan", name="Punjabi", script="Gurmukhi",
        iso639_1="pa", iso639_3="pan", label_id=8,
        unicode_ranges=SCRIPT_UNICODE_BLOCKS["Gurmukhi"],
        direction="LTR", family="Indo-Aryan",
        native_name="ਪੰਜਾਬੀ",
    ),
    "tam": LanguageInfo(
        code="tam", name="Tamil", script="Tamil",
        iso639_1="ta", iso639_3="tam", label_id=9,
        unicode_ranges=SCRIPT_UNICODE_BLOCKS["Tamil"],
        direction="LTR", family="Dravidian",
        native_name="தமிழ்",
    ),
    "tel": LanguageInfo(
        code="tel", name="Telugu", script="Telugu",
        iso639_1="te", iso639_3="tel", label_id=10,
        unicode_ranges=SCRIPT_UNICODE_BLOCKS["Telugu"],
        direction="LTR", family="Dravidian",
        native_name="తెలుగు",
    ),
    "urd": LanguageInfo(
        code="urd", name="Urdu", script="Arabic",
        iso639_1="ur", iso639_3="urd", label_id=11,
        unicode_ranges=SCRIPT_UNICODE_BLOCKS["Arabic"],
        direction="RTL", family="Indo-Aryan",
        native_name="اردو",
        romanization_scheme="ALA-LC",
    ),
}

# Derived mappings
LABEL_TO_CODE: Dict[int, str] = {v.label_id: k for k, v in LANGUAGE_REGISTRY.items()}
CODE_TO_LABEL: Dict[str, int] = {k: v.label_id for k, v in LANGUAGE_REGISTRY.items()}
ALL_LANGUAGE_CODES: List[str] = sorted(LANGUAGE_REGISTRY.keys())
NUM_CLASSES: int = len(LANGUAGE_REGISTRY)

# Script families for analysis
DRAVIDIAN_LANGUAGES = {"kan", "mal", "tam", "tel"}
INDO_ARYAN_LANGUAGES = {"ben", "guj", "hin", "mar", "ori", "pan", "urd"}
DEVANAGARI_LANGUAGES = {"hin", "mar"}
RTL_LANGUAGES = {"urd"}

# Confusable pairs (share script or similar features)
CONFUSABLE_PAIRS = [
    ("hin", "mar"),   # Both Devanagari
    ("tam", "mal"),   # Related Dravidian scripts
    ("kan", "tel"),   # Related Dravidian scripts
    ("hin", "urd"),   # Same spoken language, different scripts
    ("ben", "ori"),   # Visually similar scripts
]
