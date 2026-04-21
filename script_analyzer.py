"""
Advanced Script Analysis Engine
Deep Unicode-based analysis: script distribution, mixing, directionality,
transliteration detection, noise analysis, and linguistic features.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger

from src.utils.languages import (
    SCRIPT_UNICODE_BLOCKS,
    SCRIPT_TO_DIRECTION,
    LANGUAGE_REGISTRY,
)


# ─────────────────────────────────────────────
# Unicode Block Database
# ─────────────────────────────────────────────

UNICODE_BLOCK_NAMES: List[Tuple[int, int, str]] = [
    (0x0000, 0x007F, "Basic Latin"),
    (0x0080, 0x00FF, "Latin-1 Supplement"),
    (0x0100, 0x017F, "Latin Extended-A"),
    (0x0180, 0x024F, "Latin Extended-B"),
    (0x0250, 0x02AF, "IPA Extensions"),
    (0x0300, 0x036F, "Combining Diacritical Marks"),
    (0x0370, 0x03FF, "Greek"),
    (0x0400, 0x04FF, "Cyrillic"),
    (0x0500, 0x052F, "Cyrillic Supplement"),
    (0x0600, 0x06FF, "Arabic"),
    (0x0700, 0x074F, "Syriac"),
    (0x0750, 0x077F, "Arabic Supplement"),
    (0x0900, 0x097F, "Devanagari"),
    (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Gurmukhi"),
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Oriya"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0D80, 0x0DFF, "Sinhala"),
    (0x0E00, 0x0E7F, "Thai"),
    (0x0F00, 0x0FFF, "Tibetan"),
    (0x1000, 0x109F, "Myanmar"),
    (0x10A0, 0x10FF, "Georgian"),
    (0x1100, 0x11FF, "Hangul Jamo"),
    (0x1E00, 0x1EFF, "Latin Extended Additional"),
    (0x2000, 0x206F, "General Punctuation"),
    (0x20A0, 0x20CF, "Currency Symbols"),
    (0x2100, 0x214F, "Letterlike Symbols"),
    (0x2200, 0x22FF, "Mathematical Operators"),
    (0x3000, 0x303F, "CJK Symbols and Punctuation"),
    (0x3040, 0x309F, "Hiragana"),
    (0x30A0, 0x30FF, "Katakana"),
    (0x4E00, 0x9FFF, "CJK Unified Ideographs"),
    (0xFB00, 0xFB4F, "Alphabetic Presentation Forms"),
    (0xFB50, 0xFDFF, "Arabic Presentation Forms-A"),
    (0xFE70, 0xFEFF, "Arabic Presentation Forms-B"),
    (0xFF00, 0xFFEF, "Halfwidth and Fullwidth Forms"),
]


def get_unicode_block(cp: int) -> str:
    for lo, hi, name in UNICODE_BLOCK_NAMES:
        if lo <= cp <= hi:
            return name
    return f"U+{cp:04X} (unknown block)"


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class CharacterAnalysis:
    char: str
    codepoint: int
    unicode_name: str
    unicode_category: str
    unicode_block: str
    script: Optional[str]
    is_alpha: bool
    is_digit: bool
    is_punct: bool
    is_combining: bool


@dataclass
class ScriptSegment:
    script: str
    start: int
    end: int
    text: str
    char_count: int


@dataclass
class ScriptAnalysisResult:
    text: str
    total_chars: int
    alpha_chars: int
    digit_chars: int
    punct_chars: int
    space_chars: int
    other_chars: int

    # Script distribution
    script_distribution: Dict[str, float]
    dominant_script: str
    dominant_script_ratio: float

    # Mixing analysis
    is_mixed_script: bool
    mixing_score: float           # 0=pure, 1=maximally mixed
    script_segments: List[ScriptSegment]
    script_transitions: int

    # Directionality
    primary_direction: str
    has_bidi: bool
    rtl_ratio: float

    # Noise indicators
    noise_score: float
    has_zero_width: bool
    has_control_chars: bool
    has_excessive_punct: bool
    url_count: int
    emoji_count: int
    latin_ratio: float            # in otherwise Indic text → potential transliteration

    # Unicode statistics
    unicode_block_distribution: Dict[str, float]
    unique_chars: int
    char_entropy: float

    # Language hints
    script_to_lang_hints: Dict[str, List[str]]

    def to_dict(self) -> Dict:
        return {
            "total_chars": self.total_chars,
            "alpha_chars": self.alpha_chars,
            "alpha_ratio": self.alpha_chars / max(self.total_chars, 1),
            "dominant_script": self.dominant_script,
            "dominant_script_ratio": self.dominant_script_ratio,
            "is_mixed_script": self.is_mixed_script,
            "mixing_score": self.mixing_score,
            "script_transitions": self.script_transitions,
            "primary_direction": self.primary_direction,
            "has_bidi": self.has_bidi,
            "noise_score": self.noise_score,
            "char_entropy": self.char_entropy,
            "script_distribution": self.script_distribution,
            "script_to_lang_hints": self.script_to_lang_hints,
        }


# ─────────────────────────────────────────────
# Script Analyser
# ─────────────────────────────────────────────

class ScriptAnalyzer:
    """
    Deep script and Unicode analysis for multilingual text.
    Designed for Indic languages with mixed-script handling.
    """

    _URL_RE = re.compile(r"https?://\S+|www\.\S+")
    _EMOJI_RE = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        re.UNICODE
    )
    _ZW_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff\u00ad]")
    _CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

    # Script → language hints
    SCRIPT_LANG_HINTS: Dict[str, List[str]] = {
        "Devanagari": ["hin", "mar"],
        "Bengali": ["ben"],
        "Gurmukhi": ["pan"],
        "Gujarati": ["guj"],
        "Oriya": ["ori"],
        "Tamil": ["tam"],
        "Telugu": ["tel"],
        "Kannada": ["kan"],
        "Malayalam": ["mal"],
        "Arabic": ["urd"],
        "Latin": ["eng"],
    }

    def analyze(self, text: str) -> ScriptAnalysisResult:
        """Comprehensive script analysis of input text."""
        if not text:
            return self._empty_result(text)

        chars = list(text)
        n = len(chars)

        # Character category counts
        alpha = sum(c.isalpha() for c in chars)
        digits = sum(c.isdigit() for c in chars)
        spaces = sum(c.isspace() for c in chars)
        puncts = sum(unicodedata.category(c).startswith("P") for c in chars)
        other = n - alpha - digits - spaces - puncts

        # Script assignment per alpha char
        script_of: List[Optional[str]] = [self._assign_script(c) for c in chars]
        alpha_scripts = [s for c, s in zip(chars, script_of) if c.isalpha() and s]

        # Script distribution
        script_counter = Counter(alpha_scripts)
        total_alpha = max(len(alpha_scripts), 1)
        script_dist = {s: c / total_alpha for s, c in script_counter.most_common()}

        dominant = script_counter.most_common(1)[0][0] if script_counter else "Unknown"
        dom_ratio = script_dist.get(dominant, 0.0)

        # Segment extraction
        segments = self._extract_segments(chars, script_of)
        transitions = max(0, len(segments) - 1)

        # Mixing score: entropy-based
        mixing_score = self._script_entropy(script_dist)

        # Directionality
        rtl_langs = {"Arabic", "Hebrew"}
        rtl_chars = sum(1 for s in alpha_scripts if s in rtl_langs)
        rtl_ratio = rtl_chars / max(total_alpha, 1)
        primary_direction = "RTL" if rtl_ratio > 0.5 else "LTR"
        has_bidi = 0.05 < rtl_ratio < 0.95

        # Unicode block distribution
        block_counter: Counter = Counter()
        for c in chars:
            block = get_unicode_block(ord(c))
            block_counter[block] += 1
        block_dist = {b: c / n for b, c in block_counter.most_common(10)}

        # Noise analysis
        url_count = len(self._URL_RE.findall(text))
        emoji_count = len(self._EMOJI_RE.findall(text))
        has_zero_width = bool(self._ZW_RE.search(text))
        has_control = bool(self._CTRL_RE.search(text))
        has_excessive_punct = (puncts / max(n, 1)) > 0.15
        latin_ratio = script_dist.get("Basic Latin", 0.0) + script_dist.get("Latin", 0.0)

        noise_components = [
            0.3 * min(1.0, url_count * 0.5),
            0.2 * int(has_zero_width),
            0.2 * int(has_control),
            0.15 * int(has_excessive_punct),
            0.15 * min(1.0, emoji_count * 0.2),
        ]
        noise_score = sum(noise_components)

        # Char entropy
        char_freq = Counter(text)
        total_f = sum(char_freq.values())
        entropy = -sum(
            (c / total_f) * np.log2(c / total_f)
            for c in char_freq.values() if c > 0
        )

        # Lang hints based on detected scripts
        lang_hints: Dict[str, List[str]] = {
            s: self.SCRIPT_LANG_HINTS.get(s, [])
            for s in script_dist
        }

        return ScriptAnalysisResult(
            text=text,
            total_chars=n,
            alpha_chars=alpha,
            digit_chars=digits,
            punct_chars=puncts,
            space_chars=spaces,
            other_chars=other,
            script_distribution=script_dist,
            dominant_script=dominant,
            dominant_script_ratio=dom_ratio,
            is_mixed_script=len(script_dist) > 1 and dom_ratio < 0.95,
            mixing_score=mixing_score,
            script_segments=segments,
            script_transitions=transitions,
            primary_direction=primary_direction,
            has_bidi=has_bidi,
            rtl_ratio=rtl_ratio,
            noise_score=noise_score,
            has_zero_width=has_zero_width,
            has_control_chars=has_control,
            has_excessive_punct=has_excessive_punct,
            url_count=url_count,
            emoji_count=emoji_count,
            latin_ratio=latin_ratio,
            unicode_block_distribution=block_dist,
            unique_chars=len(set(text)),
            char_entropy=float(entropy),
            script_to_lang_hints=lang_hints,
        )

    def analyze_batch(self, texts: List[str]) -> List[ScriptAnalysisResult]:
        return [self.analyze(t) for t in texts]

    def _assign_script(self, char: str) -> Optional[str]:
        if not char.isalpha():
            return None
        cp = ord(char)
        for block_start, block_end, block_name in UNICODE_BLOCK_NAMES:
            if block_start <= cp <= block_end:
                return block_name
        return "Unknown"

    def _extract_segments(
        self, chars: List[str], scripts: List[Optional[str]]
    ) -> List[ScriptSegment]:
        """Extract contiguous script segments from text."""
        segments = []
        if not chars:
            return segments

        current_script = None
        start = 0
        seg_chars = []

        for i, (c, s) in enumerate(zip(chars, scripts)):
            if not c.isalpha():
                seg_chars.append(c)
                continue
            if s != current_script:
                if current_script is not None and seg_chars:
                    segments.append(ScriptSegment(
                        script=current_script,
                        start=start,
                        end=i,
                        text="".join(seg_chars),
                        char_count=sum(1 for ch in seg_chars if ch.isalpha()),
                    ))
                current_script = s
                start = i
                seg_chars = [c]
            else:
                seg_chars.append(c)

        if current_script and seg_chars:
            segments.append(ScriptSegment(
                script=current_script,
                start=start,
                end=len(chars),
                text="".join(seg_chars),
                char_count=sum(1 for ch in seg_chars if ch.isalpha()),
            ))

        return [s for s in segments if s.char_count >= 3]

    def _script_entropy(self, dist: Dict[str, float]) -> float:
        """Shannon entropy over script distribution (normalized to [0,1])."""
        if len(dist) <= 1:
            return 0.0
        probs = list(dist.values())
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        max_entropy = np.log2(len(probs))
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    def _empty_result(self, text: str) -> ScriptAnalysisResult:
        return ScriptAnalysisResult(
            text=text, total_chars=0, alpha_chars=0, digit_chars=0,
            punct_chars=0, space_chars=0, other_chars=0,
            script_distribution={}, dominant_script="Unknown",
            dominant_script_ratio=0.0, is_mixed_script=False,
            mixing_score=0.0, script_segments=[], script_transitions=0,
            primary_direction="LTR", has_bidi=False, rtl_ratio=0.0,
            noise_score=0.0, has_zero_width=False, has_control_chars=False,
            has_excessive_punct=False, url_count=0, emoji_count=0,
            latin_ratio=0.0, unicode_block_distribution={},
            unique_chars=0, char_entropy=0.0, script_to_lang_hints={},
        )

    def compare_texts(self, text1: str, text2: str) -> Dict:
        """Compare script profiles of two texts."""
        r1 = self.analyze(text1)
        r2 = self.analyze(text2)

        scripts1 = set(r1.script_distribution.keys())
        scripts2 = set(r2.script_distribution.keys())
        shared = scripts1 & scripts2

        jaccard = len(shared) / max(len(scripts1 | scripts2), 1)

        return {
            "script_overlap": list(shared),
            "jaccard_similarity": jaccard,
            "dominant_scripts_match": r1.dominant_script == r2.dominant_script,
            "text1_dominant": r1.dominant_script,
            "text2_dominant": r2.dominant_script,
            "text1_mixing_score": r1.mixing_score,
            "text2_mixing_score": r2.mixing_score,
        }

    def detect_transliteration(self, text: str) -> Dict:
        """
        Detect if text is a transliteration of an Indic language
        written in Latin script (Romanized Indic).
        """
        result = self.analyze(text)

        # If predominantly Latin but contains Indic vocabulary patterns
        latin_dominated = result.dominant_script in ("Basic Latin", "Latin Extended-A")
        has_indic_patterns = self._has_indic_romanization_patterns(text)

        # Common Romanization markers
        has_aspirates = bool(re.search(r"\b\w*(kh|gh|ch|jh|th|dh|ph|bh)\w*\b", text, re.I))
        has_retroflex = bool(re.search(r"\b\w*[ṭḍṇṣ]\w*\b", text))
        has_nasal_markers = bool(re.search(r"\b\w*(ng|ny|ñ|ṅ)\w*\b", text, re.I))

        is_likely_transliteration = (
            latin_dominated and (has_indic_patterns or has_aspirates)
        )

        return {
            "is_transliteration": is_likely_transliteration,
            "confidence": 0.8 if (is_likely_transliteration and has_aspirates) else 0.5,
            "evidence": {
                "latin_dominated": latin_dominated,
                "has_indic_patterns": has_indic_patterns,
                "has_aspirates": has_aspirates,
                "has_retroflex": has_retroflex,
                "has_nasal_markers": has_nasal_markers,
            }
        }

    def _has_indic_romanization_patterns(self, text: str) -> bool:
        """Common romanization patterns across Indic languages."""
        patterns = [
            r"\b(hai|hain|nahin|nahi|aur|yeh|woh|kya|kaun|kahan|kyun|mera|tera|apna)\b",  # Hindi
            r"\b(amma|appa|anna|akka|vanakkam|nandri|romba|paaru|paar)\b",   # Tamil
            r"\b(ahe|aahe|nahi|tumhi|amhi|tya|tyanche|mhanje)\b",           # Marathi
        ]
        for p in patterns:
            if re.search(p, text, re.I):
                return True
        return False


# Singleton for reuse
default_analyzer = ScriptAnalyzer()
