"""
Advanced Text Preprocessing for Multilingual Indic Text
Handles Unicode normalization, noise removal, script detection preprocessing.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from src.utils.languages import SCRIPT_UNICODE_BLOCKS, LANGUAGE_REGISTRY


# ─────────────────────────────────────────────
# Unicode Script Detection
# ─────────────────────────────────────────────

def get_char_script(char: str) -> str:
    """Get Unicode script name for a single character."""
    try:
        cat = unicodedata.category(char)
        name = unicodedata.name(char, "")
        # Extract script from Unicode name (e.g. "DEVANAGARI LETTER A" → "Devanagari")
        parts = name.split()
        if not parts:
            return "Unknown"
        first = parts[0].capitalize()
        # Multi-word scripts
        for script in SCRIPT_UNICODE_BLOCKS:
            if script.upper() in name.upper():
                return script
        return first
    except (ValueError, TypeError):
        return "Unknown"


def get_dominant_script(text: str) -> Tuple[str, Dict[str, float]]:
    """
    Detect the dominant script in text.
    Returns (dominant_script, script_distribution).
    """
    script_counts: Dict[str, int] = {}
    total_alpha = 0

    for char in text:
        if not char.isalpha():
            continue
        total_alpha += 1
        cp = ord(char)

        matched = False
        for script, ranges in SCRIPT_UNICODE_BLOCKS.items():
            for lo, hi in ranges:
                if lo <= cp <= hi:
                    script_counts[script] = script_counts.get(script, 0) + 1
                    matched = True
                    break
            if matched:
                break
        if not matched:
            script_counts["Other"] = script_counts.get("Other", 0) + 1

    if total_alpha == 0:
        return "Unknown", {}

    dist = {s: c / total_alpha for s, c in script_counts.items()}
    dominant = max(dist, key=dist.get) if dist else "Unknown"
    return dominant, dist


# ─────────────────────────────────────────────
# Preprocessor Config
# ─────────────────────────────────────────────

@dataclass
class PreprocessorConfig:
    normalize_unicode: bool = True
    nfc_form: str = "NFC"
    remove_urls: bool = True
    remove_emails: bool = True
    remove_html_tags: bool = True
    normalize_whitespace: bool = True
    remove_zero_width: bool = True
    handle_mixed_numerals: bool = True
    preserve_script_punctuation: bool = True
    lowercase_latin: bool = False
    max_length: Optional[int] = 2000
    min_length: int = 10


# ─────────────────────────────────────────────
# Main Preprocessor
# ─────────────────────────────────────────────

class IndicTextPreprocessor:
    """
    Comprehensive text preprocessor for Indic multilingual corpora.
    Handles Unicode normalization, noise removal, and script-aware cleaning.
    """

    # Regex patterns
    _URL_RE = re.compile(
        r"https?://\S+|www\.\S+|ftp://\S+", re.UNICODE
    )
    _EMAIL_RE = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    )
    _HTML_RE = re.compile(r"<[^>]+>")
    _WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)
    _ZERO_WIDTH_RE = re.compile(
        r"[\u200b\u200c\u200d\u200e\u200f\u202a-\u202e\ufeff\u00ad]"
    )
    _MULTI_PUNCT_RE = re.compile(r"([!?।॥,;.]){2,}")

    # Indic numeral mappings to ASCII (for normalization)
    _NUMERAL_MAP = str.maketrans(
        "".join([
            "०१२३४५६७८९",   # Devanagari
            "০১২৩৪৫৬৭৮৯",   # Bengali
            "੦੧੨੩੪੫੬੭੮੯",   # Gurmukhi
            "૦૧૨૩૪૫૬૭૮૯",   # Gujarati
            "୦୧୨୩୪୫୬୭୮୯",   # Odia
            "௦௧௨௩௪௫௬௭௮௯",   # Tamil
            "౦౧౨౩౪౫౬౭౮౯",   # Telugu
            "೦೧೨೩೪೫೬೭೮೯",   # Kannada
            "൦൧൨൩൪൫൬൭൮൯",   # Malayalam
        ]),
        "0123456789" * 9
    )

    def __init__(self, config: Optional[PreprocessorConfig] = None):
        self.config = config or PreprocessorConfig()

    def preprocess(self, text: str) -> str:
        """Full preprocessing pipeline."""
        if not text or not isinstance(text, str):
            return ""

        cfg = self.config

        # 1. Unicode normalization
        if cfg.normalize_unicode:
            text = unicodedata.normalize(cfg.nfc_form, text)

        # 2. Remove zero-width and invisible characters
        if cfg.remove_zero_width:
            text = self._ZERO_WIDTH_RE.sub("", text)

        # 3. Remove HTML tags
        if cfg.remove_html_tags:
            text = self._HTML_RE.sub(" ", text)

        # 4. Remove URLs
        if cfg.remove_urls:
            text = self._URL_RE.sub(" ", text)

        # 5. Remove emails
        if cfg.remove_emails:
            text = self._EMAIL_RE.sub(" ", text)

        # 6. Numeral normalization (optional)
        if cfg.handle_mixed_numerals:
            text = text.translate(self._NUMERAL_MAP)

        # 7. Collapse multiple punctuation
        text = self._MULTI_PUNCT_RE.sub(r"\1", text)

        # 8. Normalize whitespace
        if cfg.normalize_whitespace:
            text = self._WHITESPACE_RE.sub(" ", text).strip()

        # 9. Lowercase Latin (optional, e.g. for English)
        if cfg.lowercase_latin:
            text = self._selective_lowercase(text)

        # 10. Length truncation
        if cfg.max_length and len(text) > cfg.max_length:
            text = text[:cfg.max_length]

        return text

    def _selective_lowercase(self, text: str) -> str:
        """Lowercase only Latin characters, preserve others."""
        result = []
        for ch in text:
            cp = ord(ch)
            # Latin Basic + Extended
            if 0x0041 <= cp <= 0x005A or 0x00C0 <= cp <= 0x00D6:
                result.append(ch.lower())
            else:
                result.append(ch)
        return "".join(result)

    def batch_preprocess(
        self, texts: List[str], show_progress: bool = True
    ) -> List[str]:
        """Preprocess a batch of texts."""
        if show_progress:
            from tqdm.auto import tqdm
            return [self.preprocess(t) for t in tqdm(texts, desc="Preprocessing")]
        return [self.preprocess(t) for t in texts]

    def quality_score(self, text: str) -> float:
        """
        Returns a text quality score in [0, 1].
        Considers: alpha ratio, script consistency, length adequacy.
        """
        if not text:
            return 0.0

        alpha = sum(c.isalpha() for c in text)
        total = len(text)
        if total == 0:
            return 0.0

        alpha_ratio = alpha / total

        _, script_dist = get_dominant_script(text)
        script_concentration = max(script_dist.values()) if script_dist else 0.0

        length_score = min(1.0, len(text) / 100)

        return 0.3 * alpha_ratio + 0.4 * script_concentration + 0.3 * length_score

    def compute_text_features(self, text: str) -> Dict:
        """Rich feature set for a single text (used in analysis)."""
        dominant, dist = get_dominant_script(text)
        n_chars = len(text)
        n_alpha = sum(c.isalpha() for c in text)
        n_digits = sum(c.isdigit() for c in text)
        n_punct = sum(unicodedata.category(c).startswith("P") for c in text)
        n_spaces = sum(c.isspace() for c in text)
        unique_chars = len(set(text))
        type_token_ratio = unique_chars / max(n_chars, 1)

        # Character entropy
        from collections import Counter as _Counter
        char_counts = _Counter(text)
        total_f = sum(char_counts.values())
        char_entropy = -sum(
            (c / total_f) * np.log2(c / total_f)
            for c in char_counts.values() if c > 0
        ) if total_f > 0 else 0.0

        return {
            "n_chars": n_chars,
            "n_alpha": n_alpha,
            "n_digits": n_digits,
            "n_punct": n_punct,
            "n_spaces": n_spaces,
            "alpha_ratio": n_alpha / max(n_chars, 1),
            "digit_ratio": n_digits / max(n_chars, 1),
            "unique_char_ratio": type_token_ratio,
            "dominant_script": dominant,
            "script_distribution": dist,
            "script_count": len(dist),
            "is_mixed_script": len(dist) > 1 and max(dist.values()) < 0.95,
            "quality_score": self.quality_score(text),
            "n_words": len(text.split()),
            "avg_word_len": (
                np.mean([len(w) for w in text.split()]) if text.split() else 0
            ),
            "char_entropy": float(char_entropy),
        }
