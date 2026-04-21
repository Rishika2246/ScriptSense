"""
Comprehensive Unit Tests
Tests for preprocessor, script analyzer, models, and API.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd

from src.data.preprocessor import IndicTextPreprocessor, PreprocessorConfig, get_dominant_script
from src.analysis.script_analyzer import ScriptAnalyzer
from src.utils.languages import (
    LANGUAGE_REGISTRY,
    CODE_TO_LABEL,
    LABEL_TO_CODE,
    ALL_LANGUAGE_CODES,
    NUM_CLASSES,
)
from src.models.ensemble_model import UnicodeHeuristicModel, EnsembleLIDModel


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_TEXTS = {
    "hin": "भारत एक विशाल देश है जहाँ अनेक भाषाएँ बोली जाती हैं।",
    "ben": "বাংলা ভাষা ও সাহিত্যের ঐতিহ্য অনেক প্রাচীন।",
    "tam": "தமிழ் மொழி மிகவும் பழமையான மொழியாகும்.",
    "tel": "తెలుగు భాష చాలా అందమైన భాష.",
    "mal": "മലയാളം ഒരു ദ്രാവിഡ ഭാഷയാണ്.",
    "kan": "ಕನ್ನಡ ಭಾಷೆ ಕರ್ನಾಟಕದ ರಾಜ್ಯ ಭಾಷೆ.",
    "guj": "ગુજરાત ભારતનું એક મહત્ત્વનું રાજ્ય છે.",
    "mar": "मराठी महाराष्ट्राची राजभाषा आहे.",
    "pan": "ਪੰਜਾਬੀ ਉੱਤਰ ਭਾਰਤ ਵਿੱਚ ਬੋਲੀ ਜਾਂਦੀ ਭਾਸ਼ਾ ਹੈ।",
    "urd": "اردو ایک خوبصورت زبان ہے۔",
    "ori": "ଓଡ଼ିଆ ଓଡ଼ିଶାର ରାଜ୍ୟ ଭାଷା।",
    "eng": "The diversity of languages in India is remarkable.",
}


@pytest.fixture
def preprocessor():
    return IndicTextPreprocessor()


@pytest.fixture
def script_analyzer():
    return ScriptAnalyzer()


@pytest.fixture
def heuristic_model():
    return UnicodeHeuristicModel()


# ── Language Registry Tests ────────────────────────────────────────────────────

class TestLanguageRegistry:
    def test_all_languages_present(self):
        assert len(LANGUAGE_REGISTRY) == 12

    def test_expected_codes(self):
        expected = {"ben", "eng", "guj", "hin", "kan", "mal", "mar", "ori", "pan", "tam", "tel", "urd"}
        assert set(LANGUAGE_REGISTRY.keys()) == expected

    def test_label_mapping_complete(self):
        assert len(CODE_TO_LABEL) == NUM_CLASSES
        assert len(LABEL_TO_CODE) == NUM_CLASSES

    def test_label_bijection(self):
        for code, label in CODE_TO_LABEL.items():
            assert LABEL_TO_CODE[label] == code

    def test_unicode_ranges_valid(self):
        for code, info in LANGUAGE_REGISTRY.items():
            for lo, hi in info.unicode_ranges:
                assert 0 <= lo < hi <= 0xFFFF or hi <= 0x10FFFF, f"Invalid range for {code}"

    def test_rtl_languages(self):
        urd = LANGUAGE_REGISTRY["urd"]
        assert urd.direction == "RTL"

    def test_dravidian_scripts_unique(self):
        dravidian = ["kan", "mal", "tam", "tel"]
        scripts = [LANGUAGE_REGISTRY[l].script for l in dravidian]
        assert len(set(scripts)) == 4, "Dravidian languages should have unique scripts"

    def test_devanagari_shared(self):
        hin = LANGUAGE_REGISTRY["hin"]
        mar = LANGUAGE_REGISTRY["mar"]
        assert hin.script == mar.script == "Devanagari"


# ── Preprocessor Tests ────────────────────────────────────────────────────────

class TestPreprocessor:
    def test_url_removal(self, preprocessor):
        text = "Visit https://example.com for more info"
        result = preprocessor.preprocess(text)
        assert "https" not in result
        assert "example.com" not in result

    def test_html_removal(self, preprocessor):
        text = "<p>Hello <b>World</b></p>"
        result = preprocessor.preprocess(text)
        assert "<p>" not in result
        assert "<b>" not in result
        assert "Hello" in result and "World" in result

    def test_whitespace_normalization(self, preprocessor):
        text = "hello    world\t\nfoo"
        result = preprocessor.preprocess(text)
        assert "  " not in result

    def test_zero_width_removal(self, preprocessor):
        text = "hello\u200bworld"
        result = preprocessor.preprocess(text)
        assert "\u200b" not in result

    def test_unicode_normalization(self, preprocessor):
        # NFC normalization should combine precomposed chars
        text = "cafe\u0301"  # e + combining accent
        result = preprocessor.preprocess(text)
        assert len(result) <= len(text)

    def test_preserves_indic_chars(self, preprocessor):
        text = "नमस्ते दुनिया"
        result = preprocessor.preprocess(text)
        assert "नमस्ते" in result

    def test_quality_score_range(self, preprocessor):
        for text in SAMPLE_TEXTS.values():
            score = preprocessor.quality_score(text)
            assert 0.0 <= score <= 1.0

    def test_quality_score_good_text(self, preprocessor):
        score = preprocessor.quality_score("नमस्ते दुनिया, यह एक अच्छा वाक्य है।")
        assert score > 0.6

    def test_quality_score_bad_text(self, preprocessor):
        score = preprocessor.quality_score("!!! ... ??? ...")
        assert score < 0.5

    def test_batch_preprocess(self, preprocessor):
        texts = list(SAMPLE_TEXTS.values())
        results = preprocessor.batch_preprocess(texts, show_progress=False)
        assert len(results) == len(texts)

    def test_text_features(self, preprocessor):
        text = "Hello world, this is English text."
        features = preprocessor.compute_text_features(text)
        assert "n_chars" in features
        assert "dominant_script" in features
        assert "char_entropy" in features
        assert features["n_chars"] > 0


# ── Script Analyzer Tests ─────────────────────────────────────────────────────

class TestScriptAnalyzer:
    def test_devanagari_detection(self, script_analyzer):
        result = script_analyzer.analyze(SAMPLE_TEXTS["hin"])
        assert "Devanagari" in result.dominant_script or result.dominant_script_ratio > 0.5

    def test_arabic_detection(self, script_analyzer):
        result = script_analyzer.analyze(SAMPLE_TEXTS["urd"])
        assert result.primary_direction == "RTL" or result.rtl_ratio > 0.3

    def test_empty_text(self, script_analyzer):
        result = script_analyzer.analyze("")
        assert result.total_chars == 0

    def test_mixed_script(self, script_analyzer):
        mixed = "यह mixed text है with English"
        result = script_analyzer.analyze(mixed)
        assert result.is_mixed_script or len(result.script_distribution) > 1

    def test_pure_script(self, script_analyzer):
        result = script_analyzer.analyze(SAMPLE_TEXTS["tam"])
        assert result.dominant_script_ratio > 0.7

    def test_script_segments(self, script_analyzer):
        result = script_analyzer.analyze("नमस्ते hello नमस्ते")
        assert len(result.script_segments) >= 1

    def test_noise_detection_urls(self, script_analyzer):
        text = "Visit https://www.google.com for more"
        result = script_analyzer.analyze(text)
        assert result.url_count >= 1

    def test_noise_score_range(self, script_analyzer):
        for text in SAMPLE_TEXTS.values():
            result = script_analyzer.analyze(text)
            assert 0.0 <= result.noise_score <= 1.0

    def test_entropy_nonnegative(self, script_analyzer):
        for text in SAMPLE_TEXTS.values():
            result = script_analyzer.analyze(text)
            assert result.char_entropy >= 0.0

    def test_char_counts_consistent(self, script_analyzer):
        text = "Hello world!"
        result = script_analyzer.analyze(text)
        assert result.total_chars == len(text)

    def test_transliteration_detection(self, script_analyzer):
        roman_hindi = "yeh ek bahut achha din hai"
        result = script_analyzer.detect_transliteration(roman_hindi)
        assert "is_transliteration" in result
        assert "confidence" in result

    def test_compare_texts(self, script_analyzer):
        r = script_analyzer.compare_texts(SAMPLE_TEXTS["hin"], SAMPLE_TEXTS["mar"])
        assert "jaccard_similarity" in r
        assert r["dominant_scripts_match"]  # both Devanagari

    def test_lang_hints(self, script_analyzer):
        result = script_analyzer.analyze(SAMPLE_TEXTS["tam"])
        # Tamil script should hint toward "tam"
        hints = result.script_to_lang_hints
        assert any("tam" in v for v in hints.values())


# ── Heuristic Model Tests ─────────────────────────────────────────────────────

class TestUnicodeHeuristicModel:
    def test_tamil_detection(self, heuristic_model):
        lang, conf = heuristic_model.predict_single(SAMPLE_TEXTS["tam"])
        assert lang == "tam"
        assert conf > 0.5

    def test_telugu_detection(self, heuristic_model):
        lang, conf = heuristic_model.predict_single(SAMPLE_TEXTS["tel"])
        assert lang == "tel"

    def test_malayalam_detection(self, heuristic_model):
        lang, conf = heuristic_model.predict_single(SAMPLE_TEXTS["mal"])
        assert lang == "mal"

    def test_kannada_detection(self, heuristic_model):
        lang, conf = heuristic_model.predict_single(SAMPLE_TEXTS["kan"])
        assert lang == "kan"

    def test_bengali_detection(self, heuristic_model):
        lang, conf = heuristic_model.predict_single(SAMPLE_TEXTS["ben"])
        assert lang == "ben"

    def test_urdu_detection(self, heuristic_model):
        lang, conf = heuristic_model.predict_single(SAMPLE_TEXTS["urd"])
        assert lang == "urd"

    def test_english_detection(self, heuristic_model):
        lang, conf = heuristic_model.predict_single(SAMPLE_TEXTS["eng"])
        assert lang == "eng"
        assert conf > 0.5

    def test_batch_predict(self, heuristic_model):
        texts = list(SAMPLE_TEXTS.values())
        results = heuristic_model.predict(texts)
        assert len(results) == len(texts)
        for lang, conf in results:
            assert conf is not None


# ── Ensemble Model (no trained models) ────────────────────────────────────────

class TestEnsembleModel:
    def test_init(self):
        model = EnsembleLIDModel()
        assert model.heuristic_model is not None

    def test_predict_without_ml_models(self):
        model = EnsembleLIDModel()
        # Should still work via heuristics
        texts = [SAMPLE_TEXTS["tam"], SAMPLE_TEXTS["eng"]]
        results = model.predict_with_confidence(texts)
        assert len(results) == 2

    def test_result_structure(self):
        model = EnsembleLIDModel()
        results = model.predict_with_confidence([SAMPLE_TEXTS["tam"]])
        r = results[0]
        assert "predicted_lang" in r
        assert "confidence" in r
        assert "is_confident" in r
        assert 0.0 <= r["confidence"] <= 1.0


# ── get_dominant_script Tests ─────────────────────────────────────────────────

class TestDominantScript:
    def test_empty_text(self):
        script, dist = get_dominant_script("")
        assert script == "Unknown"

    def test_latin_text(self):
        script, dist = get_dominant_script("Hello world")
        assert "Latin" in script or "Basic" in script

    def test_devanagari_text(self):
        script, dist = get_dominant_script("नमस्ते")
        assert "Devanagari" in script

    def test_distribution_sums_to_one(self):
        _, dist = get_dominant_script("Hello नमस्ते")
        if dist:
            assert abs(sum(dist.values()) - 1.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
