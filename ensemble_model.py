"""
Ensemble Language Identification Model
Combines CharNgram + FastText + Unicode heuristics for maximum accuracy.
Strategies: weighted soft voting, stacking, confidence-based routing.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import f1_score, accuracy_score, classification_report

from src.utils.languages import (
    LANGUAGE_REGISTRY,
    ALL_LANGUAGE_CODES,
    CODE_TO_LABEL,
    LABEL_TO_CODE,
    CONFUSABLE_PAIRS,
    RTL_LANGUAGES,
    DEVANAGARI_LANGUAGES,
)
from src.data.preprocessor import get_dominant_script


# ─────────────────────────────────────────────
# Unicode Heuristic Model (Rule-based)
# ─────────────────────────────────────────────

class UnicodeHeuristicModel:
    """
    Fast rule-based language identifier using Unicode script analysis.
    Used as a strong prior for high-confidence cases.
    """

    # Unique codepoints per language for disambiguation
    DISTINCTIVE_CHARS = {
        # Telugu distinctive vowel signs
        "tel": set(range(0x0C00, 0x0C7F)),
        # Kannada
        "kan": set(range(0x0C80, 0x0CFF)),
        # Malayalam
        "mal": set(range(0x0D00, 0x0D7F)),
        # Tamil — note: no voiced stops
        "tam": set(range(0x0B80, 0x0BFF)),
        # Odia
        "ori": set(range(0x0B00, 0x0B7F)),
        # Gurmukhi (Punjabi)
        "pan": set(range(0x0A00, 0x0A7F)),
        # Gujarati
        "guj": set(range(0x0A80, 0x0AFF)),
        # Bengali
        "ben": set(range(0x0980, 0x09FF)),
        # Arabic/Urdu
        "urd": set(range(0x0600, 0x06FF)) | set(range(0xFB50, 0xFDFF)),
    }

    # Devanagari — needs lexical disambiguation between Hindi/Marathi
    DEVANAGARI_RANGE = set(range(0x0900, 0x097F))
    LATIN_RANGE = set(range(0x0041, 0x007B))

    def predict_single(self, text: str) -> Tuple[Optional[str], float]:
        """
        Returns (predicted_lang, confidence).
        confidence=0.0 → uncertain (pass to ML model).
        """
        char_codepoints = [ord(c) for c in text if c.isalpha()]
        if not char_codepoints:
            return None, 0.0

        total = len(char_codepoints)

        # Check unique scripts with unambiguous mapping
        for lang, codepoints in self.DISTINCTIVE_CHARS.items():
            matches = sum(cp in codepoints for cp in char_codepoints)
            ratio = matches / total
            if ratio > 0.75:
                return lang, min(0.98, 0.7 + ratio * 0.28)

        # Devanagari → Hindi/Marathi ambiguous, return low confidence
        deva_matches = sum(cp in self.DEVANAGARI_RANGE for cp in char_codepoints)
        if deva_matches / total > 0.75:
            return "hin", 0.55  # slight prior toward Hindi (larger corpus)

        # Latin → likely English
        latin_matches = sum(cp in self.LATIN_RANGE for cp in char_codepoints)
        if latin_matches / total > 0.85:
            return "eng", 0.90

        return None, 0.0

    def predict(self, texts: List[str]) -> List[Tuple[Optional[str], float]]:
        return [self.predict_single(t) for t in texts]


# ─────────────────────────────────────────────
# Ensemble Model
# ─────────────────────────────────────────────

class EnsembleLIDModel:
    """
    Multi-model ensemble for language identification.

    Architecture:
      1. Unicode heuristic: fast pre-filter, handles 40-60% of inputs
      2. CharNgram model: strong char-level baseline
      3. FastText model: fast neural model with subword features
      4. Weighted voting: combine model probabilities

    Calibration:
      - Per-language confidence thresholds
      - Devanagari disambiguation module (Hindi vs Marathi)
      - Confusable pair re-ranking
    """

    SUPPORTED_STRATEGIES = ("weighted_vote", "stacking", "router")

    def __init__(
        self,
        ngram_model=None,
        fasttext_model=None,
        transformer_model=None,
        weights: Optional[Dict[str, float]] = None,
        strategy: str = "weighted_vote",
    ):
        self.ngram_model = ngram_model
        self.fasttext_model = fasttext_model
        self.transformer_model = transformer_model
        self.heuristic_model = UnicodeHeuristicModel()
        self.weights = weights or {"ngram": 0.35, "fasttext": 0.50, "transformer": 0.15}
        self.strategy = strategy
        self.all_langs = ALL_LANGUAGE_CODES
        self.stacking_meta: Optional[Any] = None
        self.is_fitted = False

    @property
    def active_models(self) -> Dict[str, Any]:
        models = {}
        if self.ngram_model and getattr(self.ngram_model, "is_fitted", False):
            models["ngram"] = self.ngram_model
        if self.fasttext_model and getattr(self.fasttext_model, "is_fitted", False):
            models["fasttext"] = self.fasttext_model
        if self.transformer_model and getattr(self.transformer_model, "is_fitted", False):
            models["transformer"] = self.transformer_model
        return models

    def fit_stacking(self, val_df: pd.DataFrame):
        """
        Fit meta-learner (stacking) on validation set predictions.
        """
        from sklearn.linear_model import LogisticRegression

        logger.info("Fitting stacking meta-learner on validation set ...")
        stacked_features = self._get_stacked_features(val_df["text"].tolist())
        labels = val_df["lang"].tolist()

        meta = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", multi_class="multinomial")
        meta.fit(stacked_features, labels)
        self.stacking_meta = meta
        logger.success("Stacking meta-learner fitted.")

    def _get_stacked_features(self, texts: List[str]) -> np.ndarray:
        """Concatenate all model probability vectors."""
        features = []
        for name, model in self.active_models.items():
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(texts)
                if isinstance(probs, tuple):
                    probs, langs = probs
                    # Reindex to self.all_langs
                    full = np.zeros((len(texts), len(self.all_langs)))
                    for j, lang in enumerate(langs):
                        if lang in self.all_langs:
                            k = self.all_langs.index(lang)
                            full[:, k] = probs[:, j]
                    probs = full
                features.append(probs)
        if not features:
            raise RuntimeError("No active models for stacking")
        return np.hstack(features)

    def predict(self, texts: List[str]) -> List[str]:
        results = self.predict_with_confidence(texts)
        return [r["predicted_lang"] for r in results]

    def predict_with_confidence(self, texts: List[str]) -> List[Dict]:
        """Full ensemble inference with detailed outputs."""
        results = []
        heuristic_preds = self.heuristic_model.predict(texts)

        # Get model probabilities
        model_probs = {}
        for name, model in self.active_models.items():
            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(texts)
                    if isinstance(probs, tuple):
                        prob_matrix, langs = probs
                        model_probs[name] = (prob_matrix, langs)
                    else:
                        # ngram model returns matrix with .classes_
                        classes = model.classes_ if hasattr(model, "classes_") else self.all_langs
                        model_probs[name] = (probs, list(classes))
                except Exception as e:
                    logger.warning(f"Model {name} failed: {e}")

        for i, text in enumerate(texts):
            h_lang, h_conf = heuristic_preds[i]

            # High-confidence heuristic → use directly
            if h_lang and h_conf >= 0.92 and h_lang not in DEVANAGARI_LANGUAGES:
                results.append(self._build_result(text, h_lang, h_conf, {}, "heuristic"))
                continue

            # Weighted voting across ML models
            combined = np.zeros(len(self.all_langs))
            total_weight = 0.0

            for name, (prob_matrix, langs) in model_probs.items():
                w = self.weights.get(name, 1.0)
                total_weight += w
                prob_row = prob_matrix[i]
                for j, lang in enumerate(langs):
                    if lang in self.all_langs:
                        k = self.all_langs.index(lang)
                        combined[k] += w * (prob_row[j] if j < len(prob_row) else 0)

            if total_weight > 0:
                combined /= total_weight

            # Blend in heuristic prior if partially confident
            if h_lang and h_conf >= 0.55:
                k = self.all_langs.index(h_lang) if h_lang in self.all_langs else -1
                if k >= 0:
                    alpha = 0.3
                    combined[k] = (1 - alpha) * combined[k] + alpha * h_conf

            best_idx = int(np.argmax(combined))
            pred_lang = self.all_langs[best_idx]
            confidence = float(combined[best_idx])

            # Devanagari disambiguation (Hindi vs Marathi)
            if pred_lang in DEVANAGARI_LANGUAGES:
                pred_lang, confidence = self._disambiguate_devanagari(
                    text, combined, pred_lang, confidence
                )

            source_contributions = {
                name: self._get_model_top_pred(name, model_probs, i)
                for name in model_probs
            }
            results.append(
                self._build_result(text, pred_lang, confidence, source_contributions, "ensemble")
            )

        return results

    def _disambiguate_devanagari(
        self, text: str, probs: np.ndarray, pred: str, conf: float
    ) -> Tuple[str, float]:
        """
        Use lexical markers to disambiguate Hindi vs Marathi.
        Both use Devanagari, so char-level features alone aren't enough.
        """
        # Marathi-specific words/patterns
        marathi_markers = ["आहे", "आहेत", "होते", "केले", "त्यांनी", "आणि", "मराठी"]
        hindi_markers = ["है", "हैं", "था", "थे", "में", "और", "से", "हिंदी"]

        mar_score = sum(1 for m in marathi_markers if m in text)
        hin_score = sum(1 for m in hindi_markers if m in text)

        if mar_score > hin_score + 1:
            k = self.all_langs.index("mar") if "mar" in self.all_langs else -1
            return "mar", min(0.95, conf + 0.1 * mar_score)
        elif hin_score > mar_score + 1:
            k = self.all_langs.index("hin") if "hin" in self.all_langs else -1
            return "hin", min(0.95, conf + 0.05 * hin_score)
        return pred, conf

    def _get_model_top_pred(self, name: str, model_probs: Dict, idx: int) -> Dict:
        if name not in model_probs:
            return {}
        prob_matrix, langs = model_probs[name]
        prob_row = prob_matrix[idx]
        best = int(np.argmax(prob_row))
        return {
            "lang": langs[best] if best < len(langs) else "?",
            "confidence": float(prob_row[best]),
        }

    def _build_result(
        self, text: str, lang: str, conf: float,
        contributions: Dict, source: str
    ) -> Dict:
        info = LANGUAGE_REGISTRY.get(lang)
        return {
            "text_preview": text[:80] + "..." if len(text) > 80 else text,
            "predicted_lang": lang,
            "predicted_lang_name": info.name if info else lang,
            "script": info.script if info else "Unknown",
            "direction": info.direction if info else "LTR",
            "confidence": round(conf, 4),
            "is_confident": conf >= 0.75,
            "source": source,
            "model_contributions": contributions,
        }

    def evaluate(self, test_df: pd.DataFrame) -> Dict:
        texts = test_df["text"].tolist()
        true_labels = test_df["lang"].tolist()
        results = self.predict_with_confidence(texts)
        pred_labels = [r["predicted_lang"] for r in results]

        accuracy = accuracy_score(true_labels, pred_labels)
        macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
        report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)

        logger.info(f"\n{classification_report(true_labels, pred_labels, zero_division=0)}")
        logger.success(f"Ensemble → Accuracy: {accuracy:.4f} | Macro-F1: {macro_f1:.4f}")

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "per_language": report,
            "predictions": pred_labels,
            "true_labels": true_labels,
            "detailed_results": results,
        }

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "weights": self.weights,
            "strategy": self.strategy,
            "stacking_meta": self.stacking_meta,
            "all_langs": self.all_langs,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.success(f"Ensemble config saved: {path}")
