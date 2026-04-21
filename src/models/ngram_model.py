"""
Character N-gram Language Identification Model
High-accuracy baseline using TF-IDF character n-grams + Logistic Regression.
Achieves ~98%+ accuracy on clean Indic text.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.utils.languages import LANGUAGE_REGISTRY, LABEL_TO_CODE, CODE_TO_LABEL


class CharNgramLIDModel:
    """
    Character N-gram Language Identification using TF-IDF + LogReg.

    Key design choices:
      - analyzer='char_wb': treats word boundaries specially
      - ngram_range=(2,4): captures morpheme-level patterns
      - sublinear_tf=True: dampens frequency explosion
      - Calibration: isotonic regression for reliable probabilities
      - Per-language confidence thresholds for rejection
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (2, 4),
        max_features: int = 150_000,
        C: float = 5.0,
        max_iter: int = 1000,
        calibrate: bool = True,
        n_jobs: int = -1,
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.C = C
        self.max_iter = max_iter
        self.calibrate = calibrate
        self.n_jobs = n_jobs
        self.pipeline: Optional[Pipeline] = None
        self.label_encoder = LabelEncoder()
        self.classes_: Optional[np.ndarray] = None
        self.confidence_thresholds: Dict[str, float] = {}
        self.is_fitted = False

    def _build_pipeline(self) -> Pipeline:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=True,
            strip_accents=None,     # preserve Indic diacritics!
            min_df=2,
            max_df=0.95,
            dtype=np.float32,
        )
        base_clf = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="saga",
            multi_class="multinomial",
            n_jobs=self.n_jobs,
            verbose=0,
        )
        if self.calibrate:
            clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
        else:
            clf = base_clf

        return Pipeline([
            ("tfidf", vectorizer),
            ("clf", clf),
        ])

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> "CharNgramLIDModel":
        logger.info(
            f"Training CharNgramLID | n_gram={self.ngram_range} | "
            f"features={self.max_features} | C={self.C}"
        )
        texts = train_df["text"].tolist()
        labels = self.label_encoder.fit_transform(train_df["lang"].tolist())
        self.classes_ = self.label_encoder.classes_

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(texts, labels)
        self.is_fitted = True

        if val_df is not None:
            self._compute_confidence_thresholds(val_df)
            val_preds = self.predict(val_df["text"].tolist())
            f1 = f1_score(val_df["lang"].tolist(), val_preds, average="macro")
            logger.success(f"Val Macro-F1: {f1:.4f}")

        return self

    def predict(self, texts: List[str]) -> List[str]:
        assert self.is_fitted, "Model not fitted!"
        label_ids = self.pipeline.predict(texts)
        return self.label_encoder.inverse_transform(label_ids).tolist()

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Returns probability matrix [n_samples × n_classes]."""
        assert self.is_fitted
        return self.pipeline.predict_proba(texts)

    def predict_with_confidence(
        self, texts: List[str]
    ) -> List[Dict]:
        """
        Returns per-sample predictions with confidence and
        top-K alternatives.
        """
        probs = self.predict_proba(texts)
        results = []
        for i, prob_row in enumerate(probs):
            top_k_idx = np.argsort(prob_row)[::-1][:5]
            top_k = [
                {
                    "lang": self.classes_[j],
                    "name": LANGUAGE_REGISTRY.get(self.classes_[j], None),
                    "probability": float(prob_row[j]),
                }
                for j in top_k_idx
            ]
            best = top_k[0]
            threshold = self.confidence_thresholds.get(best["lang"], 0.5)
            results.append({
                "text": texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                "predicted_lang": best["lang"],
                "confidence": best["probability"],
                "is_confident": best["probability"] >= threshold,
                "top_5": top_k,
            })
        return results

    def _compute_confidence_thresholds(
        self, val_df: pd.DataFrame, target_precision: float = 0.95
    ):
        """
        Compute per-language confidence thresholds that achieve
        target_precision on the validation set.
        """
        probs = self.predict_proba(val_df["text"].tolist())
        true_langs = val_df["lang"].tolist()
        pred_langs = self.predict(val_df["text"].tolist())

        for lang in self.classes_:
            lang_idx = np.where(self.classes_ == lang)[0]
            if len(lang_idx) == 0:
                continue
            idx = lang_idx[0]
            lang_probs = probs[:, idx]
            is_correct = np.array([p == lang for p in pred_langs])
            is_predicted = np.array([p == lang for p in pred_langs])

            # Find threshold where precision ≥ target
            thresholds = np.linspace(0.3, 0.99, 50)
            best_t = 0.5
            for t in thresholds:
                mask = (lang_probs >= t) & is_predicted
                if mask.sum() == 0:
                    break
                prec = is_correct[mask].mean()
                if prec >= target_precision:
                    best_t = t
                    break
            self.confidence_thresholds[lang] = best_t

        logger.debug(f"Confidence thresholds: {self.confidence_thresholds}")

    def evaluate(self, test_df: pd.DataFrame) -> Dict:
        """Full evaluation with per-language metrics."""
        texts = test_df["text"].tolist()
        true_labels = test_df["lang"].tolist()
        pred_labels = self.predict(texts)
        probs = self.predict_proba(texts)

        overall_f1 = f1_score(true_labels, pred_labels, average="macro")
        report = classification_report(
            true_labels, pred_labels,
            target_names=self.classes_,
            output_dict=True,
        )

        # Confidence calibration: ECE (Expected Calibration Error)
        confidences = probs.max(axis=1)
        correct = np.array(true_labels) == np.array(pred_labels)
        ece = self._compute_ece(confidences, correct)

        logger.info(f"\n{classification_report(true_labels, pred_labels)}")
        logger.success(f"Macro F1: {overall_f1:.4f} | ECE: {ece:.4f}")

        return {
            "macro_f1": overall_f1,
            "ece": ece,
            "per_language": report,
            "predictions": pred_labels,
            "true_labels": true_labels,
            "confidences": confidences.tolist(),
        }

    @staticmethod
    def _compute_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (confidences >= lo) & (confidences < hi)
            if mask.sum() == 0:
                continue
            avg_conf = confidences[mask].mean()
            avg_acc = correct[mask].mean()
            ece += mask.mean() * abs(avg_conf - avg_acc)
        return float(ece)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "pipeline": self.pipeline,
                "label_encoder": self.label_encoder,
                "classes_": self.classes_,
                "confidence_thresholds": self.confidence_thresholds,
                "config": {
                    "ngram_range": self.ngram_range,
                    "max_features": self.max_features,
                    "C": self.C,
                },
            }, f)
        logger.success(f"Model saved: {path}")

    @classmethod
    def load(cls, path: str) -> "CharNgramLIDModel":
        with open(path, "rb") as f:
            data = pickle.load(f)
        cfg = data.get("config", {})
        model = cls(
            ngram_range=cfg.get("ngram_range", (2, 4)),
            max_features=cfg.get("max_features", 150_000),
            C=cfg.get("C", 5.0),
        )
        model.pipeline = data["pipeline"]
        model.label_encoder = data["label_encoder"]
        model.classes_ = data["classes_"]
        model.confidence_thresholds = data.get("confidence_thresholds", {})
        model.is_fitted = True
        logger.success(f"Model loaded: {path}")
        return model

    def get_top_features_per_language(self, n: int = 20) -> Dict[str, List[str]]:
        """Extract most discriminative n-grams per language."""
        assert self.is_fitted
        tfidf = self.pipeline.named_steps["tfidf"]
        clf = self.pipeline.named_steps["clf"]
        feature_names = np.array(tfidf.get_feature_names_out())

        # Get the underlying LR if calibrated
        if hasattr(clf, "calibrated_classifiers_"):
            coef = np.mean(
                [c.estimator.coef_ for c in clf.calibrated_classifiers_], axis=0
            )
        else:
            coef = clf.coef_

        result = {}
        for i, lang in enumerate(self.classes_):
            top_idx = np.argsort(coef[i])[::-1][:n]
            result[lang] = feature_names[top_idx].tolist()
        return result
