"""
FastText-based Language Identification
Uses Facebook's fastText for ultra-fast, high-accuracy language detection.
Two modes:
  1. Pretrained lid.176.ftz (off-the-shelf, 176 languages)
  2. Custom-trained on Pralekha (domain-specific, higher accuracy on Indic)
"""
from __future__ import annotations

import os
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False
    logger.warning("fasttext not installed. `pip install fasttext-wheel`")

from src.utils.languages import LANGUAGE_REGISTRY, CODE_TO_LABEL


PRETRAINED_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
LANG_PREFIX = "__label__"

# Pralekha lang codes → fastText ISO codes (for pretrained model)
PRALEKHA_TO_FT = {
    "ben": "bn", "eng": "en", "guj": "gu", "hin": "hi",
    "kan": "kn", "mal": "ml", "mar": "mr", "ori": "or",
    "pan": "pa", "tam": "ta", "tel": "te", "urd": "ur",
}
FT_TO_PRALEKHA = {v: k for k, v in PRALEKHA_TO_FT.items()}


class FastTextLIDModel:
    """
    FastText-based Language Identification with two modes:
      - mode='pretrained': Use Facebook's lid.176.ftz (quick, general)
      - mode='custom'    : Train from scratch on Pralekha (best Indic accuracy)
    """

    def __init__(
        self,
        mode: str = "pretrained",   # "pretrained" | "custom"
        model_dir: str = "./artifacts/fasttext",
        dim: int = 100,
        epoch: int = 25,
        lr: float = 0.1,
        word_ngrams: int = 3,
        min_count: int = 5,
        thread: int = 8,
    ):
        assert mode in ("pretrained", "custom")
        self.mode = mode
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.epoch = epoch
        self.lr = lr
        self.word_ngrams = word_ngrams
        self.min_count = min_count
        self.thread = thread
        self.model = None
        self.is_fitted = False
        self._target_langs = list(PRALEKHA_TO_FT.keys())

    # ── Pretrained Mode ───────────────────────────────────────────────────────

    def load_pretrained(self) -> "FastTextLIDModel":
        """Download and load Facebook's lid.176.ftz pretrained model."""
        assert FASTTEXT_AVAILABLE, "fasttext not installed"
        model_path = self.model_dir / "lid.176.ftz"

        if not model_path.exists():
            logger.info(f"Downloading pretrained FastText LID model ...")
            urllib.request.urlretrieve(
                PRETRAINED_MODEL_URL,
                str(model_path),
                reporthook=self._download_progress,
            )
            logger.success("Download complete.")

        self.model = fasttext.load_model(str(model_path))
        self.is_fitted = True
        self.mode = "pretrained"
        logger.success("Pretrained FastText model loaded.")
        return self

    def _download_progress(self, block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(100, downloaded * 100 / total_size)
        if block_num % 50 == 0:
            logger.debug(f"Download: {pct:.1f}%")

    # ── Custom Training Mode ──────────────────────────────────────────────────

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> "FastTextLIDModel":
        """Train custom FastText model on Pralekha data."""
        assert FASTTEXT_AVAILABLE, "fasttext not installed"

        logger.info(f"Preparing FastText training data ({len(train_df):,} samples) ...")
        train_file = self.model_dir / "train.txt"
        self._write_fasttext_format(train_df, train_file)

        logger.info(f"Training FastText | dim={self.dim} | epoch={self.epoch} | lr={self.lr}")
        self.model = fasttext.train_supervised(
            input=str(train_file),
            dim=self.dim,
            epoch=self.epoch,
            lr=self.lr,
            wordNgrams=self.word_ngrams,
            minCount=self.min_count,
            thread=self.thread,
            verbose=2,
            loss="softmax",
        )

        model_path = self.model_dir / "custom_lid.bin"
        self.model.save_model(str(model_path))
        self.is_fitted = True
        self.mode = "custom"

        if val_df is not None:
            results = self.evaluate(val_df)
            logger.success(f"Val Accuracy: {results['accuracy']:.4f} | F1: {results['macro_f1']:.4f}")

        return self

    def _write_fasttext_format(self, df: pd.DataFrame, path: Path):
        """Write data in fastText supervised format: __label__LANG text"""
        with open(path, "w", encoding="utf-8") as f:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Writing FT data"):
                text = self._clean_for_fasttext(row["text"])
                lang = row["lang"]
                label = f"{LANG_PREFIX}{lang}"
                f.write(f"{label} {text}\n")

    @staticmethod
    def _clean_for_fasttext(text: str) -> str:
        """Minimal cleaning for fastText (preserve Unicode)."""
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[\r\n\t]", " ", text)
        return text

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, texts: List[str]) -> List[str]:
        """Predict language codes."""
        assert self.is_fitted
        results = []
        for text in texts:
            pred = self._predict_single(text)
            results.append(pred)
        return results

    def _predict_single(self, text: str) -> str:
        text = self._clean_for_fasttext(text)
        labels, probs = self.model.predict(text, k=1)
        raw_label = labels[0].replace(LANG_PREFIX, "")

        if self.mode == "pretrained":
            return FT_TO_PRALEKHA.get(raw_label, raw_label)
        return raw_label

    def predict_proba(self, texts: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Returns (prob_matrix [n × n_classes], class_names).
        """
        assert self.is_fitted
        all_labels = self._target_langs
        n = len(texts)
        n_classes = len(all_labels)
        prob_matrix = np.zeros((n, n_classes))

        for i, text in enumerate(texts):
            text_clean = self._clean_for_fasttext(text)
            labels, probs = self.model.predict(text_clean, k=-1)
            for label, prob in zip(labels, probs):
                raw = label.replace(LANG_PREFIX, "")
                mapped = FT_TO_PRALEKHA.get(raw, raw) if self.mode == "pretrained" else raw
                if mapped in all_labels:
                    j = all_labels.index(mapped)
                    prob_matrix[i, j] = prob

        # Normalize
        row_sums = prob_matrix.sum(axis=1, keepdims=True)
        prob_matrix = np.where(row_sums > 0, prob_matrix / row_sums, prob_matrix)

        return prob_matrix, all_labels

    def predict_with_confidence(self, texts: List[str]) -> List[Dict]:
        """Rich predictions with top-K confidences."""
        results = []
        for text in texts:
            text_clean = self._clean_for_fasttext(text)
            labels, probs = self.model.predict(text_clean, k=5)
            top_k = []
            for label, prob in zip(labels, probs):
                raw = label.replace(LANG_PREFIX, "")
                mapped = FT_TO_PRALEKHA.get(raw, raw) if self.mode == "pretrained" else raw
                info = LANGUAGE_REGISTRY.get(mapped)
                top_k.append({
                    "lang": mapped,
                    "name": info.name if info else mapped,
                    "probability": float(prob),
                })

            results.append({
                "text": text[:100],
                "predicted_lang": top_k[0]["lang"],
                "confidence": top_k[0]["probability"],
                "is_confident": top_k[0]["probability"] >= 0.7,
                "top_5": top_k,
            })
        return results

    def evaluate(self, test_df: pd.DataFrame) -> Dict:
        """Full evaluation."""
        from sklearn.metrics import classification_report, f1_score, accuracy_score

        texts = test_df["text"].tolist()
        true_labels = test_df["lang"].tolist()
        pred_labels = self.predict(texts)

        accuracy = accuracy_score(true_labels, pred_labels)
        macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
        report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)

        logger.info(f"\n{classification_report(true_labels, pred_labels, zero_division=0)}")
        logger.success(f"Accuracy: {accuracy:.4f} | Macro-F1: {macro_f1:.4f}")

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "per_language": report,
            "predictions": pred_labels,
            "true_labels": true_labels,
        }

    def load(self, path: str) -> "FastTextLIDModel":
        assert FASTTEXT_AVAILABLE
        self.model = fasttext.load_model(path)
        self.is_fitted = True
        return self
