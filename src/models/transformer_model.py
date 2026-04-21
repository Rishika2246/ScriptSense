"""
Transformer-based Language Identification Model
Uses Google MuRIL (Multilingual Universal Representations for Indian Languages)
or XLM-RoBERTa for high-accuracy sequence classification.

Requires GPU for practical training. Set enabled: true in config.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
    )
    from transformers import DataCollatorWithPadding
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers/torch not installed. Transformer model unavailable.")

from src.utils.languages import (
    LANGUAGE_REGISTRY,
    CODE_TO_LABEL,
    LABEL_TO_CODE,
    NUM_CLASSES,
    ALL_LANGUAGE_CODES,
)


# ─────────────────────────────────────────────
# Dataset Wrapper
# ─────────────────────────────────────────────

class IndicLIDDataset:
    """Wraps a pandas DataFrame for HuggingFace Trainer."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────
# Transformer LID Model
# ─────────────────────────────────────────────

class TransformerLIDModel:
    """
    Fine-tuned transformer for language identification.
    Defaults to MuRIL (designed for Indic languages).

    MuRIL advantages over XLM-R for this task:
    - Pretrained on 17 Indian languages + transliterated text
    - Better subword coverage for Indic scripts
    - Transliteration-aware pretraining
    """

    SUPPORTED_MODELS = {
        "muril": "google/muril-base-cased",
        "xlmr": "xlm-roberta-base",
        "xlmr-large": "xlm-roberta-large",
        "indicbert": "ai4bharat/indic-bert",
    }

    def __init__(
        self,
        base_model: str = "muril",
        max_length: int = 128,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        output_dir: str = "./artifacts/models/transformer",
        fp16: bool = True,
    ):
        assert TRANSFORMERS_AVAILABLE, "transformers + torch required"
        self.model_name = self.SUPPORTED_MODELS.get(base_model, base_model)
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fp16 = fp16 and torch.cuda.is_available()

        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.is_fitted = False

        # Label mapping
        self.id2label = {i: l for i, l in LABEL_TO_CODE.items()}
        self.label2id = {l: i for i, l in self.id2label.items()}

    def _load_base_model(self):
        logger.info(f"Loading tokenizer and model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=NUM_CLASSES,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        self.model.to(device)

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> "TransformerLIDModel":
        self._load_base_model()

        train_dataset = IndicLIDDataset(train_df, self.tokenizer, self.max_length)
        val_dataset = IndicLIDDataset(val_df, self.tokenizer, self.max_length) if val_df is not None else None

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 2,
            learning_rate=self.learning_rate,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=val_dataset is not None,
            metric_for_best_model="eval_loss",
            fp16=self.fp16,
            logging_steps=100,
            logging_dir=str(self.output_dir / "logs"),
            dataloader_num_workers=4,
            seed=42,
            report_to="none",  # Disable wandb/mlflow unless configured
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if val_dataset else [],
            compute_metrics=self._compute_metrics,
        )

        logger.info("Starting transformer training ...")
        self.trainer.train()
        self.is_fitted = True
        logger.success("Training complete!")
        return self

    def _compute_metrics(self, eval_pred):
        from sklearn.metrics import f1_score, accuracy_score
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="macro", zero_division=0)
        return {"accuracy": acc, "macro_f1": f1}

    def predict(self, texts: List[str]) -> List[str]:
        assert self.is_fitted
        probs, _ = self.predict_proba(texts)
        label_ids = np.argmax(probs, axis=1)
        return [self.id2label[i] for i in label_ids]

    def predict_proba(self, texts: List[str]) -> Tuple[np.ndarray, List[str]]:
        assert self.is_fitted
        import torch
        from torch.nn.functional import softmax

        device = next(self.model.parameters()).device
        all_probs = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            encodings = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = self.model(**encodings)
                probs = softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

        prob_matrix = np.vstack(all_probs)
        class_names = [self.id2label[i] for i in range(NUM_CLASSES)]
        return prob_matrix, class_names

    def predict_with_confidence(self, texts: List[str]) -> List[Dict]:
        probs, class_names = self.predict_proba(texts)
        results = []
        for i, prob_row in enumerate(probs):
            top_k_idx = np.argsort(prob_row)[::-1][:5]
            top_k = [
                {
                    "lang": class_names[j],
                    "name": LANGUAGE_REGISTRY.get(class_names[j], None) and LANGUAGE_REGISTRY[class_names[j]].name,
                    "probability": float(prob_row[j]),
                }
                for j in top_k_idx
            ]
            best = top_k[0]
            info = LANGUAGE_REGISTRY.get(best["lang"])
            results.append({
                "text_preview": texts[i][:80],
                "predicted_lang": best["lang"],
                "predicted_lang_name": info.name if info else best["lang"],
                "script": info.script if info else "?",
                "direction": info.direction if info else "LTR",
                "confidence": best["probability"],
                "is_confident": best["probability"] >= 0.75,
                "top_5": top_k,
            })
        return results

    def evaluate(self, test_df: pd.DataFrame) -> Dict:
        from sklearn.metrics import classification_report, f1_score, accuracy_score
        texts = test_df["text"].tolist()
        true_labels = test_df["lang"].tolist()
        pred_labels = self.predict(texts)
        probs, _ = self.predict_proba(texts)

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
            "confidences": probs.max(axis=1).tolist(),
        }

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.success(f"Transformer model saved: {path}")

    @classmethod
    def load(cls, path: str) -> "TransformerLIDModel":
        assert TRANSFORMERS_AVAILABLE
        obj = cls.__new__(cls)
        obj.output_dir = Path(path)
        obj.tokenizer = AutoTokenizer.from_pretrained(path)
        obj.model = AutoModelForSequenceClassification.from_pretrained(path)
        obj.is_fitted = True
        obj.id2label = {i: l for i, l in LABEL_TO_CODE.items()}
        obj.batch_size = 32
        obj.max_length = 128
        obj.fp16 = False
        device = "cuda" if torch.cuda.is_available() else "cpu"
        obj.model.to(device)
        logger.success(f"Transformer model loaded: {path}")
        return obj
