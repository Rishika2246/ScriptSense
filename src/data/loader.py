"""
Advanced Dataset Loader for AI4Bharat/Pralekha
Supports streaming, caching, balanced sampling, and preprocessing.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

# HuggingFace datasets
try:
    from datasets import load_dataset, Dataset, DatasetDict, IterableDataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("datasets library not installed. Install with: pip install datasets")

from src.utils.languages import (
    ALL_LANGUAGE_CODES,
    LANGUAGE_REGISTRY,
    CODE_TO_LABEL,
)


# ─────────────────────────────────────────────
# Dataset Configuration
# ─────────────────────────────────────────────
PRALEKHA_REPO = "ai4bharat/Pralekha"
UNALIGNABLE_LANGS = ["ben","eng","guj","hin","kan","mal","mar","ori","pan","tam","tel","urd"]
ALIGNABLE_SPLITS  = ["train", "dev", "test"]


class PralekhaLoader:
    """
    Streaming-capable loader for the AI4Bharat/Pralekha multilingual corpus.

    Strategy:
      - 'unalignable' subset: each split IS a language (ben, hin, ...)
        → perfect for language-labelled data
      - 'alignable' subset: multilingual with 'lang' field
        → used for cross-lingual alignment tasks

    Usage:
        loader = PralekhaLoader(cache_dir="./cache/datasets")
        df = loader.load_language_labelled_dataframe(max_per_lang=10000)
        for batch in loader.stream_batches(batch_size=256):
            ...
    """

    def __init__(
        self,
        cache_dir: str = "./cache/datasets",
        streaming: bool = True,
        max_per_lang: Optional[int] = 50_000,
        min_text_len: int = 50,
        max_text_len: int = 2000,
        seed: int = 42,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.streaming = streaming
        self.max_per_lang = max_per_lang
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self._cache: Dict[str, pd.DataFrame] = {}

    # ── Core Loading ──────────────────────────────────────────────────────────

    def load_single_language(
        self, lang_code: str, max_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """Load all text for a single language from the unalignable subset."""
        assert lang_code in UNALIGNABLE_LANGS, f"Unknown language: {lang_code}"
        cache_key = f"{lang_code}_{max_samples}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        parquet_cache = self.cache_dir / f"{lang_code}.parquet"
        if parquet_cache.exists():
            logger.info(f"Loading {lang_code} from local cache ...")
            df = pd.read_parquet(parquet_cache)
            self._cache[cache_key] = df
            return df

        logger.info(f"Streaming {lang_code} from HuggingFace ...")
        records = []
        count = 0
        max_n = max_samples or self.max_per_lang or float("inf")

        try:
            ds = load_dataset(
                PRALEKHA_REPO,
                name="unalignable",
                split=lang_code,
                streaming=True,
                trust_remote_code=True,
            )
            for row in ds:
                text = row.get("text", "")
                if not self._text_filter(text):
                    continue
                records.append({
                    "text": text[:self.max_text_len],
                    "lang": lang_code,
                    "label": CODE_TO_LABEL[lang_code],
                    "doc_id": row.get("doc_id", ""),
                    "n_id": row.get("n_id", ""),
                    "text_len": len(text),
                })
                count += 1
                if count >= max_n:
                    break
        except Exception as e:
            logger.error(f"Failed to load {lang_code}: {e}")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df.to_parquet(parquet_cache, index=False)
        logger.success(f"Loaded {len(df):,} samples for {lang_code}")
        self._cache[cache_key] = df
        return df

    def load_language_labelled_dataframe(
        self,
        languages: Optional[List[str]] = None,
        max_per_lang: Optional[int] = None,
        balanced: bool = True,
    ) -> pd.DataFrame:
        """
        Load a fully labelled DataFrame with all (or selected) languages.
        If balanced=True, undersample majority classes to minority class size.
        """
        langs = languages or UNALIGNABLE_LANGS
        max_n = max_per_lang or self.max_per_lang
        dfs = []

        for lang in tqdm(langs, desc="Loading languages"):
            df = self.load_single_language(lang, max_samples=max_n)
            if not df.empty:
                dfs.append(df)

        if not dfs:
            raise RuntimeError("No data loaded!")

        combined = pd.concat(dfs, ignore_index=True)

        if balanced:
            min_count = combined["lang"].value_counts().min()
            logger.info(f"Balancing dataset to {min_count:,} samples/language")
            combined = (
                combined.groupby("lang", group_keys=False)
                .apply(lambda g: g.sample(n=min(len(g), min_count), random_state=self.seed))
                .reset_index(drop=True)
            )

        combined = combined.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        logger.success(
            f"Final dataset: {len(combined):,} samples, "
            f"{combined['lang'].nunique()} languages"
        )
        return combined

    def stream_batches(
        self,
        languages: Optional[List[str]] = None,
        batch_size: int = 256,
        shuffle: bool = True,
    ) -> Generator[List[Dict], None, None]:
        """
        Yield batches of records across all languages interleaved.
        Memory-efficient for large-scale processing.
        """
        langs = languages or UNALIGNABLE_LANGS
        iterators: Dict[str, Iterator] = {}

        for lang in langs:
            try:
                ds = load_dataset(
                    PRALEKHA_REPO,
                    name="unalignable",
                    split=lang,
                    streaming=True,
                    trust_remote_code=True,
                )
                iterators[lang] = iter(ds)
            except Exception as e:
                logger.warning(f"Could not open stream for {lang}: {e}")

        buffer: List[Dict] = []
        active_langs = list(iterators.keys())

        while active_langs:
            lang = random.choice(active_langs) if shuffle else active_langs[0]
            try:
                row = next(iterators[lang])
                text = row.get("text", "")
                if self._text_filter(text):
                    buffer.append({
                        "text": text[:self.max_text_len],
                        "lang": lang,
                        "label": CODE_TO_LABEL[lang],
                        "doc_id": row.get("doc_id", ""),
                    })
                if len(buffer) >= batch_size:
                    if shuffle:
                        random.shuffle(buffer)
                    yield buffer
                    buffer = []
            except StopIteration:
                active_langs.remove(lang)
                logger.debug(f"Exhausted stream for {lang}")

        if buffer:
            yield buffer

    def load_alignable_split(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load the 'alignable' subset (train/dev/test) which contains
        multilingual documents with a 'lang' field.
        """
        assert split in ALIGNABLE_SPLITS
        logger.info(f"Loading alignable/{split} ...")
        records = []
        count = 0
        max_n = max_samples or float("inf")

        try:
            ds = load_dataset(
                PRALEKHA_REPO,
                name="alignable",
                split=split,
                streaming=True,
                trust_remote_code=True,
            )
            for row in ds:
                text = row.get("text", "")
                lang = row.get("lang", "unk")
                if not self._text_filter(text):
                    continue
                records.append({
                    "text": text[:self.max_text_len],
                    "lang": lang,
                    "label": CODE_TO_LABEL.get(lang, -1),
                    "doc_id": row.get("doc_id", ""),
                })
                count += 1
                if count >= max_n:
                    break
        except Exception as e:
            logger.error(f"Failed to load alignable/{split}: {e}")

        return pd.DataFrame(records)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _text_filter(self, text: str) -> bool:
        """Basic quality filter for text samples."""
        if not text or not isinstance(text, str):
            return False
        text = text.strip()
        if len(text) < self.min_text_len:
            return False
        # Skip if overwhelmingly whitespace or punctuation
        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
        if alpha_ratio < 0.4:
            return False
        return True

    def get_dataset_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-language statistics."""
        stats = []
        for lang, grp in df.groupby("lang"):
            info = LANGUAGE_REGISTRY.get(lang)
            stats.append({
                "lang": lang,
                "name": info.name if info else lang,
                "script": info.script if info else "?",
                "n_samples": len(grp),
                "avg_len": grp["text_len"].mean() if "text_len" in grp else grp["text"].str.len().mean(),
                "min_len": grp["text"].str.len().min(),
                "max_len": grp["text"].str.len().max(),
                "unique_docs": grp["doc_id"].nunique() if "doc_id" in grp else 0,
            })
        return pd.DataFrame(stats).sort_values("n_samples", ascending=False)

    def create_train_val_test_split(
        self,
        df: pd.DataFrame,
        val_size: float = 0.05,
        test_size: float = 0.10,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Stratified split preserving language distribution."""
        from sklearn.model_selection import train_test_split

        train_val, test = train_test_split(
            df, test_size=test_size, stratify=df["label"], random_state=self.seed
        )
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, test_size=val_ratio, stratify=train_val["label"], random_state=self.seed
        )
        logger.info(
            f"Split → Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}"
        )
        return train, val, test

    def export_splits(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        output_dir: str = "./data/splits",
    ):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        train.to_parquet(out / "train.parquet", index=False)
        val.to_parquet(out / "val.parquet", index=False)
        test.to_parquet(out / "test.parquet", index=False)
        logger.success(f"Splits saved to {out}")

    @staticmethod
    def load_splits(data_dir: str = "./data/splits") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        d = Path(data_dir)
        return (
            pd.read_parquet(d / "train.parquet"),
            pd.read_parquet(d / "val.parquet"),
            pd.read_parquet(d / "test.parquet"),
        )
