"""
Corpus-level Statistical Linguistic Analysis
Computes per-language statistics, type-token ratios, vocabulary richness,
character frequency profiles, n-gram distributions, and cross-lingual metrics.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

from src.utils.languages import LANGUAGE_REGISTRY, ALL_LANGUAGE_CODES


# ─────────────────────────────────────────────
# Per-Language Profile
# ─────────────────────────────────────────────

class LanguageProfile:
    """
    Statistical profile for a single language corpus.
    """

    def __init__(self, lang_code: str):
        self.lang_code = lang_code
        self.info = LANGUAGE_REGISTRY.get(lang_code)
        self.char_freq: Counter = Counter()
        self.word_freq: Counter = Counter()
        self.bigram_freq: Counter = Counter()
        self.trigram_freq: Counter = Counter()
        self.text_lengths: List[int] = []
        self.word_lengths: List[int] = []
        self.n_documents = 0

    def update(self, text: str):
        """Update profile with a single document."""
        self.n_documents += 1
        self.text_lengths.append(len(text))

        # Character-level
        for char in text:
            if char.isalpha():
                self.char_freq[char] += 1

        # Word-level
        words = text.split()
        for word in words:
            clean = word.strip(".,!?;:\"'()[]{}।॥")
            if len(clean) >= 2:
                self.word_freq[clean] += 1
                self.word_lengths.append(len(clean))

        # Character n-grams (within words)
        for word in words:
            for i in range(len(word) - 1):
                self.bigram_freq[word[i:i+2]] += 1
            for i in range(len(word) - 2):
                self.trigram_freq[word[i:i+3]] += 1

    def compute_stats(self) -> Dict:
        total_chars = sum(self.char_freq.values())
        total_words = sum(self.word_freq.values())
        vocab_size = len(self.word_freq)

        # Type-Token Ratio (TTR)
        ttr = vocab_size / max(total_words, 1)

        # Hapax legomena ratio (words appearing exactly once)
        hapax = sum(1 for c in self.word_freq.values() if c == 1)
        hapax_ratio = hapax / max(vocab_size, 1)

        # Entropy
        if total_chars > 0:
            probs = np.array(list(self.char_freq.values())) / total_chars
            char_entropy = -np.sum(probs * np.log2(probs))
        else:
            char_entropy = 0.0

        return {
            "lang": self.lang_code,
            "name": self.info.name if self.info else self.lang_code,
            "script": self.info.script if self.info else "?",
            "n_documents": self.n_documents,
            "total_chars": total_chars,
            "total_words": total_words,
            "vocab_size": vocab_size,
            "ttr": round(ttr, 4),
            "hapax_ratio": round(hapax_ratio, 4),
            "char_entropy": round(float(char_entropy), 4),
            "avg_text_len": round(np.mean(self.text_lengths), 1) if self.text_lengths else 0,
            "avg_word_len": round(np.mean(self.word_lengths), 2) if self.word_lengths else 0,
            "median_text_len": round(np.median(self.text_lengths), 1) if self.text_lengths else 0,
            "top_chars": dict(self.char_freq.most_common(20)),
            "top_words": dict(self.word_freq.most_common(30)),
            "top_bigrams": dict(self.bigram_freq.most_common(20)),
            "top_trigrams": dict(self.trigram_freq.most_common(20)),
        }


# ─────────────────────────────────────────────
# Corpus Analyser
# ─────────────────────────────────────────────

class CorpusAnalyzer:
    """
    Corpus-level analysis across all languages in the Pralekha dataset.
    """

    def __init__(self):
        self.profiles: Dict[str, LanguageProfile] = {
            lang: LanguageProfile(lang) for lang in ALL_LANGUAGE_CODES
        }

    def process_dataframe(self, df: pd.DataFrame, show_progress: bool = True) -> "CorpusAnalyzer":
        """Process all texts in a labelled DataFrame."""
        iterator = tqdm(df.iterrows(), total=len(df)) if show_progress else df.iterrows()
        for _, row in iterator:
            lang = row.get("lang", "")
            text = row.get("text", "")
            if lang in self.profiles and text:
                self.profiles[lang].update(text)
        return self

    def get_all_stats(self) -> pd.DataFrame:
        rows = []
        for lang, profile in self.profiles.items():
            if profile.n_documents > 0:
                rows.append(profile.compute_stats())
        return pd.DataFrame(rows).sort_values("n_documents", ascending=False)

    def compute_cross_lingual_similarity(self) -> pd.DataFrame:
        """
        Character n-gram cosine similarity between language profiles.
        Higher = more similar script/character usage.
        """
        active = [(lang, p) for lang, p in self.profiles.items() if p.n_documents > 0]
        langs = [a[0] for a in active]
        n = len(langs)
        sim_matrix = np.zeros((n, n))

        # Build bigram vectors
        all_bigrams = set()
        for _, p in active:
            all_bigrams.update(p.bigram_freq.keys())
        all_bigrams = sorted(all_bigrams)
        bg_idx = {bg: i for i, bg in enumerate(all_bigrams)}

        vectors = []
        for _, p in active:
            total = sum(p.bigram_freq.values()) or 1
            vec = np.zeros(len(all_bigrams))
            for bg, cnt in p.bigram_freq.items():
                if bg in bg_idx:
                    vec[bg_idx[bg]] = cnt / total
            vectors.append(vec)

        for i in range(n):
            for j in range(n):
                sim_matrix[i, j] = self._cosine_sim(vectors[i], vectors[j])

        df = pd.DataFrame(sim_matrix, index=langs, columns=langs)
        return df

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    def find_confusable_languages(self, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """Identify language pairs with high n-gram similarity (hard to distinguish)."""
        sim_df = self.compute_cross_lingual_similarity()
        confusable = []
        langs = sim_df.index.tolist()
        for i, l1 in enumerate(langs):
            for j, l2 in enumerate(langs):
                if i >= j:
                    continue
                sim = sim_df.loc[l1, l2]
                if sim >= threshold:
                    confusable.append((l1, l2, round(sim, 4)))
        return sorted(confusable, key=lambda x: -x[2])

    def compute_vocabulary_richness(self) -> pd.DataFrame:
        """Guiraud, Herdan, TTR metrics per language."""
        rows = []
        for lang, p in self.profiles.items():
            if p.n_documents == 0:
                continue
            V = len(p.word_freq)  # vocab size
            N = sum(p.word_freq.values())  # total tokens
            if N == 0:
                continue
            rows.append({
                "lang": lang,
                "name": p.info.name if p.info else lang,
                "V_vocab": V,
                "N_tokens": N,
                "ttr": V / N,
                "guiraud": V / np.sqrt(N),
                "herdan": np.log(V) / np.log(N) if N > 1 else 0,
                "yule_k": self._yule_k(p.word_freq),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _yule_k(freq: Counter) -> float:
        """Yule's K measure of vocabulary diversity."""
        V = Counter(freq.values())
        N = sum(freq.values())
        M1 = sum(v * c for v, c in V.items())
        M2 = sum((v ** 2) * c for v, c in V.items())
        if M1 == 0:
            return 0.0
        return float(10000 * (M2 - M1) / (M1 ** 2))

    def save_profiles(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            lang: p.compute_stats()
            for lang, p in self.profiles.items()
            if p.n_documents > 0
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.success(f"Profiles saved to {path}")


# ─────────────────────────────────────────────
# Error Analysis
# ─────────────────────────────────────────────

class ErrorAnalyzer:
    """
    Analyze misclassification patterns.
    """

    def __init__(self, true_labels: List[str], pred_labels: List[str], texts: List[str]):
        self.true = true_labels
        self.pred = pred_labels
        self.texts = texts
        self.errors = [
            (t, p, txt)
            for t, p, txt in zip(true_labels, pred_labels, texts)
            if t != p
        ]

    def error_summary(self) -> pd.DataFrame:
        confusion = Counter((t, p) for t, p, _ in self.errors)
        rows = []
        for (true, pred), count in confusion.most_common(50):
            rows.append({
                "true_lang": true,
                "pred_lang": pred,
                "count": count,
                "true_name": LANGUAGE_REGISTRY.get(true, None) and LANGUAGE_REGISTRY[true].name,
                "pred_name": LANGUAGE_REGISTRY.get(pred, None) and LANGUAGE_REGISTRY[pred].name,
            })
        return pd.DataFrame(rows)

    def sample_errors(self, true_lang: str, pred_lang: str, n: int = 5) -> List[Dict]:
        samples = [
            {"true": t, "pred": p, "text": txt[:200]}
            for t, p, txt in self.errors
            if t == true_lang and p == pred_lang
        ]
        return samples[:n]

    def per_language_error_rate(self) -> pd.DataFrame:
        total = Counter(self.true)
        errors = Counter(t for t, p, _ in self.errors)
        rows = []
        for lang in sorted(total):
            rows.append({
                "lang": lang,
                "total": total[lang],
                "errors": errors.get(lang, 0),
                "error_rate": errors.get(lang, 0) / max(total[lang], 1),
            })
        return pd.DataFrame(rows).sort_values("error_rate", ascending=False)
