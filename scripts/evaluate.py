"""
Evaluation & Metrics Module
Comprehensive evaluation utilities: confusion matrix, ECE calibration,
per-language breakdowns, error analysis, and report generation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from loguru import logger

from src.utils.languages import LANGUAGE_REGISTRY


# ─────────────────────────────────────────────
# Core Metric Functions
# ─────────────────────────────────────────────

def compute_all_metrics(
    true_labels: List[str],
    pred_labels: List[str],
    confidences: Optional[List[float]] = None,
) -> Dict:
    """Compute a full suite of classification metrics."""
    classes = sorted(set(true_labels))

    accuracy = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
    macro_prec = precision_score(true_labels, pred_labels, average="macro", zero_division=0)
    macro_recall = recall_score(true_labels, pred_labels, average="macro", zero_division=0)

    per_class = classification_report(
        true_labels, pred_labels,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(true_labels, pred_labels, labels=classes)

    metrics = {
        "accuracy": round(accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "macro_precision": round(macro_prec, 6),
        "macro_recall": round(macro_recall, 6),
        "n_samples": len(true_labels),
        "n_classes": len(classes),
        "classes": classes,
        "per_language": per_class,
        "confusion_matrix": cm.tolist(),
    }

    if confidences is not None:
        conf_arr = np.array(confidences)
        correct = np.array(true_labels) == np.array(pred_labels)
        metrics["ece"] = round(compute_ece(conf_arr, correct), 6)
        metrics["mean_confidence"] = round(float(conf_arr.mean()), 4)
        metrics["high_confidence_ratio"] = round(float((conf_arr >= 0.9).mean()), 4)

    return metrics


def compute_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = correct[mask].mean()
        ece += (mask.sum() / len(confidences)) * abs(avg_conf - avg_acc)
    return float(ece)


# ─────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────

def plot_confusion_matrix(
    true_labels: List[str],
    pred_labels: List[str],
    output_path: Optional[str] = None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (14, 12),
) -> plt.Figure:
    """
    Annotated confusion matrix with language names.
    """
    classes = sorted(set(true_labels) | set(pred_labels))
    lang_names = [
        f"{LANGUAGE_REGISTRY[c].name}\n({c})" if c in LANGUAGE_REGISTRY else c
        for c in classes
    ]

    cm = confusion_matrix(true_labels, pred_labels, labels=classes)
    if normalize:
        cm_disp = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    else:
        cm_disp = cm

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_disp,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=lang_names,
        yticklabels=lang_names,
        ax=ax,
        linewidths=0.5,
        linecolor="#dddddd",
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )
    ax.set_xlabel("Predicted Language", fontsize=13, labelpad=10)
    ax.set_ylabel("True Language", fontsize=13, labelpad=10)
    ax.set_title(
        f"Confusion Matrix {'(Normalized)' if normalize else '(Counts)'}",
        fontsize=15, fontweight="bold", pad=15,
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.success(f"Confusion matrix saved: {output_path}")

    return fig


def plot_per_language_f1(
    metrics: Dict,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Bar chart of per-language F1 scores."""
    per_lang = metrics.get("per_language", {})
    rows = []
    for lang in metrics.get("classes", []):
        if lang in per_lang:
            info = LANGUAGE_REGISTRY.get(lang)
            rows.append({
                "lang": lang,
                "name": info.name if info else lang,
                "script": info.script if info else "?",
                "f1": per_lang[lang].get("f1-score", 0),
                "precision": per_lang[lang].get("precision", 0),
                "recall": per_lang[lang].get("recall", 0),
            })
    df = pd.DataFrame(rows).sort_values("f1", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#2ecc71" if f > 0.95 else "#f39c12" if f > 0.85 else "#e74c3c" for f in df["f1"]]

    bars = ax.barh(
        [f"{r['name']}\n({r['lang']})" for _, r in df.iterrows()],
        df["f1"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
    )

    for bar, val in zip(bars, df["f1"]):
        ax.text(
            min(val + 0.005, 0.995), bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=9, fontweight="bold",
        )

    ax.axvline(x=metrics.get("macro_f1", 0), color="#3498db", linestyle="--", lw=2, label=f"Macro F1 = {metrics.get('macro_f1', 0):.3f}")
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("F1 Score", fontsize=12)
    ax.set_title("Per-Language F1 Scores", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)

    patches = [
        mpatches.Patch(color="#2ecc71", label="F1 ≥ 0.95"),
        mpatches.Patch(color="#f39c12", label="0.85 ≤ F1 < 0.95"),
        mpatches.Patch(color="#e74c3c", label="F1 < 0.85"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confidence_histogram(
    confidences: List[float],
    correct: List[bool],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Reliability diagram / calibration plot."""
    conf_arr = np.array(confidences)
    correct_arr = np.array(correct)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # ── Left: Histogram of confidences ────────────────────────────────────────
    ax1.hist(conf_arr[correct_arr], bins=30, alpha=0.7, color="#2ecc71", label="Correct")
    ax1.hist(conf_arr[~correct_arr], bins=30, alpha=0.7, color="#e74c3c", label="Incorrect")
    ax1.set_xlabel("Confidence", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Confidence Distribution", fontsize=12, fontweight="bold")
    ax1.legend()

    # ── Right: Reliability diagram ────────────────────────────────────────────
    n_bins = 15
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    accuracies, mean_confs, counts = [], [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf_arr >= lo) & (conf_arr < hi)
        if mask.sum() == 0:
            accuracies.append(np.nan)
            mean_confs.append((lo + hi) / 2)
            counts.append(0)
        else:
            accuracies.append(correct_arr[mask].mean())
            mean_confs.append(conf_arr[mask].mean())
            counts.append(mask.sum())

    ax2.plot([0, 1], [0, 1], "--", color="gray", lw=1.5, label="Perfect calibration")
    ax2.plot(mean_confs, accuracies, "o-", color="#3498db", lw=2, ms=6, label="Model")
    ax2.fill_between(mean_confs, mean_confs, accuracies, alpha=0.15, color="#e74c3c")
    ax2.set_xlabel("Mean Predicted Confidence", fontsize=11)
    ax2.set_ylabel("Actual Accuracy", fontsize=11)
    ax2.set_title("Reliability Diagram", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_script_similarity_heatmap(
    similarity_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of cross-lingual character n-gram similarity."""
    # Rename index/columns to language names
    names = {
        code: LANGUAGE_REGISTRY[code].name if code in LANGUAGE_REGISTRY else code
        for code in similarity_df.index
    }
    df_named = similarity_df.rename(index=names, columns=names)

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.eye(len(df_named), dtype=bool)
    sns.heatmap(
        df_named,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=ax,
        mask=mask,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Cosine Similarity (char bigrams)"},
    )
    ax.set_title(
        "Cross-Lingual Script Similarity\n(Higher = harder to distinguish)",
        fontsize=14, fontweight="bold", pad=15,
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────
# Report Generator
# ─────────────────────────────────────────────

class EvaluationReporter:
    """
    Generates comprehensive evaluation reports (JSON + HTML + plots).
    """

    def __init__(self, output_dir: str = "./artifacts/eval"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_full_report(
        self,
        model_name: str,
        true_labels: List[str],
        pred_labels: List[str],
        confidences: Optional[List[float]] = None,
        texts: Optional[List[str]] = None,
    ) -> Dict:
        """Generate all evaluation artifacts for a model."""
        logger.info(f"Generating evaluation report for: {model_name}")

        # Core metrics
        metrics = compute_all_metrics(true_labels, pred_labels, confidences)

        # Save JSON
        json_path = self.output_dir / f"{model_name}_metrics.json"
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics JSON: {json_path}")

        # Confusion matrix
        plot_confusion_matrix(
            true_labels, pred_labels,
            output_path=str(self.output_dir / f"{model_name}_confusion_matrix.png"),
        )

        # Per-language F1
        plot_per_language_f1(
            metrics,
            output_path=str(self.output_dir / f"{model_name}_f1_per_lang.png"),
        )

        # Calibration (if confidences available)
        if confidences:
            correct = [t == p for t, p in zip(true_labels, pred_labels)]
            plot_confidence_histogram(
                confidences, correct,
                output_path=str(self.output_dir / f"{model_name}_calibration.png"),
            )

        # HTML report
        self._generate_html_report(model_name, metrics)

        logger.success(f"Report saved to: {self.output_dir}")
        return metrics

    def _generate_html_report(self, model_name: str, metrics: Dict):
        """Generate an HTML evaluation summary report."""
        per_lang = metrics.get("per_language", {})
        classes = metrics.get("classes", [])

        lang_rows = ""
        for lang in classes:
            if lang not in per_lang:
                continue
            info = LANGUAGE_REGISTRY.get(lang)
            f1 = per_lang[lang].get("f1-score", 0)
            prec = per_lang[lang].get("precision", 0)
            rec = per_lang[lang].get("recall", 0)
            support = per_lang[lang].get("support", 0)
            color = "#2ecc71" if f1 >= 0.95 else "#f39c12" if f1 >= 0.85 else "#e74c3c"
            lang_rows += f"""
            <tr>
                <td>{info.name if info else lang}</td>
                <td><code>{lang}</code></td>
                <td>{info.script if info else '?'}</td>
                <td>{prec:.4f}</td>
                <td>{rec:.4f}</td>
                <td style="color:{color};font-weight:bold">{f1:.4f}</td>
                <td>{int(support):,}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LID Evaluation: {model_name}</title>
<style>
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e0e0e0; margin: 40px; }}
  h1 {{ color: #00d4aa; }} h2 {{ color: #4a9eff; border-bottom: 1px solid #333; padding-bottom: 6px; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 20px 0; }}
  .metric-card {{ background: #1a1d2e; border: 1px solid #2d3148; border-radius: 12px; padding: 20px; text-align: center; }}
  .metric-value {{ font-size: 2em; font-weight: bold; color: #00d4aa; }}
  .metric-label {{ font-size: 0.85em; color: #888; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
  th {{ background: #1a1d2e; padding: 10px; text-align: left; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #222; }}
  tr:hover {{ background: #1a1d2e; }}
  code {{ background: #222; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }}
  img {{ max-width: 100%; border-radius: 8px; margin: 10px 0; }}
</style>
</head>
<body>
<h1>🔤 Language ID Evaluation Report</h1>
<p><strong>Model:</strong> {model_name} &nbsp;|&nbsp; <strong>Samples:</strong> {metrics['n_samples']:,} &nbsp;|&nbsp; <strong>Languages:</strong> {metrics['n_classes']}</p>

<h2>Overall Metrics</h2>
<div class="metric-grid">
  <div class="metric-card"><div class="metric-value">{metrics['accuracy']:.4f}</div><div class="metric-label">Accuracy</div></div>
  <div class="metric-card"><div class="metric-value">{metrics['macro_f1']:.4f}</div><div class="metric-label">Macro F1</div></div>
  <div class="metric-card"><div class="metric-value">{metrics['weighted_f1']:.4f}</div><div class="metric-label">Weighted F1</div></div>
  <div class="metric-card"><div class="metric-value">{metrics['macro_precision']:.4f}</div><div class="metric-label">Macro Precision</div></div>
  <div class="metric-card"><div class="metric-value">{metrics['macro_recall']:.4f}</div><div class="metric-label">Macro Recall</div></div>
  {f'<div class="metric-card"><div class="metric-value">{metrics["ece"]:.4f}</div><div class="metric-label">ECE (↓ better)</div></div>' if 'ece' in metrics else ''}
</div>

<h2>Per-Language Results</h2>
<table>
<tr><th>Language</th><th>Code</th><th>Script</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>
{lang_rows}
</table>

<h2>Confusion Matrix</h2>
<img src="{model_name}_confusion_matrix.png" alt="Confusion Matrix">

<h2>Per-Language F1</h2>
<img src="{model_name}_f1_per_lang.png" alt="F1 per language">

{"<h2>Confidence Calibration</h2><img src='" + model_name + "_calibration.png' alt='Calibration'>" if 'ece' in metrics else ''}

<p style="color:#555; font-size:0.85em; margin-top:40px">
  Generated by Indic Language ID System · AI4Bharat/Pralekha Dataset
</p>
</body>
</html>"""

        html_path = self.output_dir / f"{model_name}_report.html"
        html_path.write_text(html, encoding="utf-8")
        logger.info(f"HTML report: {html_path}")
