"""
Main Training Pipeline
Orchestrates data loading → preprocessing → training → evaluation → saving.

Usage:
    python scripts/train_pipeline.py --config configs/default.yaml
    python scripts/train_pipeline.py --model ngram --max_per_lang 20000
    python scripts/train_pipeline.py --model all --eval
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import track

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import PralekhaLoader
from src.data.preprocessor import IndicTextPreprocessor, PreprocessorConfig
from src.models.ngram_model import CharNgramLIDModel
from src.models.fasttext_model import FastTextLIDModel
from src.models.ensemble_model import EnsembleLIDModel, UnicodeHeuristicModel
from src.analysis.corpus_stats import CorpusAnalyzer
from src.utils.languages import ALL_LANGUAGE_CODES

console = Console()


def parse_args():
    p = argparse.ArgumentParser(description="Language ID Training Pipeline")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", choices=["ngram", "fasttext", "ensemble", "all"], default="all")
    p.add_argument("--max_per_lang", type=int, default=30_000)
    p.add_argument("--eval", action="store_true", default=True)
    p.add_argument("--output_dir", default="./artifacts/models")
    p.add_argument("--data_dir", default="./data/splits")
    p.add_argument("--skip_data_prep", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def banner():
    console.print("""
[bold cyan]╔══════════════════════════════════════════════════════╗
║   🔤 Indic Language Identification Training Pipeline   ║
║   Dataset: AI4Bharat/Pralekha (12 Indic + English)    ║
╚══════════════════════════════════════════════════════╝[/bold cyan]
""")


def step_data_preparation(args) -> tuple:
    """Load and prepare dataset splits."""
    console.rule("[bold green]Step 1: Data Preparation")

    data_path = Path(args.data_dir)
    if args.skip_data_prep and (data_path / "train.parquet").exists():
        console.print("[yellow]Loading existing splits ...")
        train, val, test = PralekhaLoader.load_splits(args.data_dir)
    else:
        console.print(f"[cyan]Loading Pralekha dataset (max {args.max_per_lang:,}/lang) ...")
        loader = PralekhaLoader(
            max_per_lang=args.max_per_lang,
            seed=args.seed,
        )
        df = loader.load_language_labelled_dataframe(balanced=True)

        # Print stats
        stats = loader.get_dataset_stats(df)
        table = Table(title="Dataset Statistics", show_header=True)
        for col in ["lang", "name", "script", "n_samples", "avg_len"]:
            table.add_column(col)
        for _, row in stats.iterrows():
            table.add_row(
                row["lang"], row["name"], row["script"],
                f"{row['n_samples']:,}", f"{row['avg_len']:.0f}",
            )
        console.print(table)

        preprocessor = IndicTextPreprocessor(PreprocessorConfig(
            normalize_unicode=True,
            remove_urls=True,
            remove_html_tags=True,
            max_length=2000,
        ))
        console.print("[cyan]Preprocessing texts ...")
        df["text"] = preprocessor.batch_preprocess(df["text"].tolist())
        df = df[df["text"].str.len() >= 30].reset_index(drop=True)

        train, val, test = loader.create_train_val_test_split(df)
        loader.export_splits(train, val, test, args.data_dir)

    console.print(
        f"[green]✓ Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}"
    )
    return train, val, test


def step_corpus_analysis(train_df) -> CorpusAnalyzer:
    """Analyze corpus statistics."""
    console.rule("[bold green]Step 2: Corpus Analysis")
    analyzer = CorpusAnalyzer()
    analyzer.process_dataframe(train_df)

    stats = analyzer.get_all_stats()
    console.print(f"[cyan]Corpus analysis complete. {len(stats)} languages profiled.")

    richness = analyzer.compute_vocabulary_richness()
    console.print("\n[bold]Vocabulary Richness:[/bold]")
    console.print(richness[["lang", "name", "vocab_size", "ttr", "guiraud"]].to_string(index=False))

    confusable = analyzer.find_confusable_languages(threshold=0.7)
    if confusable:
        console.print(f"\n[yellow]⚠ Confusable language pairs (cosine sim ≥ 0.7):")
        for l1, l2, sim in confusable[:5]:
            console.print(f"   {l1} ↔ {l2}: {sim:.3f}")

    out_path = Path("./artifacts/analysis/corpus_profiles.json")
    analyzer.save_profiles(str(out_path))
    return analyzer


def step_train_ngram(train, val) -> CharNgramLIDModel:
    """Train character n-gram model."""
    console.rule("[bold green]Step 3a: CharNgram Model")
    model = CharNgramLIDModel(
        ngram_range=(2, 4),
        max_features=150_000,
        C=5.0,
        calibrate=True,
    )
    model.fit(train, val_df=val)

    # Show top discriminative features
    console.print("\n[bold]Top discriminative n-grams per language:[/bold]")
    top_feats = model.get_top_features_per_language(n=10)
    for lang, feats in list(top_feats.items())[:4]:
        console.print(f"  [cyan]{lang}[/cyan]: {feats[:5]}")

    return model


def step_train_fasttext(train, val) -> FastTextLIDModel:
    """Train FastText model."""
    console.rule("[bold green]Step 3b: FastText Model")
    try:
        import fasttext
        model = FastTextLIDModel(
            mode="custom",
            dim=100,
            epoch=25,
            lr=0.1,
            word_ngrams=3,
        )
        model.fit(train, val_df=val)
        return model
    except ImportError:
        console.print("[yellow]fasttext not installed, loading pretrained instead ...")
        model = FastTextLIDModel(mode="pretrained")
        model.load_pretrained()
        return model


def step_build_ensemble(ngram_model, fasttext_model, val, output_dir) -> EnsembleLIDModel:
    """Build ensemble model."""
    console.rule("[bold green]Step 4: Ensemble Model")
    ensemble = EnsembleLIDModel(
        ngram_model=ngram_model,
        fasttext_model=fasttext_model,
        weights={"ngram": 0.35, "fasttext": 0.50},
        strategy="weighted_vote",
    )
    # Fit stacking on val
    try:
        ensemble.fit_stacking(val)
    except Exception as e:
        console.print(f"[yellow]Stacking skipped: {e}")

    import pickle
    with open(Path(output_dir) / "ensemble.pkl", "wb") as f:
        pickle.dump(ensemble, f)
    console.print(f"[green]✓ Ensemble saved")
    return ensemble


def step_evaluate(models: dict, test):
    """Evaluate all models."""
    console.rule("[bold green]Step 5: Evaluation")
    results_summary = {}

    for name, model in models.items():
        console.print(f"\n[bold cyan]Evaluating {name} ...[/bold cyan]")
        t0 = time.time()
        try:
            if hasattr(model, "evaluate"):
                r = model.evaluate(test)
                elapsed = time.time() - t0
                results_summary[name] = {
                    "accuracy": r.get("accuracy", 0),
                    "macro_f1": r.get("macro_f1", 0),
                    "inference_time_s": elapsed,
                }
        except Exception as e:
            console.print(f"[red]Eval failed for {name}: {e}")

    # Summary table
    table = Table(title="Model Comparison", show_header=True)
    table.add_column("Model")
    table.add_column("Accuracy")
    table.add_column("Macro F1")
    table.add_column("Inference (s)")

    for name, r in results_summary.items():
        table.add_row(
            name,
            f"{r['accuracy']:.4f}",
            f"{r['macro_f1']:.4f}",
            f"{r['inference_time_s']:.2f}s",
        )
    console.print(table)

    out = Path("./artifacts/eval/results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results_summary, f, indent=2)
    console.print(f"[green]✓ Results saved to {out}")


def main():
    banner()
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Data
    train, val, test = step_data_preparation(args)

    # Step 2: Corpus Analysis
    step_corpus_analysis(train)

    trained_models = {}

    # Step 3: Train Models
    if args.model in ("ngram", "all"):
        ngram = step_train_ngram(train, val)
        ngram.save(str(output_dir / "ngram_model.pkl"))
        trained_models["CharNgram"] = ngram
    else:
        ngram = None

    if args.model in ("fasttext", "all"):
        ft = step_train_fasttext(train, val)
        trained_models["FastText"] = ft
    else:
        ft = None

    if args.model in ("ensemble", "all") and (ngram or ft):
        ensemble = step_build_ensemble(ngram, ft, val, output_dir)
        trained_models["Ensemble"] = ensemble

    # Step 4: Evaluate
    if args.eval and trained_models:
        step_evaluate(trained_models, test)

    console.print("\n[bold green]🎉 Pipeline complete![/bold green]")
    console.print(f"Models saved in: {output_dir}")
    console.print("Start API: [bold]uvicorn src.api.server:app --reload --port 8000[/bold]")
    console.print("Open dashboard: [bold]python scripts/dashboard.py[/bold]")


if __name__ == "__main__":
    main()
