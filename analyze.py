"""
CLI for quick language identification and script analysis.

Usage:
    python scripts/analyze.py --text "नमस्ते दुनिया"
    python scripts/analyze.py --file input.txt
    python scripts/analyze.py --stdin   # pipe text
    python scripts/analyze.py --demo    # run on all sample texts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from src.analysis.script_analyzer import ScriptAnalyzer
from src.data.preprocessor import IndicTextPreprocessor
from src.utils.languages import LANGUAGE_REGISTRY

console = Console()
preprocessor = IndicTextPreprocessor()
analyzer = ScriptAnalyzer()

SAMPLE_TEXTS = [
    ("Hindi",     "भारत एक लोकतांत्रिक देश है। यहाँ कई भाषाएँ बोली जाती हैं।"),
    ("Bengali",   "বাংলা ভাষা ও সাহিত্যের ঐতিহ্য অনেক প্রাচীন।"),
    ("Tamil",     "தமிழ் மொழி மிகவும் பழமையான மொழியாகும்."),
    ("Telugu",    "తెలుగు భాష చాలా అందమైన భాష."),
    ("Malayalam", "മലയാളം ഒരു ദ്രാവിഡ ഭാഷയാണ്."),
    ("Kannada",   "ಕನ್ನಡ ಭಾಷೆ ಕರ್ನಾಟಕದ ರಾಜ್ಯ ಭಾಷೆ."),
    ("Gujarati",  "ગુજરાતી ભાષા ઈન્ડો-આર્યન ભાષા-પ્રજા સ્nt."),
    ("Marathi",   "मराठी महाराष्ट्राची राजभाषा आहे."),
    ("Punjabi",   "ਪੰਜਾਬੀ ਉੱਤਰ ਭਾਰਤ ਵਿੱਚ ਬੋਲੀ ਜਾਂਦੀ ਭਾਸ਼ਾ ਹੈ।"),
    ("Urdu",      "اردو ایک خوبصورت زبان ہے۔"),
    ("Odia",      "ଓଡ଼ିଆ ଓଡ଼ିଶାର ରାଜ୍ୟ ଭାଷା।"),
    ("English",   "The diversity of languages in India is remarkable."),
]


def load_model():
    for mpath in ["./artifacts/models/ensemble.pkl", "./artifacts/models/ngram_model.pkl"]:
        if Path(mpath).exists():
            try:
                import pickle
                with open(mpath, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
    from src.models.ensemble_model import EnsembleLIDModel
    return EnsembleLIDModel()


def analyze_text(text: str, model, show_script: bool = True):
    """Full analysis of a single text."""
    processed = preprocessor.preprocess(text)

    # Language ID
    if hasattr(model, "predict_with_confidence"):
        result = model.predict_with_confidence([processed])[0]
        lang = result.get("predicted_lang", "?")
        conf = result.get("confidence", 0.0)
        top5 = result.get("top_5", [])
    elif hasattr(model, "predict"):
        lang = model.predict([processed])[0]
        conf = 0.9
        top5 = [{"lang": lang, "probability": conf}]
    else:
        lang, conf = "eng", 0.5
        top5 = []

    info = LANGUAGE_REGISTRY.get(lang)
    name = info.name if info else lang
    script = info.script if info else "?"
    direction = info.direction if info else "LTR"
    native = info.native_name if info else ""

    # Display
    confidence_bar = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
    confidence_color = "green" if conf > 0.8 else "yellow" if conf > 0.6 else "red"

    panel_content = Text()
    panel_content.append(f"🌐 Language:  ", style="bold")
    panel_content.append(f"{name}", style="bold cyan")
    panel_content.append(f" ({lang})\n")
    panel_content.append(f"📝 Native:    {native}\n")
    panel_content.append(f"📜 Script:    {script}\n")
    panel_content.append(f"↔️  Direction: {direction}\n")
    panel_content.append(f"✅ Confidence: [{confidence_bar}] ", style="bold")
    panel_content.append(f"{conf:.2%}\n", style=confidence_color)

    console.print(Panel(panel_content, title="🔤 Language Identification", border_style="cyan"))

    if top5 and len(top5) > 1:
        table = Table(title="Top-5 Alternatives", box=box.SIMPLE)
        table.add_column("Rank", style="dim")
        table.add_column("Language")
        table.add_column("Code")
        table.add_column("Script")
        table.add_column("Probability", justify="right")

        for i, t in enumerate(top5[:5]):
            l = t.get("lang", "?")
            p = t.get("probability", 0.0)
            li = LANGUAGE_REGISTRY.get(l)
            bar = "█" * int(p * 15) + "░" * (15 - int(p * 15))
            table.add_row(
                str(i + 1),
                li.name if li else l,
                l,
                li.script if li else "?",
                f"{bar} {p:.2%}",
                style="bold" if i == 0 else "",
            )
        console.print(table)

    if show_script:
        sa = analyzer.analyze(text)
        script_table = Table(title="📊 Script Analysis", box=box.SIMPLE)
        script_table.add_column("Metric")
        script_table.add_column("Value")

        rows = [
            ("Total chars", f"{sa.total_chars:,}"),
            ("Alphabetic chars", f"{sa.alpha_chars:,} ({sa.alpha_chars/max(sa.total_chars,1):.1%})"),
            ("Dominant script", f"{sa.dominant_script} ({sa.dominant_script_ratio:.1%})"),
            ("Mixed script?", "⚠️ YES" if sa.is_mixed_script else "✅ No"),
            ("Script transitions", str(sa.script_transitions)),
            ("Char entropy", f"{sa.char_entropy:.3f} bits"),
            ("Unique chars", str(sa.unique_chars)),
            ("Noise score", f"{sa.noise_score:.3f} {'⚠️ High' if sa.noise_score > 0.3 else '✅ Low'}"),
            ("Text direction", f"{sa.primary_direction} {'(BiDi)' if sa.has_bidi else ''}"),
            ("Words", str(len(text.split()))),
        ]
        for k, v in rows:
            script_table.add_row(k, v)

        console.print(script_table)

        if sa.script_distribution:
            console.print("\n[bold]Script distribution:[/bold]")
            for sc, ratio in sorted(sa.script_distribution.items(), key=lambda x: -x[1])[:5]:
                bar = "█" * int(ratio * 30)
                console.print(f"  {sc:<20} {bar} {ratio:.1%}")

    console.print()


def main():
    p = argparse.ArgumentParser(description="Language ID & Script Analysis CLI")
    p.add_argument("--text", type=str, help="Text to analyze")
    p.add_argument("--file", type=str, help="Text file to analyze")
    p.add_argument("--stdin", action="store_true")
    p.add_argument("--demo", action="store_true", help="Run on all sample texts")
    p.add_argument("--no-script", action="store_true", help="Skip script analysis")
    p.add_argument("--json", action="store_true", help="Output JSON")
    args = p.parse_args()

    console.print("[bold cyan]🔤 Indic Language ID System[/bold cyan]")
    console.print("[dim]Loading model ...[/dim]")
    model = load_model()
    console.print("[green]✓ Ready[/green]\n")

    show_script = not args.no_script

    if args.demo:
        console.rule("Running on all sample texts")
        for lang, text in SAMPLE_TEXTS:
            console.print(f"\n[dim]--- {lang} sample ---[/dim]")
            console.print(f"[italic]{text[:80]}[/italic]")
            analyze_text(text, model, show_script=show_script)
    elif args.text:
        analyze_text(args.text, model, show_script=show_script)
    elif args.file:
        content = Path(args.file).read_text(encoding="utf-8")
        analyze_text(content, model, show_script=show_script)
    elif args.stdin:
        content = sys.stdin.read()
        analyze_text(content, model, show_script=show_script)
    else:
        # Interactive mode
        console.print("Interactive mode. Type text and press Enter. Ctrl+C to quit.\n")
        while True:
            try:
                text = input(">>> ").strip()
                if text:
                    analyze_text(text, model, show_script=show_script)
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Bye![/yellow]")
                break


if __name__ == "__main__":
    main()
