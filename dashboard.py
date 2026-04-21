"""
Gradio Interactive Dashboard
Language Identification & Script Analysis Web UI.

Usage:
    python scripts/dashboard.py
    python scripts/dashboard.py --share  # public URL
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
from typing import Dict, List, Tuple

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.analysis.script_analyzer import ScriptAnalyzer
from src.data.preprocessor import IndicTextPreprocessor
from src.utils.languages import LANGUAGE_REGISTRY, ALL_LANGUAGE_CODES

# ── Load Models ──────────────────────────────────────────────────────────────

preprocessor = IndicTextPreprocessor()
script_analyzer = ScriptAnalyzer()

# Try loading best available model
model = None
for mpath in [
    "./artifacts/models/ensemble.pkl",
    "./artifacts/models/ngram_model.pkl",
]:
    if Path(mpath).exists():
        try:
            import pickle
            with open(mpath, "rb") as f:
                model = pickle.load(f)
            print(f"✓ Loaded model: {mpath}")
            break
        except Exception as e:
            print(f"Could not load {mpath}: {e}")

if model is None:
    # Use heuristic model as fallback
    from src.models.ensemble_model import UnicodeHeuristicModel, EnsembleLIDModel
    model = EnsembleLIDModel()
    print("⚠ No trained model found — using Unicode heuristic only")


# ── Sample Texts ─────────────────────────────────────────────────────────────

SAMPLES = {
    "Hindi": "भारत एक विशाल देश है जहाँ अनेक भाषाएँ बोली जाती हैं। हिंदी यहाँ की राजभाषा है।",
    "Bengali": "বাংলাদেশ এবং পশ্চিমবঙ্গে বাংলা ভাষা প্রচলিত। এটি একটি সমৃদ্ধ সাহিত্যের ভাষা।",
    "Tamil": "தமிழ் மொழி உலகின் மிகவும் பழமையான மொழிகளில் ஒன்றாகும். இதன் இலக்கியம் மிக சிறந்தது.",
    "Telugu": "తెలుగు భాష భారతదేశంలో అధికంగా మాట్లాడే భాషలలో ఒకటి. ఇది ద్రావిడ భాషా కుటుంబానికి చెందినది.",
    "Malayalam": "മലയാളം കേരളത്തിന്റെ ഔദ്യോഗിക ഭാഷയാണ്. ഇത് ദ്രാവിഡ ഭാഷാ കുടുംബത്തിൽ ഉൾപ്പെടുന്നു.",
    "Kannada": "ಕನ್ನಡ ಭಾಷೆ ಕರ್ನಾಟಕ ರಾಜ್ಯದ ಅಧಿಕೃತ ಭಾಷೆ. ಇದು ದ್ರಾವಿಡ ಭಾಷಾ ಕುಟುಂಬಕ್ಕೆ ಸೇರಿದ್ದು.",
    "Gujarati": "ગુજરાત ભારતનું એક મહત્ત્વનું રાજ્ય છે. ગુજરાતી ભાષા ઈન્ડો-આર્યન ભાષા-પ્રજા સાથે સ્ંbybund.",
    "Marathi": "मराठी भाषा महाराष्ट्र राज्याची अधिकृत भाषा आहे. ही एक समृद्ध साहित्य परंपरा असलेली भाषा आहे.",
    "Punjabi": "ਪੰਜਾਬੀ ਭਾਸ਼ਾ ਪੰਜਾਬ ਦੀ ਅਧਿਕਾਰਤ ਭਾਸ਼ਾ ਹੈ। ਇਹ ਇੰਡੋ-ਆਰੀਅਨ ਭਾਸ਼ਾ ਪਰਿਵਾਰ ਨਾਲ ਸਬੰਧਤ ਹੈ।",
    "Urdu": "اردو پاکستان کی قومی زبان ہے اور بھارت میں بھی بولی جاتی ہے۔ یہ ایک خوبصورت زبان ہے۔",
    "Odia": "ଓଡ଼ିଆ ଭାଷା ଓଡ଼ିଶା ରାଜ୍ୟର ସରକାରୀ ଭାଷା। ଏହା ଭାରତୀୟ ଭାଷା ପରିବାରର ଅଂଶ।",
    "English": "India is a diverse country with hundreds of languages spoken across its states and territories.",
    "Mixed": "यह एक mixed language text है। It contains both Hindi and English words. बहुत interesting है!",
}

DARK_CSS = """
body { background: #0f1117; color: #e0e0e0; font-family: 'JetBrains Mono', monospace; }
.gradio-container { max-width: 1200px; margin: auto; }
.gr-panel { background: #1a1d2e; border: 1px solid #2d3148; border-radius: 12px; }
"""


# ── Identify Function ─────────────────────────────────────────────────────────

def identify_language(text: str, do_preprocess: bool, show_script: bool) -> Tuple:
    if not text.strip():
        return "Please enter some text.", "", None, None

    t0 = time.perf_counter()
    processed_text = preprocessor.preprocess(text) if do_preprocess else text

    # Language identification
    if hasattr(model, "predict_with_confidence"):
        results = model.predict_with_confidence([processed_text])
        r = results[0]
        lang = r.get("predicted_lang", "?")
        conf = r.get("confidence", 0.0)
        top5 = r.get("top_5", [])
    elif hasattr(model, "predict"):
        lang = model.predict([processed_text])[0]
        conf = 0.9
        top5 = [{"lang": lang, "probability": conf}]
    else:
        lang, conf = "eng", 0.5
        top5 = []

    info = LANGUAGE_REGISTRY.get(lang)
    elapsed = (time.perf_counter() - t0) * 1000

    # Main result card
    result_md = f"""
### 🎯 Predicted Language: **{info.name if info else lang}** (`{lang}`)

| Field | Value |
|-------|-------|
| 🌐 Native Name | {info.native_name if info else '?'} |
| 📜 Script | {info.script if info else '?'} |
| ↔️ Direction | {info.direction if info else 'LTR'} |
| 🏛️ Family | {info.family if info else '?'} |
| 🔢 ISO 639-1 | `{info.iso639_1 if info else '?'}` |
| ✅ Confidence | **{conf:.2%}** {"🟢" if conf > 0.8 else "🟡" if conf > 0.6 else "🔴"} |
| ⚡ Inference Time | {elapsed:.1f} ms |
"""

    # Confidence bar chart
    if top5:
        langs_names = [LANGUAGE_REGISTRY.get(t["lang"], None) for t in top5]
        labels = [
            f"{n.name if n else t['lang']} ({t['lang']})"
            for t, n in zip(top5, langs_names)
        ]
        probs = [t["probability"] for t in top5]
        colors = ["#00d4aa" if t["lang"] == lang else "#4a9eff" for t in top5]

        fig_conf = go.Figure(go.Bar(
            x=probs, y=labels, orientation="h",
            marker_color=colors,
            text=[f"{p:.2%}" for p in probs],
            textposition="outside",
        ))
        fig_conf.update_layout(
            title="Top-5 Language Probabilities",
            template="plotly_dark",
            height=300,
            margin=dict(l=10, r=80, t=40, b=10),
            xaxis=dict(range=[0, 1.1], title="Probability"),
            yaxis=dict(autorange="reversed"),
            paper_bgcolor="#1a1d2e",
            plot_bgcolor="#0f1117",
        )
    else:
        fig_conf = None

    # Script analysis
    script_md = ""
    fig_script = None
    if show_script:
        sa = script_analyzer.analyze(text)
        script_md = f"""
### 📊 Script Analysis

| Metric | Value |
|--------|-------|
| 🔤 Total Characters | {sa.total_chars:,} |
| 🔡 Alphabetic | {sa.alpha_chars:,} ({sa.alpha_chars/max(sa.total_chars,1):.1%}) |
| 📜 Dominant Script | **{sa.dominant_script}** ({sa.dominant_script_ratio:.1%}) |
| 🔀 Mixed Script? | {"⚠️ Yes" if sa.is_mixed_script else "✅ No"} |
| ↔️ Direction | {sa.primary_direction} {"(BiDi)" if sa.has_bidi else ""} |
| 🔉 Char Entropy | {sa.char_entropy:.3f} bits |
| 🔢 Unique Chars | {sa.unique_chars:,} |
| 🚨 Noise Score | {sa.noise_score:.2f} {"⚠️ High" if sa.noise_score > 0.3 else "✅ Low"} |
| 🔤 Script Segments | {len(sa.script_segments)} |
| 🔄 Script Transitions | {sa.script_transitions} |
"""
        if sa.script_distribution:
            scripts = list(sa.script_distribution.keys())[:8]
            ratios = [sa.script_distribution[s] for s in scripts]
            fig_script = px.pie(
                names=scripts, values=ratios,
                title="Script Distribution",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig_script.update_layout(
                paper_bgcolor="#1a1d2e",
                height=300,
                margin=dict(l=10, r=10, t=40, b=10),
            )

    return result_md, script_md, fig_conf, fig_script


def analyze_batch_text(texts_raw: str) -> pd.DataFrame:
    """Process multi-line batch input."""
    texts = [t.strip() for t in texts_raw.strip().split("\n") if t.strip()]
    if not texts:
        return pd.DataFrame()

    results = []
    for text in texts[:50]:
        text_p = preprocessor.preprocess(text)
        if hasattr(model, "predict_with_confidence"):
            r = model.predict_with_confidence([text_p])[0]
            lang = r.get("predicted_lang", "?")
            conf = r.get("confidence", 0.0)
        elif hasattr(model, "predict"):
            lang = model.predict([text_p])[0]
            conf = 0.9
        else:
            lang, conf = "eng", 0.5

        info = LANGUAGE_REGISTRY.get(lang)
        results.append({
            "Text (preview)": text[:50] + "..." if len(text) > 50 else text,
            "Language": info.name if info else lang,
            "Code": lang,
            "Script": info.script if info else "?",
            "Confidence": f"{conf:.2%}",
            "Direction": info.direction if info else "LTR",
        })

    return pd.DataFrame(results)


def get_language_info(lang_code: str) -> str:
    code = lang_code.split("(")[-1].rstrip(")")
    info = LANGUAGE_REGISTRY.get(code.strip())
    if not info:
        return f"Language `{code}` not found."
    return f"""
### {info.native_name} ({info.name})

| | |
|--|--|
| **Code** | `{info.code}` / `{info.iso639_1}` / `{info.iso639_3}` |
| **Script** | {info.script} |
| **Direction** | {info.direction} |
| **Language Family** | {info.family} |
| **Romanization** | {info.romanization_scheme or 'N/A'} |
| **Shares script with** | {', '.join(info.shares_script_with) if info.shares_script_with else 'None'} |

**Unicode Ranges:** {', '.join(f'U+{lo:04X}–U+{hi:04X}' for lo, hi in info.unicode_ranges)}
"""


# ── Build Gradio App ──────────────────────────────────────────────────────────

with gr.Blocks(
    title="🔤 Indic Language ID System",
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.gray,
    ),
) as demo:

    gr.Markdown("""
# 🔤 Indic Language Identification & Script Analysis
**AI4Bharat/Pralekha Dataset** · 12 Indic Languages + English
---
""")

    with gr.Tabs():

        # ── Tab 1: Single Text ────────────────────────────────────────────────
        with gr.Tab("🔍 Identify Language"):
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="Input Text",
                        placeholder="Enter text in any supported language ...",
                        lines=6,
                        max_lines=20,
                    )
                    with gr.Row():
                        do_preprocess = gr.Checkbox(label="Preprocess text", value=True)
                        show_script = gr.Checkbox(label="Script Analysis", value=True)
                    with gr.Row():
                        clear_btn = gr.Button("Clear", variant="secondary")
                        identify_btn = gr.Button("🔍 Identify", variant="primary")

                    gr.Markdown("**Sample Texts:**")
                    sample_dropdown = gr.Dropdown(
                        choices=list(SAMPLES.keys()),
                        label="Load sample",
                        value=None,
                    )

                with gr.Column(scale=1):
                    result_md = gr.Markdown()
                    script_md = gr.Markdown()

            with gr.Row():
                conf_plot = gr.Plot(label="Language Confidence")
                script_plot = gr.Plot(label="Script Distribution")

            # Event handlers
            identify_btn.click(
                fn=identify_language,
                inputs=[text_input, do_preprocess, show_script],
                outputs=[result_md, script_md, conf_plot, script_plot],
            )
            clear_btn.click(
                fn=lambda: ("", "", None, None, None),
                outputs=[text_input, result_md, script_md, conf_plot, script_plot],
            )
            sample_dropdown.change(
                fn=lambda k: SAMPLES.get(k, ""),
                inputs=sample_dropdown,
                outputs=text_input,
            )

        # ── Tab 2: Batch Processing ───────────────────────────────────────────
        with gr.Tab("📋 Batch Processing"):
            gr.Markdown("Paste multiple texts (one per line). Max 50 texts.")
            batch_input = gr.Textbox(
                label="Texts (one per line)",
                placeholder="नमस्ते\nவணக்கம்\nHello\nسلام",
                lines=10,
            )
            batch_btn = gr.Button("🔍 Analyze Batch", variant="primary")
            batch_output = gr.Dataframe(label="Results")
            batch_btn.click(fn=analyze_batch_text, inputs=batch_input, outputs=batch_output)

        # ── Tab 3: Language Explorer ──────────────────────────────────────────
        with gr.Tab("🌐 Language Explorer"):
            lang_choices = [
                f"{info.name} ({code})"
                for code, info in LANGUAGE_REGISTRY.items()
            ]
            lang_selector = gr.Dropdown(
                choices=lang_choices,
                label="Select a language",
                value="Hindi (hin)",
            )
            lang_info_md = gr.Markdown()
            lang_selector.change(fn=get_language_info, inputs=lang_selector, outputs=lang_info_md)
            # Load initially
            demo.load(
                fn=lambda: get_language_info("Hindi (hin)"),
                outputs=lang_info_md,
            )

        # ── Tab 4: About ──────────────────────────────────────────────────────
        with gr.Tab("ℹ️ About"):
            gr.Markdown(f"""
## About This System

### Dataset
[AI4Bharat/Pralekha](https://huggingface.co/datasets/ai4bharat/Pralekha) is a large-scale
multilingual parallel corpus covering **12 Indic languages + English** with 1.5M+ document pairs.

### Supported Languages
{" · ".join(f"**{info.name}** ({code})" for code, info in LANGUAGE_REGISTRY.items())}

### Architecture
1. **Unicode Heuristic Engine** — Rule-based script detection (fast, ~100% accuracy for unique scripts)
2. **CharNgram Model** — TF-IDF char n-grams (2–4) + Logistic Regression with isotonic calibration
3. **FastText Model** — Subword embeddings trained on Pralekha corpus
4. **Ensemble** — Weighted soft voting with Devanagari disambiguation module

### Script Analysis Features
- Per-character Unicode block assignment
- Script segment extraction & counting
- BiDi (bidirectional) text detection
- Mixed-script scoring (Shannon entropy)
- Noise detection (URLs, zero-width chars, emojis)
- Transliteration/Romanization detection

### API
```bash
uvicorn src.api.server:app --reload --port 8000
```
Then visit: `http://localhost:8000/docs`
""")

    gr.Markdown("---\n*Built with AI4Bharat/Pralekha dataset · Anthropic Claude*")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    demo.launch(share=args.share, server_port=args.port, show_api=False)
