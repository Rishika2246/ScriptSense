# 🔤 Language Identification & Script Analysis System
### Powered by AI4Bharat/Pralekha Dataset

An advanced NLP pipeline for **language identification**, **script recognition**, and **linguistic analysis** across 12 Indic languages + English using the [Pralekha dataset](https://huggingface.co/datasets/ai4bharat/Pralekha).

---

## 📦 Languages Supported

| Language | Script | Code |
|----------|--------|------|
| Bengali | Bengali | ben |
| English | Latin | eng |
| Gujarati | Gujarati | guj |
| Hindi | Devanagari | hin |
| Kannada | Kannada | kan |
| Malayalam | Malayalam | mal |
| Marathi | Devanagari | mar |
| Odia | Odia | ori |
| Punjabi | Gurmukhi | pan |
| Tamil | Tamil | tam |
| Telugu | Telugu | tel |
| Urdu | Perso-Arabic | urd |

---

## 🏗️ Architecture

```
lang_id_project/
├── src/
│   ├── data/           # Dataset loading, preprocessing, streaming
│   ├── models/         # ML models: char-ngram, transformer, ensemble
│   ├── analysis/       # Script analysis, Unicode inspection, statistics
│   ├── utils/          # Helpers, logging, metrics
│   └── api/            # FastAPI REST service
├── notebooks/          # EDA and experiment notebooks
├── configs/            # YAML configs
├── tests/              # Unit + integration tests
└── scripts/            # CLI tools
```

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# Train all models
python scripts/train_pipeline.py --config configs/default.yaml

# Run API server
uvicorn src.api.server:app --reload --port 8000

# Analyze text
python scripts/analyze.py --text "नमस्ते दुनिया"

# Run full evaluation
python scripts/evaluate.py --split test
```

## 📊 Features

- **Multi-model ensemble**: Character n-gram + Transformer + Unicode heuristics
- **Script-level analysis**: Unicode block detection, script mixing, directionality
- **Streaming dataset pipeline**: Handles 1.5M+ rows with minimal RAM
- **FastAPI REST API**: Production-ready with async support
- **Detailed metrics**: Per-language F1, confusion matrix, confidence calibration
- **Interactive dashboard**: Gradio-powered web UI
