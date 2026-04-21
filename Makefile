# ─────────────────────────────────────────────
# Makefile — Language ID & Script Analysis
# ─────────────────────────────────────────────

.PHONY: install train train-ngram train-fasttext dashboard api test lint clean

PYTHON = python
CONFIG = configs/default.yaml

# ── Setup ──────────────────────────────────────

install:
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

install-dev: install
	pip install jupyter black isort ruff mypy
	@echo "✓ Dev dependencies installed"

# ── Data ──────────────────────────────────────

prepare-data:
	$(PYTHON) scripts/train_pipeline.py \
		--config $(CONFIG) \
		--model ngram \
		--max_per_lang 50000 \
		--no-eval
	@echo "✓ Data prepared"

# ── Training ──────────────────────────────────

train:
	$(PYTHON) scripts/train_pipeline.py \
		--config $(CONFIG) \
		--model all \
		--max_per_lang 50000
	@echo "✓ All models trained"

train-ngram:
	$(PYTHON) scripts/train_pipeline.py \
		--model ngram \
		--max_per_lang 30000
	@echo "✓ CharNgram model trained"

train-fasttext:
	$(PYTHON) scripts/train_pipeline.py \
		--model fasttext \
		--max_per_lang 50000
	@echo "✓ FastText model trained"

train-quick:
	$(PYTHON) scripts/train_pipeline.py \
		--model ngram \
		--max_per_lang 5000
	@echo "✓ Quick training done"

# ── Evaluation ─────────────────────────────────

eval:
	$(PYTHON) scripts/train_pipeline.py \
		--skip_data_prep \
		--model all \
		--eval
	@echo "✓ Evaluation complete. Check artifacts/eval/"

# ── Analysis ──────────────────────────────────

analyze:
	@echo "Enter text: "
	$(PYTHON) scripts/analyze.py

demo:
	$(PYTHON) scripts/analyze.py --demo

analyze-text:
	$(PYTHON) scripts/analyze.py --text "$(TEXT)"

# ── API Server ─────────────────────────────────

api:
	uvicorn src.api.server:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload \
		--log-level info
	@echo "API running at http://localhost:8000"

api-prod:
	uvicorn src.api.server:app \
		--host 0.0.0.0 \
		--port 8000 \
		--workers 4

# ── Dashboard ─────────────────────────────────

dashboard:
	$(PYTHON) scripts/dashboard.py

dashboard-public:
	$(PYTHON) scripts/dashboard.py --share

# ── Notebooks ─────────────────────────────────

notebook:
	cd notebooks && jupyter notebook

# ── Testing ───────────────────────────────────

test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/ -v --tb=short -x -q

test-coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term
	@echo "Coverage report: htmlcov/index.html"

# ── Code Quality ──────────────────────────────

lint:
	ruff check src/ scripts/ tests/

format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

typecheck:
	mypy src/ --ignore-missing-imports

# ── Cleanup ────────────────────────────────────

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null; true
	@echo "✓ Cleaned"

clean-cache:
	rm -rf cache/datasets/*.parquet
	@echo "✓ Dataset cache cleared"

clean-models:
	rm -rf artifacts/models/*
	@echo "✓ Models cleared"

clean-all: clean clean-cache clean-models

# ── Help ──────────────────────────────────────

help:
	@echo ""
	@echo "🔤 Indic Language ID System — Commands"
	@echo "────────────────────────────────────────"
	@echo "  make install        Install dependencies"
	@echo "  make train          Train all models (ngram + fasttext + ensemble)"
	@echo "  make train-quick    Quick training (5k samples/lang, for testing)"
	@echo "  make eval           Evaluate all trained models"
	@echo "  make demo           Run analysis on all sample texts"
	@echo "  make api            Start REST API server (port 8000)"
	@echo "  make dashboard      Launch Gradio interactive UI"
	@echo "  make test           Run all unit tests"
	@echo "  make notebook       Open Jupyter notebook"
	@echo "  make clean          Clean Python cache"
	@echo ""
	@echo "  make analyze-text TEXT='your text here'"
	@echo ""
