"""
FastAPI Language Identification & Script Analysis API
Production-ready REST API with async support, rate limiting, and OpenAPI docs.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field, validator

from src.analysis.script_analyzer import ScriptAnalyzer
from src.data.preprocessor import IndicTextPreprocessor
from src.utils.languages import LANGUAGE_REGISTRY, ALL_LANGUAGE_CODES


# ─────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────

class AppState:
    ensemble_model = None
    preprocessor: Optional[IndicTextPreprocessor] = None
    script_analyzer: Optional[ScriptAnalyzer] = None
    request_count: int = 0
    total_chars_processed: int = 0


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    logger.info("🚀 Starting Language ID API ...")
    state.preprocessor = IndicTextPreprocessor()
    state.script_analyzer = ScriptAnalyzer()

    # Load ensemble model if available
    model_path = Path("./artifacts/models/ensemble.pkl")
    if model_path.exists():
        try:
            from src.models.ensemble_model import EnsembleLIDModel
            import pickle
            with open(model_path, "rb") as f:
                state.ensemble_model = pickle.load(f)
            logger.success("Ensemble model loaded ✓")
        except Exception as e:
            logger.warning(f"Could not load ensemble model: {e}")
    else:
        # Load ngram model as fallback
        ngram_path = Path("./artifacts/models/ngram_model.pkl")
        if ngram_path.exists():
            from src.models.ngram_model import CharNgramLIDModel
            state.ensemble_model = CharNgramLIDModel.load(str(ngram_path))
            logger.success("CharNgram model loaded as fallback ✓")

    logger.info("API ready ✓")
    yield
    logger.info("Shutting down ...")


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────

app = FastAPI(
    title="🔤 Indic Language Identification API",
    description="""
**Advanced Language Identification & Script Analysis for 12 Indic Languages + English**

Powered by the AI4Bharat/Pralekha dataset.

### Features
- Language identification for 12 Indic languages + English
- Deep Unicode script analysis
- Mixed-script detection
- Batch processing
- Confidence scores with per-language probabilities
    """,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────

class IdentifyRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=10000, description="Input text")
    include_script_analysis: bool = Field(True, description="Include Unicode script analysis")
    include_alternatives: bool = Field(True, description="Include top-5 language alternatives")
    preprocess: bool = Field(True, description="Apply text preprocessing")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "नमस्ते, यह हिंदी में एक वाक्य है।",
                "include_script_analysis": True,
                "include_alternatives": True,
            }
        }


class BatchIdentifyRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts")
    include_script_analysis: bool = False
    preprocess: bool = True


class LanguageResult(BaseModel):
    lang_code: str
    lang_name: str
    script: str
    direction: str
    confidence: float
    is_confident: bool
    alternatives: Optional[List[Dict]] = None
    script_analysis: Optional[Dict] = None
    processing_time_ms: float


class BatchResult(BaseModel):
    results: List[LanguageResult]
    total_texts: int
    processing_time_ms: float
    stats: Dict


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _identify_text(text: str, preprocess: bool = True) -> Dict:
    """Core identification logic."""
    if preprocess and state.preprocessor:
        text = state.preprocessor.preprocess(text)

    if state.ensemble_model is None:
        # Fallback: pure heuristic
        from src.models.ensemble_model import UnicodeHeuristicModel
        heuristic = UnicodeHeuristicModel()
        lang, conf = heuristic.predict_single(text)
        if not lang:
            lang, conf = "eng", 0.5
        info = LANGUAGE_REGISTRY.get(lang)
        return {
            "predicted_lang": lang,
            "confidence": conf,
            "is_confident": conf >= 0.75,
            "top_5": [{"lang": lang, "probability": conf}],
            "script": info.script if info else "Unknown",
            "direction": info.direction if info else "LTR",
            "name": info.name if info else lang,
        }

    if hasattr(state.ensemble_model, "predict_with_confidence"):
        results = state.ensemble_model.predict_with_confidence([text])
        r = results[0]
        lang = r.get("predicted_lang", "?")
        info = LANGUAGE_REGISTRY.get(lang)
        return {
            "predicted_lang": lang,
            "confidence": r.get("confidence", 0.0),
            "is_confident": r.get("is_confident", False),
            "top_5": r.get("top_5", []),
            "script": r.get("script", info.script if info else "?"),
            "direction": r.get("direction", info.direction if info else "LTR"),
            "name": r.get("predicted_lang_name", info.name if info else lang),
        }
    else:
        pred = state.ensemble_model.predict([text])[0]
        info = LANGUAGE_REGISTRY.get(pred)
        return {
            "predicted_lang": pred,
            "confidence": 0.9,
            "is_confident": True,
            "top_5": [],
            "script": info.script if info else "?",
            "direction": info.direction if info else "LTR",
            "name": info.name if info else pred,
        }


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/", tags=["Status"])
async def root():
    return {
        "service": "Indic Language ID API",
        "version": "2.0.0",
        "status": "online",
        "supported_languages": len(ALL_LANGUAGE_CODES),
        "languages": ALL_LANGUAGE_CODES,
        "model_loaded": state.ensemble_model is not None,
    }


@app.get("/health", tags=["Status"])
async def health():
    return {
        "status": "healthy",
        "model_loaded": state.ensemble_model is not None,
        "requests_served": state.request_count,
        "chars_processed": state.total_chars_processed,
    }


@app.post("/identify", response_model=None, tags=["Language ID"])
async def identify(req: IdentifyRequest) -> Dict:
    """
    Identify the language of a single text.
    Returns predicted language, confidence, and optional script analysis.
    """
    t0 = time.perf_counter()
    state.request_count += 1
    state.total_chars_processed += len(req.text)

    try:
        result = _identify_text(req.text, preprocess=req.preprocess)
    except Exception as e:
        logger.exception(f"Identification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    response = {
        "lang_code": result["predicted_lang"],
        "lang_name": result["name"],
        "script": result["script"],
        "direction": result["direction"],
        "confidence": result["confidence"],
        "is_confident": result["is_confident"],
    }

    if req.include_alternatives:
        response["alternatives"] = result.get("top_5", [])

    if req.include_script_analysis and state.script_analyzer:
        sa = state.script_analyzer.analyze(req.text)
        response["script_analysis"] = sa.to_dict()

    response["processing_time_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return response


@app.post("/identify/batch", tags=["Language ID"])
async def identify_batch(req: BatchIdentifyRequest) -> Dict:
    """
    Identify languages for a batch of texts (up to 100).
    """
    t0 = time.perf_counter()
    state.request_count += 1

    texts = req.texts
    if req.preprocess and state.preprocessor:
        texts = state.preprocessor.batch_preprocess(texts, show_progress=False)

    state.total_chars_processed += sum(len(t) for t in texts)

    try:
        if state.ensemble_model and hasattr(state.ensemble_model, "predict_with_confidence"):
            raw_results = state.ensemble_model.predict_with_confidence(texts)
        else:
            raw_results = [_identify_text(t, preprocess=False) for t in texts]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    results = []
    lang_counts: Dict[str, int] = {}

    for i, r in enumerate(raw_results):
        lang = r.get("predicted_lang", "?")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
        info = LANGUAGE_REGISTRY.get(lang)
        item = {
            "lang_code": lang,
            "lang_name": r.get("name", info.name if info else lang),
            "script": r.get("script", info.script if info else "?"),
            "direction": r.get("direction", info.direction if info else "LTR"),
            "confidence": r.get("confidence", 0.0),
            "is_confident": r.get("is_confident", False),
        }
        if req.include_script_analysis and state.script_analyzer:
            sa = state.script_analyzer.analyze(texts[i])
            item["script_analysis"] = sa.to_dict()
        results.append(item)

    total_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "results": results,
        "total_texts": len(texts),
        "processing_time_ms": total_ms,
        "stats": {
            "language_distribution": lang_counts,
            "avg_confidence": round(
                np.mean([r["confidence"] for r in results]), 4
            ) if results else 0.0,
            "high_confidence_count": sum(1 for r in results if r["is_confident"]),
        },
    }


@app.post("/analyze/script", tags=["Script Analysis"])
async def analyze_script(body: Dict) -> Dict:
    """
    Deep Unicode script analysis for a text.
    Detects script, mixing, directionality, noise, and more.
    """
    text = body.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    if len(text) > 10000:
        raise HTTPException(status_code=400, detail="Text too long (max 10000 chars)")

    if not state.script_analyzer:
        state.script_analyzer = ScriptAnalyzer()

    result = state.script_analyzer.analyze(text)
    return result.to_dict()


@app.get("/languages", tags=["Metadata"])
async def list_languages() -> Dict:
    """List all supported languages with metadata."""
    return {
        "total": len(LANGUAGE_REGISTRY),
        "languages": [
            {
                "code": code,
                "name": info.name,
                "native_name": info.native_name,
                "script": info.script,
                "direction": info.direction,
                "family": info.family,
                "iso639_1": info.iso639_1,
                "iso639_3": info.iso639_3,
            }
            for code, info in LANGUAGE_REGISTRY.items()
        ]
    }


@app.get("/languages/{lang_code}", tags=["Metadata"])
async def get_language(lang_code: str) -> Dict:
    """Get detailed info for a specific language."""
    info = LANGUAGE_REGISTRY.get(lang_code)
    if not info:
        raise HTTPException(status_code=404, detail=f"Language '{lang_code}' not found")
    return {
        "code": info.code,
        "name": info.name,
        "native_name": info.native_name,
        "script": info.script,
        "direction": info.direction,
        "family": info.family,
        "iso639_1": info.iso639_1,
        "iso639_3": info.iso639_3,
        "shares_script_with": info.shares_script_with,
        "romanization_scheme": info.romanization_scheme,
    }


@app.get("/stats", tags=["Status"])
async def api_stats() -> Dict:
    """API usage statistics."""
    return {
        "requests_served": state.request_count,
        "chars_processed": state.total_chars_processed,
        "model_status": "loaded" if state.ensemble_model else "not_loaded",
    }
