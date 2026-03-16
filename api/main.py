"""
api/main.py
────────────────────────────────────────────────────────────────
FastAPI application that exposes the trained model as a REST API.

Endpoints
  GET  /              – health check / welcome
  GET  /health        – liveness probe
  GET  /model-info    – model metadata & evaluation scores
  POST /predict       – predict house price from feature JSON

Run (from project root):
    uvicorn api.main:app --reload --port 8000
────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import pickle
import logging
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Allow imports from project root when running this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import FEATURE_COLS

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title="House Price Prediction API",
    description=(
        "Predicts California median house prices using a trained "
        "Machine Learning model (scikit-learn Pipeline)."
    ),
    version="1.0.0",
)

# Allow requests from the local frontend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Model loading ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "best_model.pkl")
EVAL_PATH  = os.path.join(os.path.dirname(__file__), "..", "model", "evaluation_results.json")

_model = None   # loaded lazily on first request


def get_model():
    """Load (and cache) the trained model pipeline from disk."""
    global _model
    if _model is None:
        abs_path = os.path.abspath(MODEL_PATH)
        if not os.path.exists(abs_path):
            raise RuntimeError(
                f"Model file not found at {abs_path}. "
                "Please run: python model/train.py"
            )
        with open(abs_path, "rb") as f:
            _model = pickle.load(f)
        logger.info("Model loaded from %s", abs_path)
    return _model


# ── Request / Response schemas ────────────────────────────────

class HouseFeatures(BaseModel):
    """
    Input features for a single prediction request.
    All fields are validated to be non-negative real numbers.
    """
    MedInc:     float = Field(..., ge=0,  description="Median income ($10k units)")
    HouseAge:   float = Field(..., ge=0,  description="Median house age (years)")
    AveRooms:   float = Field(..., ge=0,  description="Avg rooms per household")
    AveBedrms:  float = Field(..., ge=0,  description="Avg bedrooms per household")
    Population: float = Field(..., ge=0,  description="Block group population")
    AveOccup:   float = Field(..., ge=0,  description="Avg household size")
    Latitude:   float = Field(..., ge=-90,  le=90,  description="Latitude")
    Longitude:  float = Field(..., ge=-180, le=180, description="Longitude")

    # Example payload shown in the interactive docs
    model_config = {
        "json_schema_extra": {
            "example": {
                "MedInc":     3.87,
                "HouseAge":   29.0,
                "AveRooms":   5.43,
                "AveBedrms":  1.09,
                "Population": 1015.0,
                "AveOccup":   2.72,
                "Latitude":   34.05,
                "Longitude":  -118.24,
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_price_100k: float = Field(..., description="Predicted price in $100k units")
    predicted_price_usd:  float = Field(..., description="Predicted price in USD")
    model_used:           str   = Field(..., description="Name of the model pipeline")
    input_features:       dict  = Field(..., description="Echo of the input features")


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "message": "🏠 House Price Prediction API is running!",
        "docs":    "http://localhost:8000/docs",
        "predict": "POST /predict",
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Liveness probe for orchestration systems (Docker, k8s, etc.)."""
    return {"status": "ok"}


@app.get("/model-info", tags=["Model"])
def model_info():
    """Return model metadata and (if available) evaluation scores."""
    abs_eval = os.path.abspath(EVAL_PATH)
    evaluation = {}
    if os.path.exists(abs_eval):
        with open(abs_eval) as f:
            evaluation = json.load(f)

    return {
        "features":   FEATURE_COLS,
        "target":     "MedHouseVal (median house value in $100k)",
        "evaluation": evaluation,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: HouseFeatures):
    """
    Predict the median house value for the given input features.

    Returns the price in both $100k units (model native) and
    plain USD for convenience.
    """
    try:
        model = get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Convert Pydantic model → ordered numpy array
    feature_values = np.array([[
        features.MedInc,
        features.HouseAge,
        features.AveRooms,
        features.AveBedrms,
        features.Population,
        features.AveOccup,
        features.Latitude,
        features.Longitude,
    ]])

    try:
        prediction_100k = float(model.predict(feature_values)[0])
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    # Clamp to realistic range (model may extrapolate below zero)
    prediction_100k = max(prediction_100k, 0.0)

    return PredictionResponse(
        predicted_price_100k=round(prediction_100k, 4),
        predicted_price_usd=round(prediction_100k * 100_000, 2),
        model_used=type(model.named_steps["regressor"]).__name__,
        input_features=features.model_dump(),
    )
