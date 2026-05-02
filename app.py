from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("house_price_linear_regression.pkl")
SCALER_PATH = Path("scaler_linear_regression.pkl")

FEATURE_ORDER = [
    "Square_Footage",
    "Num_Bedrooms",
    "Num_Bathrooms",
    "Year_Built",
    "Lot_Size",
    "Garage_Size",
    "Neighborhood_Quality",
]


class HouseFeatures(BaseModel):
    Square_Footage: float = Field(..., gt=0)
    Num_Bedrooms: float = Field(..., ge=0)
    Num_Bathrooms: float = Field(..., ge=0)
    Year_Built: float = Field(..., ge=1700, le=2100)
    Lot_Size: float = Field(..., gt=0)
    Garage_Size: float = Field(..., ge=0)
    Neighborhood_Quality: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    predicted_house_price: float
    currency: str
    model: str


app = FastAPI(
    title="House Price Linear Regression API",
    version="1.0.0",
    description="API untuk prediksi harga rumah menggunakan model Linear Regression.",
)


model = None
scaler = None


def _model_files_status() -> Dict[str, bool]:
    return {
        str(MODEL_PATH): MODEL_PATH.exists(),
        str(SCALER_PATH): SCALER_PATH.exists(),
    }


@app.on_event("startup")
def load_artifacts() -> None:
    global model, scaler
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "House Price Linear Regression API aktif."}


@app.get("/health")
def health() -> Dict[str, object]:
    files = _model_files_status()
    ready = model is not None and scaler is not None
    return {
        "status": "ready" if ready else "not_ready",
        "model_loaded": ready,
        "artifact_files": files,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: HouseFeatures) -> PredictionResponse:
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Model/scaler belum tersedia. Jalankan notebook untuk menghasilkan artifact .pkl terlebih dahulu.",
                "artifact_files": _model_files_status(),
            },
        )

    input_df = pd.DataFrame([[getattr(payload, c) for c in FEATURE_ORDER]], columns=FEATURE_ORDER)
    scaled = scaler.transform(input_df)
    pred = model.predict(scaled)[0]

    if not np.isfinite(pred):
        raise HTTPException(status_code=500, detail="Prediksi tidak valid.")

    return PredictionResponse(
        predicted_house_price=float(pred),
        currency="USD",
        model="LinearRegression",
    )
