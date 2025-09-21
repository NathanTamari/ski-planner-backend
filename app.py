# app.py
from datetime import datetime
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import joblib
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Ski Planner Backend")

# ----- CORS -----
ALLOWED_ORIGINS = [
    "https://nathantamari.github.io",  # prod (GitHub Pages)
    "http://localhost:3000",           # local dev
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Load models once (at startup) -----
model = joblib.load("rf_model.joblib")
kmeans = joblib.load("kmeans_model.joblib")


# ----- Schemas -----
class TripRequest(BaseModel):
    lat: float
    lon: float
    guests: int = Field(ge=1)
    check_in: str  # "YYYY-MM-DD" or ISO 8601 date string
    check_out: str

class PredictPriceResponse(BaseModel):
    predicted_price: Optional[float]  # None if failed

class BatchPredictRequest(BaseModel):
    items: List[TripRequest]

class BatchPredictItem(BaseModel):
    price: Optional[float]  # None if failed

class BatchPredictResponse(BaseModel):
    results: List[BatchPredictItem]


# ----- Core single prediction (shared by both endpoints) -----
def predict_one(trip: TripRequest, *, now: Optional[datetime] = None) -> Optional[float]:
    try:
        # Parse dates
        check_in = datetime.fromisoformat(trip.check_in)
        check_out = datetime.fromisoformat(trip.check_out)
        if check_out <= check_in:
            return None

        # Features
        stay_length = (check_out - check_in).days
        month = check_in.month
        day_of_week = check_in.weekday()
        season = (month % 12) // 3
        now = now or datetime.now()
        days_until_checkin = (check_in - now).days
        is_holiday = 1 if (month == 12 and 20 <= check_in.day <= 31) else 0
        location_cluster = int(kmeans.predict([[trip.lat, trip.lon]])[0])

        features = [[
            trip.lat,
            trip.lon,
            trip.guests,
            stay_length,
            month,
            day_of_week,
            season,
            days_until_checkin,
            is_holiday,
            location_cluster,
        ]]

        # Model predict; undo log1p if that’s how it was trained
        y_hat = model.predict(features)[0]
        predicted_price = float(np.expm1(y_hat))
        return round(predicted_price, 2)
    except Exception:
        return None


# ----- Health & root -----
@app.get("/")
def root():
    return {"message": "FastAPI backend is running with ML pricing!"}

@app.get("/health")
def health():
    return {"status": "ok"}


# ----- Single-item endpoint (kept for compatibility) -----
@app.post("/predict_price", response_model=PredictPriceResponse)
def predict_price(trip: TripRequest):
    price = predict_one(trip)
    return PredictPriceResponse(predicted_price=price)


# ----- NEW: Batch endpoint -----
@app.post("/predict_prices", response_model=BatchPredictResponse)
def predict_prices(batch: BatchPredictRequest):
    # Use a single "now" for all items so days_until_checkin is consistent
    now = datetime.now()

    results: List[Optional[float]] = [None] * len(batch.items)

    # Parallelize if there are many items
    max_workers = min(16, max(1, len(batch.items)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(predict_one, item, now=now): idx
            for idx, item in enumerate(batch.items)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception:
                results[idx] = None

    return BatchPredictResponse(results=[BatchPredictItem(price=p) for p in results])


# Local dev 
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
