# app.py
from __future__ import annotations
import os
import warnings
from functools import lru_cache
from datetime import datetime, date
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---- Performance knobs for small Render instances (avoid thread thrash) ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Silence the noisy (and harmless) sklearn feature-name warning
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but",
    category=UserWarning,
)

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

# ===== Schemas =====
class TripRequest(BaseModel):
    lat: float
    lon: float
    guests: int = Field(ge=1)
    check_in: str  # "YYYY-MM-DD" (prefer) or ISO 8601
    check_out: str

class PredictPriceResponse(BaseModel):
    predicted_price: Optional[float]  # None if failed


# ===== Model load (once) & warmup =====
model = joblib.load("rf_model.joblib")
kmeans = joblib.load("kmeans_model.joblib")

# A tiny fast date parser for "YYYY-MM-DD" (fallback to fromisoformat)
def _parse_date(s: str) -> date:
    # fast path for 'YYYY-MM-DD'
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        y = int(s[0:4]); m = int(s[5:7]); d = int(s[8:10])
        return date(y, m, d)
    return datetime.fromisoformat(s).date()

@lru_cache(maxsize=1024)
def _loc_cluster(lat_rounded: int, lon_rounded: int) -> int:
    # use rounded coords as cache key to dedupe repeated resorts
    lat = lat_rounded / 1_000_000
    lon = lon_rounded / 1_000_000
    return int(kmeans.predict([[lat, lon]])[0])

def _is_holiday(month: int, day: int) -> int:
    # keep logic tiny & branchless-ish
    return 1 if (month == 12 and 20 <= day <= 31) else 0

def predict_one(trip: TripRequest, *, now_date: Optional[date] = None) -> Optional[float]:
    try:
        check_in_d = _parse_date(trip.check_in)
        check_out_d = _parse_date(trip.check_out)
        if check_out_d <= check_in_d:
            return None

        stay_length = (check_out_d - check_in_d).days
        month = check_in_d.month
        day_of_week = check_in_d.weekday()
        season = (month % 12) // 3
        nd = now_date or datetime.now().date()
        days_until_checkin = (check_in_d - nd).days
        is_holiday = _is_holiday(month, check_in_d.day)

        # cache KMeans per rounded lat/lon to avoid repeated predictions
        lat_i = int(round(trip.lat * 1_000_000))
        lon_i = int(round(trip.lon * 1_000_000))
        location_cluster = _loc_cluster(lat_i, lon_i)

        # Keep as numpy array for very small overhead
        X = np.array([[
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
        ]], dtype=float)

        y_hat = model.predict(X)[0]
        predicted_price = float(np.expm1(y_hat))
        return round(predicted_price, 2)
    except Exception:
        return None


# ===== Routes =====
@app.get("/")
def root():
    return {"message": "FastAPI backend is running with ML pricing!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict_price", response_model=PredictPriceResponse)
def predict_price(trip: TripRequest):
    # use a consistent "now" so repeated calls in a burst are stable
    price = predict_one(trip, now_date=datetime.now().date())
    return PredictPriceResponse(predicted_price=price)


# Warm the models & caches at startup to reduce first-hit latency
@app.on_event("startup")
def _warmup():
    try:
        _ = kmeans.predict([[0.0, 0.0]])
    except Exception:
        pass
    try:
        dummy = TripRequest(
            lat=0.0, lon=0.0, guests=2,
            check_in="2026-01-10", check_out="2026-01-11"
        )
        _ = predict_one(dummy, now_date=date(2025, 9, 22))
    except Exception:
        pass


# Local dev
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    # For local testing, 1 worker is usually fastest on small laptops
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True, log_level="warning")
