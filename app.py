from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  uvicorn app:app --reload to run the server
model = joblib.load("rf_model.joblib")
kmeans = joblib.load("kmeans_model.joblib")

class TripRequest(BaseModel):
    lat: float
    lon: float
    guests: int
    check_in: str
    check_out: str

@app.get("/")
def root():
    return {"message": "FastAPI backend is running with ML pricing!"}

@app.post("/predict_price")
def predict_price(trip: TripRequest):
    try:
        # Convert dates
        check_in = datetime.fromisoformat(trip.check_in)
        check_out = datetime.fromisoformat(trip.check_out)

        # Features
        stay_length = (check_out - check_in).days
        month = check_in.month
        day_of_week = check_in.weekday()
        season = month % 12 // 3
        days_until_checkin = (check_in - datetime.today()).days
        is_holiday = 1 if (month == 12 and 20 <= check_in.day <= 31) else 0
        location_cluster = int(
            kmeans.predict([[trip.lat, trip.lon]])[0]
        )

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
            location_cluster
        ]]

        # Predict (remember to undo log transform)
        predicted_price = np.expm1(model.predict(features)[0])

        return {"predicted_price": round(float(predicted_price), 2)}

    except Exception as e:
        return {"error": str(e)}
