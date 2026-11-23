
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from src.common.preprocessing import (
    prepare_aqi_data,
    SELECTED_CITIES
)

from src.prediction.predict import predict_all_models


# PROJECT CONFIGURATION


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT / "dataset" / "raw" / "city_day.csv"
)


# BEST MODEL FOR EACH CITY
# Selected using historical test-set MAE

BEST_MODELS = {
    "Delhi": {
        "model": "Random Forest",
        "mae": 32.19
    },
    "Bengaluru": {
        "model": "SARIMA",
        "mae": 8.15
    },
    "Lucknow": {
        "model": "SARIMA",
        "mae": 27.00
    },
    "Chennai": {
        "model": "SARIMA",
        "mae": 14.84
    },
    "Hyderabad": {
        "model": "LSTM",
        "mae": 11.16
    }
}

# APPLICATION LIFESPAN


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Loading AQI dataset...")

    try:
        app.state.city_data = prepare_aqi_data(
            DATASET_PATH
        )

        print("AQI dataset loaded successfully.")

    except Exception as error:
        print(f"Dataset loading failed: {error}")
        raise

    yield

    print("AQI API shutting down.")


# FASTAPI APPLICATION

app = FastAPI(
    title="AQI Forecasting API",
    description=(
        "Next-day AQI prediction using "
        "SARIMA, Random Forest, XGBoost and LSTM."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# REQUEST SCHEMA

class PredictionRequest(BaseModel):
    city: str


# ROOT ENDPOINT


@app.get("/")
def home():

    return {
        "message": "AQI Forecasting API is running.",
        "documentation": "/docs"
    }



# HEALTH CHECK ENDPOINT


@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }


# GET SUPPORTED CITIES


@app.get("/api/cities")
def get_cities():

    return {
        "cities": SELECTED_CITIES
    }

# NEXT-DAY AQI PREDICTION ENDPOINT


@app.post("/api/predict")
def predict_aqi(
    request: PredictionRequest,
    api_request: Request
):

    # Normalize city name
    requested_city = request.city.strip().lower()

    city_lookup = {
        city.lower(): city
        for city in SELECTED_CITIES
    }

    if requested_city not in city_lookup:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported city. "
                "Please select a supported city."
            )
        )

    city = city_lookup[requested_city]

    # Retrieve preprocessed city data
    city_data = api_request.app.state.city_data

    if city not in city_data:

        raise HTTPException(
            status_code=404,
            detail=f"No historical AQI data found for {city}."
        )

    try:

        # Generate predictions using all four saved models
        predictions_df = predict_all_models(
            city=city,
            daily_data=city_data[city]
        )

        # Get forecast date
        forecast_date = pd.Timestamp(
            predictions_df["Forecast_Date"].iloc[0]
        ).strftime("%Y-%m-%d")

        # Format predictions
        predictions = [
            {
                "model": row["Model"],
                "predicted_aqi": float(row["Predicted_AQI"])
            }
            for _, row in predictions_df.iterrows()
        ]

        # Get best-performing model for selected city
        best_model = BEST_MODELS[city]

        # Final API response
        return {
            "city": city,
            "forecast_date": forecast_date,
            "forecast_type": "Next-day AQI prediction",

            "best_model": best_model["model"],
            "best_model_mae": best_model["mae"],

            "predictions": predictions
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"AQI prediction failed: {str(error)}"
        )

