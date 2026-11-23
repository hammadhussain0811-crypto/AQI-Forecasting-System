
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from tensorflow.keras.models import load_model

from src.common.feature_engineering import FEATURE_COLUMNS


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"


# Supported cities
SUPPORTED_CITIES = [
    "Delhi",
    "Bengaluru",
    "Lucknow",
    "Chennai",
    "Hyderabad"
]


def prepare_aqi_series(daily_data):
    """
    Prepare continuous daily AQI history for prediction.
    """

    data = daily_data.copy()

    data["Date"] = pd.to_datetime(data["Date"])

    data = data.sort_values("Date")

    if data["Date"].duplicated().any():
        raise ValueError("Duplicate dates found.")

    data = data.set_index("Date")

    full_dates = pd.date_range(
        start=data.index.min(),
        end=data.index.max(),
        freq="D"
    )

    aqi_series = data["AQI"].reindex(full_dates)

    aqi_series.index.name = "Date"

    return aqi_series


def load_city_models(city):
    """
    Load all four saved models and the LSTM scaler
    for the selected city.
    """

    if city not in SUPPORTED_CITIES:
        raise ValueError(
            f"Unsupported city: {city}"
        )

    city_key = city.lower()

    # Load SARIMA
    sarima_path = (
        MODEL_DIR / "sarima" /
        f"{city_key}_sarima.pkl"
    )

    sarima_model = SARIMAXResults.load(
        str(sarima_path)
    )

    # Load Random Forest
    rf_path = (
        MODEL_DIR / "random_forest" /
        f"{city_key}_rf.joblib"
    )

    rf_model = joblib.load(rf_path)

    # Load XGBoost
    xgb_path = (
        MODEL_DIR / "xgboost" /
        f"{city_key}_xgb.joblib"
    )

    xgb_model = joblib.load(xgb_path)

    # Load LSTM
    lstm_path = (
        MODEL_DIR / "lstm" /
        f"{city_key}_lstm.keras"
    )

    lstm_model = load_model(lstm_path)

    # Load LSTM scaler
    scaler_path = (
        MODEL_DIR / "lstm" /
        f"{city_key}_scaler.pkl"
    )

    lstm_scaler = joblib.load(scaler_path)

    return {
        "SARIMA": sarima_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
        "LSTM": lstm_model,
        "LSTM_Scaler": lstm_scaler
    }


def create_latest_features(aqi_series):
    """
    Create the exact 11 features required by
    Random Forest and XGBoost for next-day prediction.
    """

    if len(aqi_series) < 8:
        raise ValueError(
            "At least 8 calendar days of AQI history are required."
        )

    recent_values = aqi_series.iloc[-8:]

    if not np.isfinite(recent_values.to_numpy(dtype=float)).all():
        raise ValueError(
            "Missing AQI values found in the latest 8 days. "
            "Valid recent daily observations are required."
        )

    latest_date = aqi_series.index[-1]

    target_date = latest_date + pd.Timedelta(days=1)

    values = recent_values.to_numpy(dtype=float)

    features = {
        "aqi_lag_0": values[-1],
        "aqi_lag_1": values[-2],
        "aqi_lag_2": values[-3],
        "aqi_lag_3": values[-4],
        "aqi_lag_7": values[-8],

        "rolling_mean_3": np.mean(values[-3:]),
        "rolling_mean_7": np.mean(values[-7:]),

        "target_dayofweek": target_date.dayofweek,
        "target_month": target_date.month,
        "target_dayofyear": target_date.dayofyear,
        "target_is_weekend": int(target_date.dayofweek >= 5)
    }

    feature_df = pd.DataFrame(
        [features],
        columns=FEATURE_COLUMNS
    )

    return feature_df, target_date


def predict_sarima(sarima_model, aqi_series):
    """
    Update the saved SARIMA model with observations
    after its training endpoint and forecast one day ahead.
    """

    current_model = sarima_model

    # Find the last date already included in the saved model
    model_dates = current_model.model.data.row_labels

    last_model_date = pd.Timestamp(model_dates[-1])

    # Add only observations not already included in the model
    new_observations = aqi_series.loc[
        aqi_series.index > last_model_date
    ]

    if not new_observations.empty:

        current_model = current_model.append(
            new_observations,
            refit=False
        )

    forecast = current_model.get_forecast(
        steps=1
    ).predicted_mean

    return float(forecast.iloc[0])


def predict_random_forest(rf_model, feature_df):
    """
    Generate next-day AQI prediction using Random Forest.
    """

    prediction = rf_model.predict(feature_df)

    return float(prediction[0])


def predict_xgboost(xgb_model, feature_df):
    """
    Generate next-day AQI prediction using XGBoost.
    """

    prediction = xgb_model.predict(feature_df)

    return float(prediction[0])


def predict_lstm(lstm_model, scaler, aqi_series):
    """
    Generate next-day AQI prediction using LSTM.
    """

    lookback = 7

    recent_values = aqi_series.iloc[-lookback:].to_numpy(
        dtype=float
    )

    if not np.isfinite(recent_values).all():
        raise ValueError(
            "Missing AQI values in the latest 7 days."
        )

    scaled_values = scaler.transform(
        recent_values.reshape(-1, 1)
    )

    X = scaled_values.reshape(
        1, lookback, 1
    ).astype(np.float32)

    scaled_prediction = lstm_model.predict(
        X,
        verbose=0
    )

    prediction = scaler.inverse_transform(
        scaled_prediction
    )

    return float(prediction[0][0])


def predict_all_models(city, daily_data):
    """
    Generate next-day AQI predictions using all four models.

    Parameters:
        city: Selected city.
        daily_data: Historical daily AQI DataFrame.

    Returns:
        DataFrame containing four model predictions.
    """

    if city not in SUPPORTED_CITIES:
        raise ValueError(
            f"Unsupported city: {city}"
        )

    aqi_series = prepare_aqi_series(daily_data)

    if aqi_series.empty:
        raise ValueError("No AQI history available.")

    if not np.isfinite(aqi_series.iloc[-1]):
        raise ValueError(
            "The latest day's AQI is missing."
        )

    # Load saved models
    models = load_city_models(city)

    # Create features for RF and XGBoost
    feature_df, target_date = create_latest_features(
        aqi_series
    )

    # Generate predictions
    predictions = {
        "SARIMA": predict_sarima(
            models["SARIMA"],
            aqi_series
        ),

        "Random Forest": predict_random_forest(
            models["Random Forest"],
            feature_df
        ),

        "XGBoost": predict_xgboost(
            models["XGBoost"],
            feature_df
        ),

        "LSTM": predict_lstm(
            models["LSTM"],
            models["LSTM_Scaler"],
            aqi_series
        )
    }

    # Prepare final prediction DataFrame
    results = pd.DataFrame([
        {
            "City": city,
            "Forecast_Date": target_date,
            "Model": model_name,
            "Predicted_AQI": prediction
        }
        for model_name, prediction in predictions.items()
    ])

    return results
