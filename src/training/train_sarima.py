
import warnings

import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


def train_and_predict_sarima(
    city,
    daily_data,
    city_split,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7)
):
    """
    Train SARIMA using historical daily AQI and generate
    rolling one-day-ahead forecasts for the test period.

    Missing daily AQI observations are preserved.
    Actual AQI from a test day is only added after that
    day's forecast has been generated.
    """

    data = daily_data.copy()

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date")

    # Create a continuous daily AQI series
    data = data.set_index("Date")

    full_dates = pd.date_range(
        start=data.index.min(),
        end=data.index.max(),
        freq="D"
    )

    aqi_series = data["AQI"].reindex(full_dates)
    aqi_series.index.name = "Date"

    # Get the testing period from our existing split
    test_dates = pd.to_datetime(city_split["test"]["Date"])

    test_start = test_dates.min()
    test_end = test_dates.max()

    # Training observations must be before the test period
    train_series = aqi_series.loc[
        aqi_series.index < test_start
    ]

    # Preserve every calendar day in the test period
    test_series = aqi_series.loc[
        (aqi_series.index >= test_start)
        & (aqi_series.index <= test_end)
    ]

    if train_series.notna().sum() < 100:
        raise ValueError(
            f"Not enough historical AQI observations for {city}."
        )

    print(f"\nTraining SARIMA for {city}...")
    print("Training observations:", train_series.notna().sum())
    print("Testing calendar days:", len(test_series))

    # Build and fit the SARIMA model
    model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted_model = model.fit(disp=False)

    print("Model fitting completed.")

    # Rolling one-day-ahead forecasting
    current_model = fitted_model
    predictions = []

    for date, actual in test_series.items():

        # Predict the next day before observing its actual AQI
        forecast = current_model.get_forecast(
            steps=1
        ).predicted_mean.iloc[0]

        predictions.append({
            "Date": date,
            "actual_AQI": actual,
            "predicted_AQI": forecast
        })

        # Update the model with the actual observation
        # only after making that day's prediction.
      
        new_observation = pd.Series(
            [actual],
            index=pd.DatetimeIndex([date]),
            name=train_series.name
        )

        current_model = current_model.append(
            endog=new_observation,
            refit=False
        )


    predictions_df = pd.DataFrame(predictions)

    # Keep exactly the target dates used in the ML test set
    predictions_df = predictions_df[
        predictions_df["Date"].isin(test_dates)
    ].reset_index(drop=True)

    return fitted_model, predictions_df
