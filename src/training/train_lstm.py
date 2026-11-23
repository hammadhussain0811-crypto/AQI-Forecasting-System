
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


LOOKBACK = 7


def create_sequences(aqi_values, target_indices, lookback=LOOKBACK):
    """
    Create sequences of historical AQI values.

    Each sequence contains the previous 7 days.
    The target is the AQI of the following day.
    """

    X = []
    y = []
    valid_indices = []

    for target_idx in target_indices:

        if target_idx < lookback:
            continue

        sequence = aqi_values[
            target_idx - lookback:target_idx
        ]

        target = aqi_values[target_idx]

        # Skip samples with missing historical AQI
        if not np.isfinite(sequence).all():
            continue

        # Skip missing target AQI during training
        if not np.isfinite(target):
            continue

        X.append(sequence)
        y.append(target)
        valid_indices.append(target_idx)

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(valid_indices)
    )


def train_and_predict_lstm(
    city,
    daily_data,
    city_split,
    lookback=LOOKBACK,
    epochs=50,
    batch_size=32,
    random_state=42
):
    """
    Train an LSTM model for next-day AQI forecasting.

    Uses chronological training and rolling one-day-ahead
    predictions for the test period.
    """

    tf.keras.utils.set_random_seed(random_state)

    # Prepare daily AQI data
    data = daily_data.copy()

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    data = data.set_index("Date")

    full_dates = pd.date_range(
        start=data.index.min(),
        end=data.index.max(),
        freq="D"
    )

    aqi_series = data["AQI"].reindex(full_dates)

    aqi_values = aqi_series.to_numpy(dtype=float)

    # Determine training and testing boundaries
    test_dates = pd.DatetimeIndex(
        pd.to_datetime(city_split["test"]["Date"])
    )

    test_start = test_dates.min()

    # Only observations before the test period
    training_mask = aqi_series.index < test_start

    training_values = aqi_values[
        training_mask
    ]

    # Fit scaler only on training-period AQI
    scaler = MinMaxScaler()

    scaler.fit(
        training_values[
            np.isfinite(training_values)
        ].reshape(-1, 1)
    )

    # Transform available AQI values
    scaled_values = np.full(
        len(aqi_values),
        np.nan,
        dtype=float
    )

    valid_mask = np.isfinite(aqi_values)

    scaled_values[valid_mask] = scaler.transform(
        aqi_values[valid_mask].reshape(-1, 1)
    ).flatten()

    # Find training target positions
    train_indices = np.where(
        training_mask
    )[0]

    X_train, y_train, _ = create_sequences(
        scaled_values,
        train_indices,
        lookback
    )

    if len(X_train) < 100:
        raise ValueError(
            f"Insufficient training sequences for {city}."
        )

    # Find testing target positions
    test_indices = np.where(
        aqi_series.index.isin(test_dates)
    )[0]

    # Build test sequences using historical observations
    # available before each individual prediction
    X_test = []
    test_actuals = []
    prediction_dates = []

    for idx in test_indices:

        if idx < lookback:
            continue

        sequence = scaled_values[
            idx - lookback:idx
        ]

        if not np.isfinite(sequence).all():
            continue

        X_test.append(sequence)

        test_actuals.append(aqi_values[idx])

        prediction_dates.append(
            aqi_series.index[idx]
        )

    X_test = np.array(
        X_test,
        dtype=np.float32
    ).reshape(-1, lookback, 1)

    X_train = X_train.reshape(
        -1, lookback, 1
    )

    print(f"\nTraining LSTM for {city}...")
    print("Training sequences:", len(X_train))
    print("Testing sequences:", len(X_test))

    # Build LSTM neural network
    model = Sequential([
        Input(shape=(lookback, 1)),

        LSTM(
            64,
            return_sequences=True
        ),

        Dropout(0.2),

        LSTM(32),

        Dropout(0.2),

        Dense(16, activation="relu"),

        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    # Train model chronologically
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        shuffle=False,
        callbacks=[early_stopping],
        verbose=1
    )

    print("LSTM training completed.")

    # Generate predictions
    scaled_predictions = model.predict(
        X_test,
        verbose=0
    )

    predicted_values = scaler.inverse_transform(
        scaled_predictions
    ).flatten()

    # Create prediction results
    predictions_df = pd.DataFrame({
        "Date": prediction_dates,
        "actual_AQI": test_actuals,
        "predicted_AQI": predicted_values
    })

    # Evaluate only available actual AQI values
    evaluation_data = predictions_df.dropna(
        subset=["actual_AQI", "predicted_AQI"]
    )

    if len(evaluation_data) < 2:
        raise ValueError(
            f"Insufficient evaluation records for {city}."
        )

    y_true = evaluation_data["actual_AQI"]
    y_pred = evaluation_data["predicted_AQI"]

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    metrics = {
        "City": city,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Evaluated_Records": len(evaluation_data)
    }

    return model, predictions_df, metrics


def train_all_lstm(city_data, city_splits):
    """
    Train independent LSTM models for all cities.
    """

    trained_models = {}
    all_predictions = {}
    all_metrics = []

    for city, daily_data in city_data.items():

        model, predictions, metrics = (
            train_and_predict_lstm(
                city=city,
                daily_data=daily_data,
                city_split=city_splits[city]
            )
        )

        trained_models[city] = model
        all_predictions[city] = predictions
        all_metrics.append(metrics)

        print(
            f"MAE: {metrics['MAE']:.2f} | "
            f"RMSE: {metrics['RMSE']:.2f} | "
            f"R2: {metrics['R2']:.4f}"
        )

    results_df = pd.DataFrame(all_metrics)

    return trained_models, all_predictions, results_df
