
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.common.feature_engineering import (
    FEATURE_COLUMNS,
    TARGET_COLUMN
)


def train_and_predict_random_forest(
    city,
    city_split,
    random_state=42
):
    """
    Train a Random Forest regression model for one city
    and predict next-day AQI on its test dataset.
    """

    train_data = city_split["train"].copy()
    test_data = city_split["test"].copy()

    # Prepare training features and target
    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMN]

    # Prepare testing features and target
    X_test = test_data[FEATURE_COLUMNS]
    y_test = test_data[TARGET_COLUMN]

    print(f"\nTraining Random Forest for {city}...")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    # Initialize Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=random_state,
        n_jobs=-1
    )

    # Train model
    model.fit(X_train, y_train)

    print("Model training completed.")

    # Generate predictions
    predictions = model.predict(X_test)

    # Create prediction results
    predictions_df = pd.DataFrame({
        "Date": test_data["Date"].values,
        "actual_AQI": y_test.values,
        "predicted_AQI": predictions
    })

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predictions)

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(y_test, predictions)

    metrics = {
        "City": city,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Evaluated_Records": len(y_test)
    }

    return model, predictions_df, metrics


def train_all_random_forest(city_splits):
    """
    Train Random Forest independently for all cities.
    """

    trained_models = {}
    all_predictions = {}
    all_metrics = []

    for city, splits in city_splits.items():

        model, predictions, metrics = (
            train_and_predict_random_forest(
                city=city,
                city_split=splits
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
