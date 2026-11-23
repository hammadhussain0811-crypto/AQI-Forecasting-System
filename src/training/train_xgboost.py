
import pandas as pd

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.common.feature_engineering import (
    FEATURE_COLUMNS,
    TARGET_COLUMN
)


def train_and_predict_xgboost(
    city,
    city_split,
    random_state=42
):
    """
    Train an XGBoost regression model for one city
    and predict next-day AQI.
    """

    train_data = city_split["train"].copy()
    test_data = city_split["test"].copy()

    # Prepare training data
    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMN]

    # Prepare testing data
    X_test = test_data[FEATURE_COLUMNS]
    y_test = test_data[TARGET_COLUMN]

    print(f"\nTraining XGBoost for {city}...")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    # Initialize XGBoost model
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=random_state,
        n_jobs=-1
    )

    # Train the model
    model.fit(X_train, y_train)

    print("Model training completed.")

    # Generate predictions
    predictions = model.predict(X_test)

    # Create prediction DataFrame
    predictions_df = pd.DataFrame({
        "Date": test_data["Date"].values,
        "actual_AQI": y_test.values,
        "predicted_AQI": predictions
    })

    # Evaluate model
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


def train_all_xgboost(city_splits):
    """
    Train independent XGBoost models for all cities.
    """

    trained_models = {}
    all_predictions = {}
    all_metrics = []

    for city, splits in city_splits.items():

        model, predictions, metrics = (
            train_and_predict_xgboost(
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
