
import pandas as pd

from src.common.feature_engineering import (
    FEATURE_COLUMNS,
    TARGET_COLUMN
)


def split_city_data(city_data, train_ratio=0.80):
    """
    Split one city's feature dataset chronologically.

    Earlier observations are used for training.
    Later observations are used for testing.

    No random shuffling is performed.
    """

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    data = city_data.copy()

    # Ensure data is arranged chronologically
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    # Check required columns
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        col for col in required_columns if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if data[required_columns].isna().any().any():
        raise ValueError(
            "Missing values found in features or target."
        )

    # Calculate chronological split position
    split_index = int(len(data) * train_ratio)

    if split_index == 0 or split_index == len(data):
        raise ValueError("Not enough records for train/test splitting.")

    # Split without shuffling
    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()

    return {
        "train": train_data,
        "test": test_data
    }


def split_all_cities(city_features, train_ratio=0.80):
    """
    Perform chronological train/test splitting for all cities.
    """

    city_splits = {}

    for city, data in city_features.items():

        if data.empty:
            continue

        city_splits[city] = split_city_data(
            data,
            train_ratio=train_ratio
        )

    return city_splits
