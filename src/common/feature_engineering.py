
import pandas as pd


# Historical AQI values used as input features
LAG_DAYS = [0, 1, 2, 3, 7]

# Rolling windows
ROLLING_WINDOWS = [3, 7]

# Model input columns
FEATURE_COLUMNS = [
    "aqi_lag_0",
    "aqi_lag_1",
    "aqi_lag_2",
    "aqi_lag_3",
    "aqi_lag_7",
    "rolling_mean_3",
    "rolling_mean_7",
    "target_dayofweek",
    "target_month",
    "target_dayofyear",
    "target_is_weekend",
]

TARGET_COLUMN = "target_AQI"


def create_features_for_city(city_data):
    """
    Create supervised learning features for next-day AQI prediction.

    Inputs:
        Historical AQI values up to the current day.
        Calendar information for the next day.

    Target:
        AQI of the next day.
    """

    data = city_data.copy()

    # Ensure chronological order
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    if data["Date"].duplicated().any():
        raise ValueError("Duplicate dates found for this city.")

    # Check that dates are continuous
    date_difference = data["Date"].diff().dropna()

    if not date_difference.eq(pd.Timedelta(days=1)).all():
        raise ValueError(
            "Dates must be continuous daily dates. "
            "Run preprocessing before feature engineering."
        )

    features = pd.DataFrame(index=data.index)

    # Keep the city name
    features["City"] = data["City"]

    # Prediction origin: the day on which the forecast is made
    features["forecast_origin_date"] = data["Date"]

    # Target date: the following day
    features["Date"] = data["Date"].shift(-1)

    # Historical AQI lag features
    for lag in LAG_DAYS:
        features[f"aqi_lag_{lag}"] = data["AQI"].shift(lag)

    # Rolling averages include today's AQI,
    # but never use future AQI values.
    for window in ROLLING_WINDOWS:
        features[f"rolling_mean_{window}"] = (
            data["AQI"].rolling(window=window).mean()
        )

    # Calendar features of the target (next) day
    target_dates = features["Date"]

    features["target_dayofweek"] = target_dates.dt.dayofweek
    features["target_month"] = target_dates.dt.month
    features["target_dayofyear"] = target_dates.dt.dayofyear

    features["target_is_weekend"] = (
        target_dates.dt.dayofweek >= 5
    ).astype(float)

    # Tomorrow's AQI is the prediction target
    features[TARGET_COLUMN] = data["AQI"].shift(-1)

    # Remove samples with missing input features or target
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    features = features.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    # Convert weekend indicator to integer
    features["target_is_weekend"] = (
        features["target_is_weekend"].astype(int)
    )

    return features


def create_all_city_features(city_data):
    """
    Create next-day AQI features for all selected cities.

    Input:
        Dictionary returned by prepare_aqi_data().

    Output:
        Dictionary containing supervised data for each city.
    """

    city_features = {}

    for city, data in city_data.items():

        if data.empty:
            continue

        city_df = create_features_for_city(data)

        if city_df.empty:
            print(f"Warning: No valid training samples for {city}.")
            continue

        city_features[city] = city_df

    return city_features
