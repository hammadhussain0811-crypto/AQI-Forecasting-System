
from pathlib import Path
import warnings

import pandas as pd


# Cities selected for AQI forecasting
SELECTED_CITIES = [
    "Delhi",
    "Bengaluru",
    "Lucknow",
    "Chennai",
    "Hyderabad"
]


def load_raw_data(file_path):
    """Load the original CPCB dataset."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    data = pd.read_csv(file_path)

    required_columns = ["City", "Date", "AQI"]

    missing_columns = [
        col for col in required_columns if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns missing from dataset: {missing_columns}"
        )

    return data


def clean_basic_data(data):
    """Clean essential columns and check duplicate city-date records."""

    data = data.copy()

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["AQI"] = pd.to_numeric(data["AQI"], errors="coerce")

    # Remove rows without a valid city or date
    data = data.dropna(subset=["City", "Date"])

    data["City"] = data["City"].astype(str).str.strip()

    # Check duplicate records for the same city and date
    duplicates = data.duplicated(subset=["City", "Date"], keep=False)

    if duplicates.any():
        duplicate_rows = data.loc[duplicates, ["City", "Date"]]
        raise ValueError(
            "Duplicate City-Date records found:\n"
            f"{duplicate_rows.head(10)}"
        )

    return data.sort_values(["City", "Date"]).reset_index(drop=True)


def filter_selected_cities(data):
    """Keep only the five selected cities."""

    available_cities = set(data["City"].unique())

    for city in SELECTED_CITIES:
        if city not in available_cities:
            warnings.warn(f"{city} is not present in the dataset.")

    return data[data["City"].isin(SELECTED_CITIES)].copy()


def prepare_city_time_series(city_data):
    """Create a continuous daily date index for one city."""

    city_data = city_data.sort_values("Date").copy()

    city_name = city_data["City"].iloc[0]

    # Create a complete daily date range
    full_dates = pd.date_range(
        start=city_data["Date"].min(),
        end=city_data["Date"].max(),
        freq="D"
    )

    # Reindex so missing calendar dates become NaN
    city_data = city_data.set_index("Date")
    city_data = city_data.reindex(full_dates)

    city_data.index.name = "Date"
    city_data["City"] = city_name

    return city_data.reset_index()[["City", "Date", "AQI"]]


def prepare_aqi_data(file_path):
    """Run the complete preprocessing pipeline."""

    data = load_raw_data(file_path)

    data = clean_basic_data(data)

    data = filter_selected_cities(data)

    city_data = {}

    for city in SELECTED_CITIES:

        city_df = data[data["City"] == city].copy()

        if city_df.empty:
            continue

        city_data[city] = prepare_city_time_series(city_df)

    return city_data
