
import requests
import pandas as pd
import streamlit as st


# PAGE CONFIGURATION

st.set_page_config(
    page_title="AQI Forecasting System",
    page_icon="🌍",
    layout="wide"
)


# API CONFIGURATION

API_URL = "http://127.0.0.1:8000"

CITIES = [
    "Delhi",
    "Bengaluru",
    "Lucknow",
    "Chennai",
    "Hyderabad"
]


# AQI CATEGORY CLASSIFICATION

def get_aqi_category(aqi):

    if aqi < 0:
        return "Invalid AQI", "#808080"

    elif aqi <= 50:
        return "Good", "#00B050"

    elif aqi <= 100:
        return "Satisfactory", "#92D050"

    elif aqi <= 200:
        return "Moderate", "#FFFF00"

    elif aqi <= 300:
        return "Poor", "#FF9900"

    elif aqi <= 400:
        return "Very Poor", "#FF0000"

    elif aqi <= 500:
        return "Severe", "#800000"

    else:
        return "Above CPCB Scale", "#808080"


# DISPLAY AQI CATEGORY

def display_aqi_category(aqi):

    category, color = get_aqi_category(aqi)

    text_color = "#000000" if color in [
        "#FFFF00",
        "#92D050"
    ] else "#FFFFFF"

    st.markdown(
        f"""
        <div style="
            background-color: {color};
            color: {text_color};
            padding: 12px;
            border-radius: 8px;
            text-align: center;
            font-weight: bold;
            margin-bottom: 10px;
        ">
            {category}
        </div>
        """,
        unsafe_allow_html=True
    )


# PAGE HEADER

st.title("🌍 AQI Forecasting System")

st.markdown(
    """
    ### Next-Day Air Quality Index Prediction

    Compare predictions from four machine learning and
    time-series models for five Indian cities.
    """
)

st.divider()


# API CONNECTION CHECK

try:

    health_response = requests.get(
        f"{API_URL}/health",
        timeout=5
    )

    health_response.raise_for_status()

    st.success("Backend connected successfully.")

except requests.RequestException:

    st.error(
        "Cannot connect to FastAPI. "
        "Please start the backend first."
    )

    st.stop()


# CITY SELECTION

st.subheader("Select Your City")

selected_city = st.selectbox(
    "Choose a city",
    CITIES
)


# PREDICTION REQUEST

if st.button("Predict AQI", type="primary"):

    with st.spinner(
        f"Generating AQI predictions for {selected_city}..."
    ):

        try:

            response = requests.post(
                f"{API_URL}/api/predict",
                json={"city": selected_city},
                timeout=120
            )

            response.raise_for_status()

            result = response.json()

            st.session_state["prediction_result"] = result

        except requests.RequestException as error:

            st.error(
                f"Prediction request failed: {error}"
            )

            st.session_state.pop(
                "prediction_result",
                None
            )


# DISPLAY PREDICTION RESULTS

if "prediction_result" in st.session_state:

    result = st.session_state["prediction_result"]

    st.divider()

    st.subheader(
        f"AQI Prediction Results - {result['city']}"
    )

    st.info(
        f"Forecast date: {result['forecast_date']}"
    )

    predictions = result["predictions"]

    results_df = pd.DataFrame(predictions)

    # BEST PERFORMING MODEL

    st.subheader("Best Performing Model")

    best_model = result.get("best_model")
    best_model_mae = result.get("best_model_mae")

    if best_model is not None and best_model_mae is not None:

        best_prediction_data = next(
            (
                item for item in predictions
                if item["model"] == best_model
            ),
            None
        )

        if best_prediction_data is not None:

            best_prediction = float(
                best_prediction_data["predicted_aqi"]
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Best Model",
                    best_model
                )

            with col2:
                st.metric(
                    "Predicted AQI",
                    f"{best_prediction:.2f}"
                )

            with col3:
                st.metric(
                    "Historical MAE",
                    f"{best_model_mae:.2f}"
                )

            st.markdown("### Predicted Air Quality")

            display_aqi_category(best_prediction)

            st.caption(
                "Best model selected using historical test-set MAE. "
                "Historical MAE is not the error of this individual forecast."
            )

        else:

            st.warning(
                "Best model prediction is unavailable."
            )

    else:

        st.warning(
            "Best model information is missing from the backend response."
        )

    # INDIVIDUAL MODEL PREDICTIONS

    st.subheader("Predictions from All Four Models")

    columns = st.columns(4)

    for column, prediction in zip(columns, predictions):

        predicted_aqi = float(
            prediction["predicted_aqi"]
        )

        category, color = get_aqi_category(predicted_aqi)

        with column:

            st.metric(
                label=prediction["model"],
                value=f"{predicted_aqi:.2f}"
            )

            display_aqi_category(predicted_aqi)

    # COMPARISON TABLE

    st.subheader("Model Comparison")

    results_df["predicted_aqi"] = (
        results_df["predicted_aqi"].round(2)
    )

    results_df["AQI Category"] = (
        results_df["predicted_aqi"].apply(
            lambda value: get_aqi_category(value)[0]
        )
    )

    display_df = results_df.rename(
        columns={
            "model": "Model",
            "predicted_aqi": "Predicted AQI"
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # COMPARISON CHART

    st.subheader("AQI Prediction Comparison")

    chart_df = results_df.set_index("model")

    st.bar_chart(
        chart_df["predicted_aqi"],
        y_label="Predicted AQI",
        x_label="Model"
    )

    st.caption(
        "These values are model predictions, "
        "not actual measured AQI."
    )

    st.caption(
        "AQI categories are assigned using CPCB numerical ranges. "
        "They classify the predicted AQI values; they are not "
        "independently calculated from pollutant concentrations."
    )


# AQI CATEGORY REFERENCE

st.divider()

st.subheader("CPCB AQI Category Reference")

category_data = pd.DataFrame({
    "Category": [
        "Good",
        "Satisfactory",
        "Moderate",
        "Poor",
        "Very Poor",
        "Severe"
    ],
    "AQI Range": [
        "0–50",
        "51–100",
        "101–200",
        "201–300",
        "301–400",
        "401–500"
    ]
})

st.dataframe(
    category_data,
    use_container_width=True,
    hide_index=True
)


# FOOTER

st.divider()

st.caption(
    "AQI Forecasting System | "
    "SARIMA • Random Forest • XGBoost • LSTM"
)

st.caption(
    "AQI category reference: Central Pollution Control Board (CPCB), India."
)
