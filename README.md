# AQI Forecasting System for Indian Cities

## Project Overview

The AQI Forecasting System is a machine learning-based project that predicts the next day's Air Quality Index (AQI) for five selected Indian cities.

The project compares four forecasting models:

- SARIMA
- Random Forest
- XGBoost
- Long Short-Term Memory (LSTM)

The system evaluates the models using historical test data and identifies the best-performing model for each city based on Mean Absolute Error (MAE).

## Cities Covered

1. Delhi
2. Bengaluru
3. Lucknow
4. Chennai
5. Hyderabad

## Project Objectives

- Analyze historical air quality data.
- Perform data preprocessing and feature engineering.
- Develop and compare four forecasting models.
- Evaluate model performance using MAE, RMSE and R².
- Identify the best-performing model for each city.
- Build an interactive dashboard for AQI predictions.

## Dataset

The project uses historical air quality data from the Central Pollution Control Board (CPCB).

**Dataset file:** `city_day.csv`

The dataset contains historical daily AQI observations and pollutant measurements.

**Important limitation:** The available dataset ends on 1 July 2020. Therefore, the current prediction pipeline forecasts 2 July 2020 using the historical observations. It does not currently provide live AQI forecasts.

## Technologies Used

| Component | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Statistical Model | SARIMA |
| Machine Learning | Scikit-learn, XGBoost |
| Deep Learning | TensorFlow / Keras |
| Backend | FastAPI |
| Frontend | Streamlit |
| Model Persistence | Joblib, Pickle, Keras |

## Machine Learning Models

### 1. SARIMA
A statistical time-series forecasting model that captures trend and seasonal patterns.

### 2. Random Forest
An ensemble learning algorithm that combines multiple decision trees to make predictions.

### 3. XGBoost
A gradient boosting algorithm that builds decision trees sequentially to improve prediction performance.

### 4. LSTM
A recurrent neural network architecture designed to learn patterns from sequential data.

## Model Evaluation

The models are evaluated using:

- **MAE:** Measures the average absolute prediction error.
- **RMSE:** Measures prediction error while penalizing larger errors.
- **R² Score:** Measures how well the model explains variation in the target values.

The model with the lowest historical test-set MAE is selected as the best-performing model for each city among the four evaluated models.

## Project Structure

```text
AQI-Forecasting-System/
│
├── dataset/
│   ├── raw/
│   │   └── city_day.csv
│   └── processed/
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── common/
│   │   ├── preprocessing.py
│   │   ├── feature_engineering.py
│   │   └── splitting.py
│   │
│   ├── training/
│   │   ├── train_sarima.py
│   │   ├── train_random_forest.py
│   │   ├── train_xgboost.py
│   │   └── train_lstm.py
│   │
│   ├── evaluation/
│   │   └── evaluate_models.py
│   │
│   └── prediction/
│       └── predict.py
│
├── models/
│   ├── sarima/
│   ├── random_forest/
│   ├── xgboost/
│   └── lstm/
│
├── backend/
│   └── main.py
│
├── frontend/
│   ├── app.py
│   └── requirements.txt
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Application Architecture

The project follows this workflow:

1. Historical CPCB data is loaded.
2. Data preprocessing and feature engineering are performed.
3. Data is divided chronologically into training and testing sets.
4. Four forecasting models are trained and evaluated.
5. Trained models are saved for future predictions.
6. The prediction module loads the trained models.
7. FastAPI provides prediction endpoints.
8. Streamlit communicates with FastAPI and displays the predictions.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/hammadhussain0811-crypto/AQI-Forecasting-System.git
cd AQI-Forecasting-System
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the FastAPI Backend

```bash
python -m uvicorn backend.main:app --reload
```

API documentation:

`http://127.0.0.1:8000/docs`

### 6. Start the Streamlit Frontend

Open another terminal and activate the virtual environment if needed.

```bash
streamlit run frontend/app.py
```

## Features

- AQI forecasting for five Indian cities.
- Comparison of four forecasting models.
- Best-performing model identification.
- AQI category classification using CPCB ranges.
- Colour-coded air quality indicators.
- Interactive prediction dashboard.
- FastAPI backend integration.

## Future Enhancements

- Integration of updated AQI observations.
- Live or near-real-time AQI forecasting.
- Forecasting for additional Indian cities.
- Improved model hyperparameter tuning.
- Extended forecasting horizons.

## Project Author

Developed as an academic machine learning project.
