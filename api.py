import json

import joblib
from tensorflow import keras
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException


# ==========================================
# Configuration
# ==========================================

CONFIG_PATH = "models/model_config.json"
SCALER_PATH = "models/scaler.pkl"
MODEL_PATH = "models/lstm_50_capped.keras"
TEST_PATH = "dataset/test_FD001.txt"

MODEL_NAME = "PredictiveMaintenance_LSTM"
MODEL_VERSION = "1"


# ==========================================
# FastAPI app
# ==========================================

app = FastAPI(
    title="Predictive Maintenance API",
    description="LSTM-based Remaining Useful Life prediction API",
    version="1.0.0"
)


# ==========================================
# Load model configuration
# ==========================================

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

SEQUENCE_LENGTH = config["sequence_length"]
RUL_CAP = config["rul_cap"]
FEATURES = config["lstm_features"]


# ==========================================
# Load scaler
# ==========================================

scaler = joblib.load(SCALER_PATH)


# ==========================================
# Load registered MLflow model
# ==========================================

MODEL_PATH = "models/lstm_50_capped.keras"

model = keras.models.load_model(MODEL_PATH)


# ==========================================
# Load C-MAPSS test data
# ==========================================

columns = [
    "unit_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
]

sensor_columns = [f"sensor_{i}" for i in range(1, 22)]

df = pd.read_csv(
    TEST_PATH,
    sep=r"\s+",
    header=None
)

df.columns = columns + sensor_columns


# ==========================================
# Health check
# ==========================================

@app.get("/")
def root():
    return {
        "message": "Predictive Maintenance API is running",
        "model": MODEL_NAME,
        "version": MODEL_VERSION
    }


# ==========================================
# RUL prediction
# ==========================================

@app.get("/predict/{engine_id}")
def predict_rul(engine_id: int):

    engine_data = df[df["unit_id"] == engine_id].copy()

    if engine_data.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Engine {engine_id} not found."
        )

    X_raw = engine_data[FEATURES].copy()

    observed_cycles = len(X_raw)

    # Prepare 50-cycle sequence
    if observed_cycles >= SEQUENCE_LENGTH:

        X_raw = X_raw.tail(SEQUENCE_LENGTH)

    else:

        rows_needed = SEQUENCE_LENGTH - observed_cycles

        padding = pd.concat(
            [X_raw.iloc[[0]]] * rows_needed,
            ignore_index=True
        )

        X_raw = pd.concat(
            [padding, X_raw],
            ignore_index=True
        )

    # Scale
    X_scaled = scaler.transform(X_raw)

    # Add batch dimension
    X_input = np.expand_dims(X_scaled, axis=0)

    # Predict
    prediction = model.predict(
        X_input,
        verbose=0
    )

    predicted_rul = float(prediction[0][0])

    predicted_rul = float(
        np.clip(predicted_rul, 0, RUL_CAP)
    )

    return {
        "engine_id": engine_id,
        "observed_cycles": observed_cycles,
        "predicted_rul": round(predicted_rul, 2),
        "unit": "cycles",
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION
    }