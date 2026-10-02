import json
import numpy as np
import pandas as pd
import joblib
import mlflow


# ==============================
# Paths
# ==============================

CONFIG_PATH = "models/model_config.json"
SCALER_PATH = "models/scaler.pkl"
TEST_PATH = "dataset/test_FD001.txt"
RUL_PATH = "dataset/RUL_FD001.txt"

MODEL_NAME = "PredictiveMaintenance_LSTM"
MODEL_VERSION = "1"


# ==============================
# Load configuration
# ==============================

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

sequence_length = config["sequence_length"]
rul_cap = config["rul_cap"]
features = config["lstm_features"]

print("Configuration loaded:")
print("Sequence length:", sequence_length)
print("RUL cap:", rul_cap)
print("Number of features:", len(features))


# ==============================
# Load scaler
# ==============================

print("\nLoading scaler...")

scaler = joblib.load(SCALER_PATH)

print("Scaler loaded successfully.")


# ==============================
# Load registered model
# ==============================

print("\nLoading registered model...")

model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.keras.load_model(model_uri)

print("Registered model loaded successfully.")


# ==============================
# Load test dataset
# ==============================

print("\nLoading C-MAPSS test data...")

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

print("Test data shape:", df.shape)


# ==============================
# Load actual RUL values
# ==============================

rul_df = pd.read_csv(
    RUL_PATH,
    sep=r"\s+",
    header=None
)

actual_rul = float(rul_df.iloc[0, 0])


# ==============================
# Select engine
# ==============================

engine_id = 1

engine_data = df[df["unit_id"] == engine_id].copy()

print(f"\nEngine {engine_id} observed cycles:", len(engine_data))
print(f"Actual RUL: {actual_rul:.0f} cycles")


# ==============================
# Prepare 50-cycle window
# ==============================

X_raw = engine_data[features].copy()

if len(X_raw) >= sequence_length:

    # Use the latest 50 cycles
    X_raw = X_raw.tail(sequence_length)

    print(f"Using latest {sequence_length} cycles.")

else:

    # Number of rows needed
    rows_needed = sequence_length - len(X_raw)

    # Repeat the earliest observed sensor values
    padding = pd.concat(
        [X_raw.iloc[[0]]] * rows_needed,
        ignore_index=True
    )

    X_raw = pd.concat(
        [padding, X_raw],
        ignore_index=True
    )

    print(
        f"Engine has only {len(engine_data)} cycles."
    )
    print(
        f"Added {rows_needed} padded cycles."
    )


# ==============================
# Scale features
# ==============================

X_scaled = scaler.transform(X_raw)

# Add batch dimension
X_input = np.expand_dims(X_scaled, axis=0)

print("Model input shape:", X_input.shape)


# ==============================
# Predict RUL
# ==============================

prediction = model.predict(
    X_input,
    verbose=0
)

predicted_rul = float(prediction[0][0])

predicted_rul = np.clip(
    predicted_rul,
    0,
    rul_cap
)


# ==============================
# Error
# ==============================

absolute_error = abs(
    predicted_rul - actual_rul
)


# ==============================
# Result
# ==============================

print("\n" + "=" * 55)
print("REAL ENGINE RUL PREDICTION")
print("=" * 55)

print(f"Engine ID:              {engine_id}")
print(f"Observed cycles:        {len(engine_data)}")
print(f"Model window:           {sequence_length} cycles")
print(f"Actual RUL:             {actual_rul:.2f} cycles")
print(f"Predicted RUL:          {predicted_rul:.2f} cycles")
print(f"Absolute error:          {absolute_error:.2f} cycles")

print("=" * 55)