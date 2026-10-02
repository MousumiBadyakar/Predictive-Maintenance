import json
import os

import mlflow
import mlflow.keras
import tensorflow as tf
from mlflow.models import infer_signature


# ============================================================
# 1. Project paths
# ============================================================

MODEL_PATH = "models/lstm_50_capped.keras"
SCALER_PATH = "models/scaler.pkl"
CONFIG_PATH = "models/model_config.json"
METRICS_PATH = "models/metrics.json"


# ============================================================
# 2. Check required files
# ============================================================

required_files = [
    MODEL_PATH,
    SCALER_PATH,
    CONFIG_PATH,
    METRICS_PATH
]

for file_path in required_files:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")

print("All model files found.")


# ============================================================
# 3. Load configuration and metrics
# ============================================================

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)


# ============================================================
# 4. Load trained Keras model
# ============================================================

print("Loading trained LSTM model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# 5. Create a sample input for model signature
# ============================================================

sequence_length = config["sequence_length"]
num_features = len(config["lstm_features"])

sample_input = tf.zeros(
    (1, sequence_length, num_features),
    dtype=tf.float32
)

sample_output = model.predict(sample_input, verbose=0)

signature = infer_signature(
    sample_input.numpy(),
    sample_output
)


# ============================================================
# 6. Create MLflow experiment
# ============================================================

mlflow.set_experiment("Predictive_Maintenance_FD001")


# ============================================================
# 7. Start MLflow run
# ============================================================

with mlflow.start_run(run_name="LSTM_50_Capped_Final"):

    # -----------------------------
    # Parameters
    # -----------------------------

    mlflow.log_param("model", "LSTM")
    mlflow.log_param(
        "sequence_length",
        sequence_length
    )
    mlflow.log_param(
        "rul_cap",
        config["rul_cap"]
    )
    mlflow.log_param(
        "num_features",
        num_features
    )
    mlflow.log_param(
        "dataset",
        "NASA C-MAPSS FD001"
    )

    # -----------------------------
    # Validation metrics
    # -----------------------------

    mlflow.log_metric(
        "val_mae",
        metrics["validation"]["mae"]
    )

    mlflow.log_metric(
        "val_rmse",
        metrics["validation"]["rmse"]
    )

    mlflow.log_metric(
        "val_r2",
        metrics["validation"]["r2"]
    )

    # -----------------------------
    # Official test metrics
    # -----------------------------

    mlflow.log_metric(
        "test_mae",
        metrics["test"]["mae"]
    )

    mlflow.log_metric(
        "test_rmse",
        metrics["test"]["rmse"]
    )

    mlflow.log_metric(
        "test_r2",
        metrics["test"]["r2"]
    )

    # -----------------------------
    # Supporting artifacts
    # -----------------------------

    mlflow.log_artifact(CONFIG_PATH)
    mlflow.log_artifact(METRICS_PATH)
    mlflow.log_artifact(SCALER_PATH)

    # -----------------------------
    # Log actual Keras model
    # -----------------------------

    model_info = mlflow.keras.log_model(
        model,
        name="lstm_rul_model",
        signature=signature
    )

    print("\n" + "=" * 60)
    print("MLflow run completed successfully!")
    print("=" * 60)
    print("Run ID:", mlflow.active_run().info.run_id)
    print("Model URI:", model_info.model_uri)