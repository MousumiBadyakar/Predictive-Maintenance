import mlflow

RUN_ID = "e95a206b563f4c96a3e913c870e58240"
MODEL_NAME = "PredictiveMaintenance_LSTM"

model_uri = f"runs:/{RUN_ID}/lstm_rul_model"

print("Registering model...")
print("Model URI:", model_uri)

result = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

print("\n========================================")
print("MODEL REGISTERED SUCCESSFULLY!")
print("========================================")
print("Name:", result.name)
print("Version:", result.version)
print("========================================")