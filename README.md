# ✈️ Predictive Maintenance & RUL Prediction

An end-to-end **Predictive Maintenance system** that predicts the **Remaining Useful Life (RUL)** of aircraft engines using multivariate time-series sensor data from the **NASA C-MAPSS FD001 dataset**.

The project combines **Deep Learning, MLflow, FastAPI, and Streamlit** to create a complete machine learning pipeline — from model training and evaluation to model tracking, registration, API serving, and an interactive web interface.

---

## 🚀 Project Overview

Aircraft engines generate continuous sensor measurements during operation. As an engine degrades, these sensor patterns change over time.

This project uses historical sensor data to learn those degradation patterns and estimate:

> **How many operating cycles an engine has remaining before failure?**

The final model is an **LSTM-based time-series regression model** trained on sequences of 50 cycles using 15 selected sensor features.

---

## 🎯 Objectives

- Predict Remaining Useful Life (RUL) of aircraft engines
- Capture temporal degradation patterns using LSTM
- Compare traditional ML models with deep learning
- Track experiments and metrics using MLflow
- Register the trained model using MLflow Model Registry
- Serve predictions through a FastAPI REST API
- Build an interactive Streamlit dashboard
- Create a complete MLOps-oriented project workflow

---

## 🧠 Machine Learning Approach

### Dataset

The project uses the **NASA C-MAPSS FD001** dataset.

Each observation contains:

- Engine ID
- Operating cycle
- 3 operational settings
- 21 sensor measurements

For training data, RUL is calculated using:

```text
RUL = Maximum Engine Cycle - Current Cycle
```
To make the model focus on realistic degradation behavior, the target RUL is capped at 125 cycles.

---

## 🔬 Data Preprocessing
The following preprocessing steps were performed:
1. Loaded the C-MAPSS FD001 training and test data
2. Assigned meaningful column names
3. Calculated RUL for training engines
4. Removed constant / near-constant sensors
5. Selected 15 informative sensor features
6. Applied feature scaling
7. Created sliding-window sequences
8. Used a sequence length of 50 cycles
9. Capped RUL at 125 cycle

## Selected Features
```text
sensor_2
sensor_3
sensor_4
sensor_6
sensor_7
sensor_8
sensor_9
sensor_11
sensor_12
sensor_13
sensor_14
sensor_15
sensor_17
sensor_20
sensor_21
```

## 🤖 Model
The final model is a Long Short-Term Memory (LSTM) neural network.
LSTM is suitable for this problem because engine sensor data is sequential and degradation depends on patterns observed across previous operating cycles.

### Final Configuration

| Parameter | Value |
|:---|:---|
| **Model** | LSTM |
| **Sequence Length** | 50 cycles |
| **Input Features** | 15 sensors |
| **RUL Cap** | 125 cycles |
| **Task** | Regression |

### 📊 Model Performance
### 📊 Final LSTM — Validation

| Metric | Score |
|---|---:|
| **MAE** | **8.24 cycles** |
| **RMSE** | **11.31 cycles** |
| **R²** | **0.925** |


### 📊 Final LSTM — Official FD001 Test

| Metric | Score |
|---|---:|
| **MAE** | **10.22 cycles** |
| **RMSE** | **14.12 cycles** |
| **R²** | **0.798** |



## What does this mean?
The final model achieved a test MAE of 10.22 cycles, meaning that the predicted RUL differs from the actual RUL by approximately 10 operating cycles on average.

### 📈 Model Comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Baseline Random Forest | 25.86 | 35.35 | 0.710 |
| Temporal Random Forest | 24.20 | 32.87 | 0.749 |
| XGBoost | 24.54 | 33.42 | 0.741 |
| Fair Random Forest | 21.80 | 30.37 | 0.726 |
| 30-Cycle LSTM | 16.87 | 23.58 | 0.835 |
| 50-Cycle LSTM | 13.64 | 19.66 | 0.863 |
| **50-Cycle Capped LSTM** | **8.24** | **11.31** | **0.925** |

The 50-cycle capped LSTM achieved the strongest validation performance among the evaluated models.

---

## 🏗️ System Architecture

                 NASA C-MAPSS FD001
                         │
                         ▼
              Data Preprocessing
                         │
                         ▼
              Feature Selection
                  15 Sensors
                         │
                         ▼
             Sliding Window Creation
                  50 Cycles
                         │
                         ▼
                    LSTM Model
                         │
                         ▼
                  RUL Prediction
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
          MLflow                 Model
     Tracking & Registry       Artifacts
              │
              ▼
       FastAPI REST API
              │
              ▼
       Streamlit Dashboard
---

## 🛠️ Tech Stack
### Machine Learning
- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas
### MLOps
- MLflow
- MLflow Model Registry
- Model versioning
- Experiment tracking
### Backend
- FastAPI
- Uvicorn
### Frontend
- Streamlit
### Development
- Jupyter Notebook
- Git
- GitHub
---

## 📁 Project Structure
```
Predictive-Maintenance/
│
├── app.py
├── api.py
├── inference.py
├── mlflow_tracking.py
├── register_model.py
├── requirements.txt
├── .gitignore
│
├── models/
│   ├── lstm_50_capped.keras
│   ├── scaler.pkl
│   ├── model_config.json
│   └── metrics.json
│
└── Predictive_Maintenance_RUL_Prediction.ipynb
```
The raw C-MAPSS dataset, virtual environment, MLflow database, and MLflow run artifacts are excluded from the repository.

---

## 🔄 MLOps Workflow
The project uses MLflow to track and manage the trained model.
### Experiment Tracking
The following information is tracked:
- Model parameters
- Sequence length
- RUL cap
- Number of features
- Validation metrics
- Test metrics
- Model artifacts
### Model Registry
The final model is registered as:
PredictiveMaintenance_LSTM

### Current registered version:
Version 1

The FastAPI application loads the registered model for inference.

---

