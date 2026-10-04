# ✈️ Predictive Maintenance & RUL Prediction

An end-to-end **Predictive Maintenance system** that predicts the **Remaining Useful Life (RUL)** of aircraft engines using multivariate time-series sensor data from the **NASA C-MAPSS FD001 dataset**.

The project combines **LSTM Deep Learning, MLflow, FastAPI, and Streamlit** to build a complete machine learning and MLOps workflow — from model training and evaluation to experiment tracking, model registration, API serving, and an interactive dashboard.

---

## 🎯 Project Overview

Aircraft engines generate sensor measurements during operation. As an engine degrades, these sensor patterns change over time.

This project uses historical sensor data to learn these degradation patterns and predict:

> **How many operating cycles an engine has remaining before failure?**

The final model is an **LSTM-based regression model** trained on sequences of **50 operating cycles** using **15 selected sensor features**.

---

## 🚀 Key Features

- ✈️ Remaining Useful Life prediction
- 🧠 LSTM-based time-series regression
- 📊 Comparison of multiple machine learning models
- 🔬 Sensor feature selection and preprocessing
- 📈 Model evaluation using MAE, RMSE, and R²
- 🧪 MLflow experiment tracking
- 📦 MLflow Model Registry
- ⚡ FastAPI REST API
- 🎨 Interactive Streamlit dashboard
- 🔄 End-to-end ML and MLOps workflow

---

## 🗂️ Dataset

This project uses the **NASA C-MAPSS FD001** dataset.

The dataset contains simulated aircraft engine degradation data with:

- Engine ID
- Operating cycle
- 3 operational settings
- 21 sensor measurements

For the training data, Remaining Useful Life is calculated as:

```text
RUL = Maximum Engine Cycle - Current Cycle
```

The target RUL is capped at **125 cycles** to focus the model on the degradation region most relevant to predictive maintenance.

> **Note:** The raw C-MAPSS dataset is not included in this repository.

---

## 🔬 Data Preprocessing

The preprocessing pipeline includes:

1. Loading the C-MAPSS FD001 dataset
2. Assigning meaningful column names
3. Calculating training RUL
4. Removing constant and near-constant sensors
5. Selecting 15 informative sensor features
6. Scaling sensor features
7. Creating sliding-window sequences
8. Using a sequence length of 50 cycles
9. Capping RUL at 125 cycles

### Selected Sensor Features

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

---

# 🤖 Model

The final model is a **Long Short-Term Memory (LSTM)** neural network.

LSTM is well suited for this problem because engine sensor data is sequential, and degradation depends on patterns observed across previous operating cycles.

## Final Configuration

| Parameter | Value |
|---|---|
| **Model** | LSTM |
| **Sequence Length** | 50 cycles |
| **Input Features** | 15 sensors |
| **RUL Cap** | 125 cycles |
| **Task** | Regression |

---

# 📊 Model Performance

## Final LSTM — Validation

| Metric | Score |
|---|---:|
| **MAE** | **8.24 cycles** |
| **RMSE** | **11.31 cycles** |
| **R²** | **0.925** |

## Final LSTM — Official FD001 Test

| Metric | Score |
|---|---:|
| **MAE** | **10.22 cycles** |
| **RMSE** | **14.12 cycles** |
| **R²** | **0.798** |

### What Does This Mean?

The final model achieved a test **MAE of 10.22 cycles**, meaning that the predicted RUL differs from the actual RUL by approximately **10 operating cycles on average**.

---

# 📈 Model Comparison

Several models were evaluated during development.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Baseline Random Forest | 25.86 | 35.35 | 0.710 |
| Temporal Random Forest | 24.20 | 32.87 | 0.749 |
| XGBoost | 24.54 | 33.42 | 0.741 |
| Fair Random Forest | 21.80 | 30.37 | 0.726 |
| 30-Cycle LSTM | 16.87 | 23.58 | 0.835 |
| 50-Cycle LSTM | 13.64 | 19.66 | 0.863 |
| **50-Cycle Capped LSTM** | **8.24** | **11.31** | **0.925** |

The **50-cycle capped LSTM** achieved the strongest validation performance among the evaluated models.

---

# 🏗️ System Architecture

```text
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
              ┌─────────────┴─────────────┐
              ▼                           ▼
       MLflow Tracking             Model Artifacts
       & Model Registry
              │
              ▼
        FastAPI REST API
              │
              ▼
       Streamlit Dashboard
```

---

# 🛠️ Tech Stack

## Machine Learning

- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas

## MLOps

- MLflow
- MLflow Model Registry
- Experiment Tracking
- Model Versioning

## Backend

- FastAPI
- Uvicorn

## Frontend

- Streamlit

## Development

- Jupyter Notebook
- Git
- GitHub

---

# 📁 Project Structure

```text
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

> The raw C-MAPSS dataset, virtual environment, MLflow database, MLflow run artifacts, and Python cache files are excluded from the repository.

---

# 🔄 MLOps Workflow

MLflow is used to track experiments and manage the trained model.

## Experiment Tracking

The following information is tracked:

- Model parameters
- Sequence length
- RUL cap
- Number of features
- Validation metrics
- Test metrics
- Model artifacts

## Model Registry

The final model is registered as:

```text
PredictiveMaintenance_LSTM
```

Current registered version:

```text
Version 1
```

The FastAPI application loads the registered model for inference.

---

# ⚡ FastAPI

The trained model is exposed through a REST API.

## Start the API

```bash
uvicorn api:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

## Prediction Endpoint

```text
GET /predict/{engine_id}
```

Example:

```text
http://127.0.0.1:8000/predict/1
```

## Example Response

```json
{
    "engine_id": 1,
    "observed_cycles": 31,
    "predicted_rul": 124.8,
    "unit": "cycles",
    "model": "PredictiveMaintenance_LSTM",
    "model_version": "1"
}
```

---

# 🎨 Streamlit Dashboard

The project includes an interactive Streamlit dashboard for generating RUL predictions.

## Start the Dashboard

```bash
streamlit run app.py
```

The dashboard allows users to:

- Select an Engine ID
- Generate an RUL prediction
- View predicted remaining cycles
- View observed cycles
- View the registered model and version

The Streamlit application communicates with the FastAPI backend.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/MousumiBadyakar/Predictive-Maintenance.git
cd Predictive-Maintenance
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

## 3. Activate the Environment

### Windows

```bash
.venv\Scripts\activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📊 MLflow

Start the MLflow tracking server:

```bash
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

MLflow can be used to inspect:

- Experiments
- Runs
- Parameters
- Metrics
- Model artifacts
- Registered models

---

# ▶️ Running the Project

The recommended workflow is to use three terminals.

## Terminal 1 — MLflow

```bash
mlflow ui
```

## Terminal 2 — FastAPI

```bash
uvicorn api:app --reload
```

## Terminal 3 — Streamlit

```bash
streamlit run app.py
```

Then open the Streamlit URL displayed in the terminal.

---

# 🧪 Inference

A standalone inference script is also included.

Run:

```bash
python inference.py
```

The script loads:

- Registered LSTM model
- Feature scaler
- Model configuration
- FD001 test data

and generates an RUL prediction for a selected engine.

---

# ⚠️ Important Note About Test Data

The official C-MAPSS FD001 test set contains **partial engine histories**.

For the Streamlit/API demonstration, engines with fewer than 50 observed cycles are padded to satisfy the LSTM's required sequence length.

This padding strategy is used only for the **interactive inference demonstration**.

The **official FD001 test metrics reported above are calculated using the proper test evaluation procedure** and should be considered the authoritative model performance.

---

# 💡 Key Learnings

Through this project, I worked with:

- Time-series preprocessing
- Remaining Useful Life prediction
- LSTM sequence modeling
- Feature selection
- Sliding-window generation
- Model evaluation
- MLflow experiment tracking
- MLflow Model Registry
- REST API development with FastAPI
- Streamlit application development
- Model serving
- Git and GitHub workflow
🔮 Future Improvements
- Deploy the FastAPI backend to the cloud
- Deploy the Streamlit frontend
- Add automated CI/CD
- Add Docker containerization
- Add real-time sensor data simulation
- Add model monitoring
- Add automated model retraining
- Add prediction visualizations and degradation curves

## 👩‍💻 Author
Mousumi Badyakar
