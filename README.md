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
Final Configuration
Parameter	Value
Model	LSTM
Sequence Length	50 cycles
Input Features	15 sensors
RUL Cap	125 cycles
Task	Regression

📊 Model Performance
Final LSTM — Validation
Metric	Score
MAE	8.24 cycles
RMSE	11.31 cycles
R²	0.925


Final LSTM — Official FD001 Test
Metric	Score
MAE	10.22 cycles
RMSE	14.12 cycles
R²	0.798


What this means
The final model's test MAE of 10.22 cycles means that, on average, the predicted remaining useful life differs from the actual RUL by approximately 10 operating cycles.
📈 Model Comparison
Several models were evaluated during development.
Model	MAE	RMSE	R²
Baseline Random Forest	25.86	35.35	0.710
Temporal Random Forest	24.20	32.87	0.749
XGBoost	24.54	33.42	0.741
Fair Random Forest	21.80	30.37	0.726
30-Cycle LSTM	16.87	23.58	0.835
50-Cycle LSTM	13.64	19.66	0.863
50-Cycle Capped LSTM	8.24	11.31	0.925


The final capped LSTM demonstrated the strongest validation performance among the evaluated approaches.
🏗️ System Architecture
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

🛠️ Tech Stack
Machine Learning
- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas
MLOps
- MLflow
- MLflow Model Registry
- Model versioning
- Experiment tracking
Backend
- FastAPI
- Uvicorn
Frontend
- Streamlit
Development
- Jupyter Notebook
- Git
- GitHub
📁 Project Structure
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

The raw C-MAPSS dataset is not included in the repository.

🔄 MLOps Workflow
The project uses MLflow to track and manage the trained model.
Experiment Tracking
The following information is tracked:
- Model parameters
- Sequence length
- RUL cap
- Number of features
- Validation metrics
- Test metrics
- Model artifacts
Model Registry
The final model is registered as:
PredictiveMaintenance_LSTM

Current registered version:
Version 1

The FastAPI application loads the registered model for inference.
⚡ FastAPI
The trained model is exposed through a REST API.
Start the API:
uvicorn api:app --reload

The API will be available at:
http://127.0.0.1:8000

Prediction Endpoint
GET /predict/{engine_id}

Example:
http://127.0.0.1:8000/predict/1

Example response:
{
    "engine_id": 1,
    "observed_cycles": 31,
    "predicted_rul": 124.8,
    "unit": "cycles",
    "model": "PredictiveMaintenance_LSTM",
    "model_version": "1"
}

🎨 Streamlit Dashboard
The project includes an interactive Streamlit interface for selecting an engine and viewing its predicted RUL.
Start Streamlit:
streamlit run app.py

The dashboard allows users to:
- Select an Engine ID
- Request an RUL prediction
- View predicted remaining cycles
- View observed cycles
- See the registered model and version
The Streamlit application communicates with the FastAPI backend.
⚙️ Installation
1. Clone the repository
git clone https://github.com/MousumiBadyakar/Predictive-Maintenance.git

cd Predictive-Maintenance

2. Create a virtual environment
python -m venv .venv

3. Activate the environment
Windows:
.venv\Scripts\activate

4. Install dependencies
pip install -r requirements.txt

📊 MLflow
Start the MLflow tracking server:
mlflow ui

Then open:
http://127.0.0.1:5000

The MLflow interface can be used to inspect:
- Experiments
- Runs
- Parameters
- Metrics
- Model artifacts
- Registered models
▶️ Running the Project
The recommended workflow is:
Terminal 1 — Start MLflow
mlflow ui

Terminal 2 — Start FastAPI
uvicorn api:app --reload

Terminal 3 — Start Streamlit
streamlit run app.py

Then open the Streamlit URL shown in the terminal.
🧪 Inference
A standalone inference script is also provided:
python inference.py

The script loads:
- Registered LSTM model
- Feature scaler
- Model configuration
- FD001 test data
and generates an RUL prediction for a selected engine.
⚠️ Important Note About Test Data
The official C-MAPSS FD001 test set contains partial engine histories.
For the Streamlit/API demonstration, engines with fewer than 50 observed cycles are padded to satisfy the LSTM's required sequence length.
This padding strategy is used only for the interactive inference demonstration.
The official test metrics reported above are calculated using the proper FD001 test evaluation procedure and should be considered the authoritative model performance.
💡 Key Learnings
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
👩‍💻 Author
Mousumi Badyakar
B.Tech — Information Technology