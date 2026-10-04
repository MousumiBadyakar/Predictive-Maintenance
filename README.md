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