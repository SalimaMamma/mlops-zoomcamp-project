# 🚀 Crypto Time Series MLOps Platform

A robust, production-grade MLOps pipeline for real-time **cryptocurrency price variation forecasting**, featuring automated data ingestion, model training, drift monitoring, and low-latency serving.

---

## 📌 Problem Statement

### ✅ What This Solves
- **Real-time crypto price *variation* prediction** for Bitcoin
- Advanced **time series modeling** with rich feature engineering: lags, rolling windows, technical indicators.
- Fully automated ML lifecycle: **ingest → train → monitor → retrain → serve**.
- Scalable API for **algorithmic trading**, portfolio management, and market trend detection.

### 💼 Business Value
- Enable quantitative & algorithmic trading strategies.
- Automate portfolio hedging & risk management.
- Provide live signals for investment decision engines.
- Gain market insights from fine-grained trend modeling.

---

## 🧩 Key Technical Highlights

- **Prediction Target**: Predicts *price variation* (delta) over the forecast horizon instead of raw price.
- **Forecast Horizon**: Fixed to **1 step ahead** (1-hour delta) — effectively an hourly streaming problem.
- **Timezone Awareness**
- **Rich Feature Engineering**: Extensive lags, rolling stats, and technical indicators to capture trend, momentum, and volatility.
- The model predicts **price variations** (returns), not absolute prices — this focuses on direction and magnitude, which matter most for trading decisions.  

---

## ⚙️ Tech Stack

| Layer | Tools & Frameworks |
|----------------------|-------------------------------|
| **Workflow Orchestration** | Apache Airflow |
| **Model Registry & Tracking** | MLflow |
| **API Serving** | FastAPI |
| **Containerization** | Docker, Docker Compose |
| **Storage** | LocalStack (S3 emulation), PostgreSQL |
| **ML & Data** | Pandas, NumPy, scikit-learn, LightGBM,TA-Lib, CCXT |
| **Testing** | Pytest, Moto (for mocking AWS) |
| **Monitoring** | Evidently |
| **Planned to improve the project ** | Terraform (real cloud infra), Kafka (streaming), Grafana + Prometheus (monitoring) |

---

## 🏗️ Architecture

```plaintext
┌────────────── Client ───────────────┐
                │  HTTP API
┌────────────── FastAPI ──────────────┐
│ Prediction Endpoints │ Model Loader │ Health & Auth │
                │
┌────────────── Storage ──────────────┐
│ LocalStack (S3) │ PostgreSQL │ MLflow │
                │
┌────────────── Airflow ──────────────┐
│ Ingestion DAG │ Training DAG │ Drift Monitoring DAG │



---

## 📋 Key DAGs Explained

### 1. Model Training Pipeline

- **Schedule:** Triggered manually / by drift alerts.
- **Purpose:** Trains and validates regression models for price forecasting.
- **Process:**  
  - Loads historical features and targets from S3 storage.  
  - Trains machine learning models (e.g., XGBoost, Random Forest).  
  - Evaluates model performance using metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score.  
  - Saves model artifacts and evaluation reports to artifact storage.  
  - Registers new model versions for deployment.

### 2. Hourly prediction pipeline 

- **Schedule:** Runs every hour.
- **Purpose:** Fetches data using API
- **Process:**  
  - Compute technical features (returns, rolling means, volatility, etc.).  
  - Save processed features as Parquet files on an S3-compatible storage backend for downstream use.
  - Compute target values next hour''s price changeand save them alongside features.


### 3. Drift Monitoring Pipeline

- **Schedule:** Runs daily ( or weekly) at a configurable time.
- **Purpose:** Detects data or concept drift by monitoring model prediction quality.
- **Process:**  
  - Aggregates hourly predictions and true target values over a sliding 7-day window.  
  - Computes MAE per hour and averages over the window.  
  - Compares average MAE to a configurable drift threshold.  
  - Generates detailed drift reports using the Evidently library. If Evidently is not available, produces a fallback HTML report with core metrics.  
  - Automatically triggers the Model Training Pipeline if drift is detected.

---


## ⚙️ Configuration

- Centralized settings via a YAML file including:  
  - AWS credentials and S3 bucket info  
  - Drift detection thresholds  
  - Cryptocurrency symbols monitored  
  - LocalStack endpoint for local dev/testing
- Airflow Variables and Connections must be set up accordingly in the deployment environment.

---


## 🚀 Quick Start Guide

### Prerequisites

- Docker & Docker Compose  
- Python 3.9+  


### Setup & Run

```bash
git clone <repo-url>
cd crypto-mlops-platform

cp .env.example .env
# Update .env with your configs (API keys, DB, etc.)

docker-compose -f docker-compose.airflow.yml up --build
Go to http://localhost:8081 to access Airflow UI
Trigger the Model Training Pipeline manually to start the process
Access mlflow UI at http://localhost:5000 to see model versions and metrics

docker-compose -f docker-compose.api.yml up --build

Access the FastAPI at http://localhost:8000

You can now use the client_.py file :
Example : python src/client_.py --date 2025-03-28 --hour 9

Or call the API to make predictions:

curl -X GET "http://localhost:8000/predict?date=2025-07-28&hour=15"

