# Drift-Retrainer

An end-to-end MLOps pipeline that trains a customer churn model, monitors it for data and prediction drift, and automatically retrains it when drift is detected — all wired together with Airflow, MLflow, Evidently, and a Streamlit UI.

---

## What it does

1. **Train** an XGBoost churn classifier on tabular customer data
2. **Serve** predictions via a Flask REST API
3. **Detect drift** hourly using Evidently AI (comparing training distribution vs. live inference data)
4. **Auto-retrain** when drift is flagged, registering the new model back into MLflow
5. **Visualize** the full workflow interactively through a Streamlit dashboard

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI (:8501)                 │
│  Upload CSV / Generate Synthetic Data → Train → Drift   │
└──────────────────────────┬──────────────────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │         Flask API (:8000)       │
          │    POST /predict   GET /health  │
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │       MLflow Tracking (:5000)   │
          │  Experiments · Registry · Runs  │
          │  Backend: mlflow.db (SQLite)    │
          │  Artifacts: ./mlruns/           │
          └────────────────┬────────────────┘
                           │
     ┌─────────────────────▼───────────────────────┐
     │               Apache Airflow                │
     │                                             │
     │  drift_detection  (hourly)                  │
     │    └─ run drift_check.py                    │
     │         └─ Evidently drift report           │
     │         └─ writes trigger_retrain.flag      │
     │                                             │
     │  retrain_pipeline (hourly)                  │
     │    └─ FileSensor: trigger_retrain.flag      │
     │    └─ run retrain_pipeline.py               │
     │         └─ calls train.py                   │
     │         └─ registers new ChurnModel version │
     └─────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| ML model | XGBoost (binary classifier) |
| Experiment tracking & registry | MLflow |
| Drift detection | Evidently AI |
| Orchestration | Apache Airflow |
| Inference API | Flask + Gunicorn |
| Dashboard | Streamlit |
| Containerization | Docker + Docker Compose |
| Data / metrics | pandas, scikit-learn, NumPy |

---

## Project Structure

```
drift-retrainer/
├── dags/
│   ├── drift_detection_dag.py     # Airflow DAG: hourly drift check
│   └── retraining_dag.py          # Airflow DAG: flag-triggered retrain
├── data/
│   └── processed/
│       ├── train.csv              # Reference training data
│       ├── latest_inference.csv   # Current inference data (drift target)
│       ├── data_generation.py     # Synthetic data generator (with drift)
│       └── visualize_data.py
├── docker/
│   └── Dockerfile.api             # Flask API image
├── inference/
│   └── app.py                     # Flask inference server
├── src/
│   ├── train.py                   # XGBoost training + MLflow logging
│   ├── drift_check.py             # Evidently drift report + flag logic
│   ├── retrain_pipeline.py        # Reads flag, calls train.py, clears flag
│   └── utils/
│       ├── data_loader.py
│       └── mlflow_utils.py
├── streamlit_app.py               # Interactive ML playground UI
├── docker-compose.yml             # API + MLflow + Streamlit services
├── requirements.txt
└── mlflow.db                      # SQLite MLflow backend store
```

---

## Data Model

The churn dataset has four numerical features and one binary label:

| Column | Description |
|---|---|
| `age` | Customer age |
| `balance` | Account balance |
| `num_transactions` | Transaction count |
| `days_active` | Days since last activity |
| `target` | Churn label (1 = churned, 0 = retained) |

The synthetic data generator in `data/processed/data_generation.py` produces:
- **Training data** (`train.csv`): 1,000 samples drawn from a normal distribution centered at realistic values
- **Inference data** (`latest_inference.csv`): 100 samples with the `balance` feature intentionally shifted higher (1,500–2,500 range) to simulate real-world distribution drift

---

## Drift Detection Logic

`src/drift_check.py` uses Evidently to compare the reference training set against the latest inference batch:

1. Loads the `ChurnModel/Production` model from the MLflow registry
2. Generates predictions on both reference and current data
3. Runs `DataDriftTable` and `PredictionDriftTable` reports
4. Saves an HTML report to `data_drift_report.html`
5. If either feature data or prediction distribution has drifted → writes `trigger_retrain.flag`
6. If no drift → removes any stale flag

---

## Airflow DAGs

### `drift_detection` — runs hourly
Executes `src/drift_check.py`. Sets the retrain flag if drift is found.

### `retrain_pipeline` — runs hourly
Uses a `FileSensor` to wait for `trigger_retrain.flag`. When the flag appears, runs `src/retrain_pipeline.py`, which:
- Calls `src/train.py` to fit a new XGBoost model
- Logs metrics (accuracy, F1) and parameters to MLflow
- Registers the new model as a new version of `ChurnModel`
- Removes the flag file

---

## MLflow

MLflow is used for:
- **Experiment tracking**: every training run logs accuracy, F1, and all XGBoost hyperparameters under the `Churn-Detection-Drift` experiment
- **Model registry**: trained models are registered as `ChurnModel`, with versions automatically incremented on each retrain
- **Model serving**: the drift check and inference API both load models directly from the registry by name/stage

Backend store: `mlflow.db` (SQLite, local)  
Artifact store: `./mlruns/` (local filesystem)

---

## API

The Flask API (`inference/app.py`) runs on port `8000`:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check, returns `200 OK` |
| `/predict` | POST | Returns churn prediction for a single record |

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "balance": 950.0, "num_transactions": 4, "days_active": 120}'
```

**Example response:**
```json
{
  "prediction": 0,
  "model_version": "<run_id>"
}
```

---

## Streamlit UI

The dashboard (`streamlit_app.py`) provides a single-page workflow:

1. **Upload CSV** — upload your own customer data or use the synthetic generator
2. **Generate Synthetic Data** — produces `latest_inference.csv` with built-in drift
3. **Train XGBoost Model** — kicks off `src/train.py` and logs to MLflow
4. **Live Prediction** — form-based inference via the Flask API (upload mode only)
5. **Drift Detection & Auto-Retrain** — runs the Evidently drift check, shows the HTML report inline, and exposes a retrain button when drift is detected

---

## Running Locally (Docker Compose)

```bash
# Build and start all services
docker-compose up --build

# Services
# MLflow UI:    http://localhost:5000
# Inference API: http://localhost:8000
# Streamlit UI:  http://localhost:8501
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://host.docker.internal:5000` | MLflow server URL |

---

## Running Without Docker

```bash
# 1. Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start MLflow tracking server
mlflow server --backend-store-uri sqlite:///mlflow.db \
              --default-artifact-root ./mlruns \
              --host 0.0.0.0 --port 5000

# 4. Generate synthetic data (optional)
python data/processed/data_generation.py

# 5. Train the model
python src/train.py

# 6. Run drift check
python src/drift_check.py

# 7. Retrain if flag was set
python src/retrain_pipeline.py

# 8. Launch the Streamlit UI
streamlit run streamlit_app.py

# 9. Launch the inference API (separate terminal)
python inference/app.py
```

### Airflow setup (optional, for scheduled automation)

```bash
pip install apache-airflow
export AIRFLOW_HOME=$(pwd)/.airflow
airflow db init
airflow webserver -p 8080 &
airflow scheduler &
# Copy dags/ into your $AIRFLOW_HOME/dags directory
```

---

## Dependencies

```
pandas · xgboost · scikit-learn · mlflow · flask
evidently==0.3.1 · sqlalchemy · numpy==1.26.4
psycopg2-binary · streamlit · requests · gunicorn
```
