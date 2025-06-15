#!/usr/bin/env python3
import os
import pandas as pd
import mlflow.pyfunc
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import DataDriftTable, PredictionDriftTable

# ─── Paths ──────────────────────────────────────────────────────────────────────
REF_PATH  = "data/processed/train.csv"
CURR_PATH = "data/processed/latest_inference.csv"
REPORT_HTML = "data_drift_report.html"
FLAG_PATH   = "trigger_retrain.flag"

# ─── Load data ──────────────────────────────────────────────────────────────────
ref_df  = pd.read_csv(REF_PATH)
curr_df = pd.read_csv(CURR_PATH)

# ─── Load model from MLflow registry ───────────────────────────────────────────
# Make sure MLFLOW_TRACKING_URI is set in your env, e.g. http://mlflow:5000
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000")
mlflow.set_tracking_uri(mlflow_uri)
model = mlflow.pyfunc.load_model("models:/ChurnModel/Production")

# ─── Generate predictions for drift metrics ────────────────────────────────────
features = [c for c in ref_df.columns if c != "target"]
ref_df["prediction"]  = model.predict(ref_df[features])
curr_df["prediction"] = model.predict(curr_df[features])

# ─── Define which columns to check ─────────────────────────────────────────────
column_mapping = ColumnMapping(
    target="target",
    prediction="prediction",
    numerical_features=features,
    categorical_features=[]
)

# ─── Build & run the report ────────────────────────────────────────────────────
report = Report(metrics=[DataDriftTable(), PredictionDriftTable()])
report.run(
    reference_data=ref_df,
    current_data=curr_df,
    column_mapping=column_mapping,
)
report.save_html(REPORT_HTML)
print(f"🔍 Drift report saved to {REPORT_HTML}")

# ─── Extract results ────────────────────────────────────────────────────────────
res = report.as_dict()["metrics"]
data_drift = res[0]["result"]["dataset_drift"]
pred_drift = res[1]["result"]["dataset_drift"]
overall_drift = data_drift or pred_drift

# ─── Write or clear the retrain flag ────────────────────────────────────────────
if overall_drift:
    print("🚩 Data or prediction drift detected.")
    with open(FLAG_PATH, "w") as f:
        f.write("1")
    print(f"Flag created at '{FLAG_PATH}'")
else:
    print("✅ No significant data/prediction drift.")
    if os.path.exists(FLAG_PATH):
        os.remove(FLAG_PATH)
        print(f"Removed stale flag at '{FLAG_PATH}'")
