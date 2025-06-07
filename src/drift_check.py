import os
import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import DataDriftTable

# ─── Step 1: Load reference (training) and current (inference) data ──────────────────

# (1) Reference data is your original training set:
ref_df = pd.read_csv("data/processed/train.csv")

# (2) Current data is a recent batch (e.g., logged inference inputs):
curr_df = pd.read_csv("data/processed/latest_inference.csv")

# ─── Step 2: Define Column Mapping ──────────────────────────────────────────────────

# Define column mapping (optional, but useful for specifying feature/target columns)
column_mapping = ColumnMapping(
    target=None,  # Specify the target column if applicable
    prediction=None,  # Specify the prediction column if applicable
    numerical_features=["age", "balance", "num_transactions", "days_active"],  # Numerical features
    categorical_features=None  # Add categorical features if applicable
)

# ─── Step 3: Create and Run the Data Drift Report ───────────────────────────────────

# Create a Report with the DataDriftTable metric
report = Report(metrics=[DataDriftTable()])
report.run(reference_data=ref_df, current_data=curr_df, column_mapping=column_mapping)

# Save the report as an HTML file
REPORT_PATH = "data_drift_report.html"
report.save_html(REPORT_PATH)
print(f"Data drift report saved to {REPORT_PATH}")

# ─── Step 4: Trigger Retraining Based on Drift ──────────────────────────────────────

# Extract drift results
report_dict = report.as_dict()
overall_drift = report_dict["metrics"][0]["result"]["dataset_drift"]
print(f"Overall drift detected: {overall_drift}")

# Trigger retraining if drift is detected
FLAG_PATH = "trigger_retrain.flag"

if overall_drift:
    print("⚠️  Data drift detected.")
    # Create the retrain flag file
    with open(FLAG_PATH, "w") as f:
        f.write("1")
    print(f"→ Retrain flag created at '{FLAG_PATH}'.")
else:
    print("✅ No significant data drift detected.")
    # Remove any old flag if exists
    if os.path.exists(FLAG_PATH):
        os.remove(FLAG_PATH)
        print(f"→ Removed stale drift flag at '{FLAG_PATH}'.")