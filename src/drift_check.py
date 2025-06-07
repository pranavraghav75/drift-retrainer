import os
import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import DataDriftTable

ref_df = pd.read_csv("data/processed/train.csv")
curr_df = pd.read_csv("data/processed/latest_inference.csv")

# column mapping for Data Drift Table
column_mapping = ColumnMapping(
    target=None,  
    prediction=None,  
    numerical_features=["age", "balance", "num_transactions", "days_active"],
    categorical_features=None  
)

# write report metrics to HTML file
report = Report(metrics=[DataDriftTable()])
report.run(reference_data=ref_df, current_data=curr_df, column_mapping=column_mapping)
REPORT_PATH = "data_drift_report.html"
report.save_html(REPORT_PATH)

# extract drift metrics and compare for any drift
report_dict = report.as_dict()
custom_threshold = 0.2 
overall_drift = report_dict["metrics"][0]["result"]["share_of_drifted_columns"] > custom_threshold
FLAG_PATH = "trigger_retrain.flag"

if overall_drift:
    print("Data drift detected.")

    with open(FLAG_PATH, "w") as f:
        f.write("1")
    print(f"Retrain flag created at '{FLAG_PATH}'.")
else:
    print("No significant data drift detected.")
    
    # remove old flag if it still exists
    if os.path.exists(FLAG_PATH):
        os.remove(FLAG_PATH)
        print(f"Removed stale drift flag at '{FLAG_PATH}'.")