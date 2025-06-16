import os
import subprocess
import pandas as pd
import streamlit as st
import requests
import mlflow
from mlflow.tracking import MlflowClient

if "data_source" not in st.session_state:
    st.session_state.data_source = None 

if "raw_path" not in st.session_state:
    st.session_state.raw_path = None

if "processed_path" not in st.session_state:
    st.session_state.processed_path = None

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000")
mlflow.set_tracking_uri(TRACKING_URI)

# ─── Sanity check: do we have a registered ChurnModel? ───────────────────
client = MlflowClient()
try:
    versions = client.get_latest_versions("ChurnModel")
    if not versions:
        raise ValueError("no versions")
    model_available = True
except Exception:
    model_available = False
    st.warning("🚧 No registered ChurnModel found—please Train your model first.")

##### Page header #####
st.set_page_config(page_title="ML Drift & Retrain Playground", layout="wide")
st.title("ML Drift-&-Retrain Playground")

##### Data Input Tabs #####
tab1, tab2 = st.tabs(["Upload CSV", "Generate Synthetic Data"])

with tab1:
    uploaded = st.file_uploader("Upload your raw CSV file", type=["csv"])
    if uploaded is not None:
        raw_path = os.path.join("data/raw", uploaded.name)
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        with open(raw_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Raw data saved to `{raw_path}`")
        st.session_state.data_source = "upload"
        st.session_state.raw_path = raw_path

        st.write("Cleaning uploaded data…")
        cleaned_path = "data/processed/train.csv"
        subprocess.run(
            ["python", "src/clean.py", raw_path, cleaned_path],
            check=True,
        )
        st.session_state.processed_path = cleaned_path

        st.success("Data cleaned, below is a preview of what your data looks like!")

        df = pd.read_csv(cleaned_path)
        st.dataframe(df.head(), use_container_width=True)

with tab2:
    with st.expander("Click to Generate Synthetic Data"):
        st.write(
            "This will run your data-generation script and\n"
            "produce `data/processed/latest_inference.csv`."
        )
        if st.button("Generate Synthetic Data Now"):
            subprocess.run(
                ["python", "data/processed/data_generation.py"],
                check=True
            )
            st.success("Synthetic data generated, below is a preview of what your generated data looks like!")
            gen_path = "data/processed/latest_inference.csv"
            st.session_state.data_source = "generate"
            st.session_state.processed_path = gen_path

            df = pd.read_csv(gen_path)
            st.dataframe(df.head(), use_container_width=True)

##### Training Section #####
st.markdown("---")
st.header("Train XGBoost Model")
if st.session_state.processed_path:
    train_input = "data/processed/train.csv"
    st.write(f"Using data: `{train_input}`")
    if st.button("Train Model"):
        st.write("Training… this may take a minute.")
        try:
            result = subprocess.run(
                ["python", "src/train.py"],
                check=True,
                capture_output=True,
                text=True
            )
            st.success("Training complete! Check MLflow for details.")
        except subprocess.CalledProcessError as e:
            st.error("Training failed. See error below:")
            st.code(e.stderr or e.stdout or str(e), language="bash")
else:
    st.info("Upload or generate data above to enable training.")

##### Predict Section (only if user uploaded) #####
if model_available and st.session_state.data_source == "upload":
    st.markdown("---")
    st.header("Live Prediction")
    age = st.number_input("Age", 0, 120, 30)
    balance = st.number_input("Balance", 0.0, 1e6, 1000.0)
    txns = st.number_input("Num Transactions", 0, 1000, 5)
    days = st.number_input("Days Active", 0, 365, 180)

    if st.button("Predict"):
        payload = {
            "age": age,
            "balance": balance,
            "num_transactions": txns,
            "days_active": days,
        }
        resp = requests.post("http://api:8000/predict", json=payload)
        if resp.ok:
            st.json(resp.json())
        else:
            st.error(f"Error: {resp.text}")

elif st.session_state.data_source == "generate":
    st.info("Synthetic data generated → skip live prediction. Train your model above.")

##### Drift Detection & Auto-Retrain #####
if model_available and st.session_state.processed_path:
    st.markdown("---")
    st.header("Drift Detection & Auto-Retrain")

    if st.button("Check Data Drift"):
        st.info("Running drift check…")

        cmd = [
            "python", "src/drift_check.py",
            "data/processed/train.csv",
            st.session_state.processed_path,
            "trigger_retrain.flag"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        drift_line = [l for l in result.stdout.splitlines() if l.startswith("DRIFT=")][0]
        drift = drift_line.split("=")[1] == "True"
        if drift:
            st.error("Data drift detected!")
        else:
            st.success("No significant drift.")

        with open("data_drift_report.html", "r") as f:
            html = f.read()
        st.components.v1.html(html, height=400, scrolling=True)

    # retraining
    if os.path.exists("trigger_retrain.flag"):
        if st.button("Retrain Model"):
            st.info("Retraining…")
            sub = subprocess.run(
                ["python", "src/retrain_pipeline.py"], 
                capture_output=True, text=True
            )
            if sub.returncode == 0:
                st.success("Retrain complete! New model in MLflow.")
                os.remove("trigger_retrain.flag")
            else:
                st.error("Retrain failed:")
                st.code(sub.stderr or sub.stdout, language="bash")
elif not model_available:
    st.info("You need to train a model before you can check drift or retrain.")
