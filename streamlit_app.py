# streamlit_app.py
import streamlit as st
import pandas as pd
import subprocess, os

st.title("ML Drift‐&‐Retrain Playground")

uploaded = st.file_uploader("Upload raw CSV", type=["csv"])
if uploaded is not None:
    raw_path = os.path.join("data/raw", uploaded.name)
    with open(raw_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success(f"Saved to {raw_path}")

    st.write("Cleaning data…")
    subprocess.run(["python", "src/clean.py", raw_path, "data/processed/cleaned.csv"], check=True)
    df = pd.read_csv("data/processed/cleaned.csv")
    st.dataframe(df.head())

    if st.button("Train XGBoost"):
        st.write("Training model…")
        subprocess.run(["python", "src/train.py",
                        "--in", "data/processed/cleaned.csv",
                        "--out", "models/trained/model"], check=True)
        st.success("Training complete! View results in MLflow UI.")

st.header("Live Prediction")
age = st.number_input("Age", 0, 120, 30)
balance = st.number_input("Balance", 0.0, 1e6, 1000.0)
txns = st.number_input("Num Transactions", 0, 1000, 5)
days = st.number_input("Days Active", 0, 365, 180)

if st.button("Predict"):
    payload = {"age": age, "balance": balance,
               "num_transactions": txns, "days_active": days}
    r = st.experimental_get_query_params
    import requests
    resp = requests.post("http://api:8000/predict", json=payload)
    if resp.ok:
        st.json(resp.json())
    else:
        st.error(f"Error: {resp.text}")