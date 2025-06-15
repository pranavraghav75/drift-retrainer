from flask import Flask, request, jsonify
import pandas as pd
import mlflow.pyfunc
import logging
import os

app = Flask(__name__)

MLFLOW_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://host.docker.internal:5000"
)
mlflow.set_tracking_uri(MLFLOW_URI)

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

# load latest model from MLflow registry
model_name = "ChurnModel"
model_stage = "None"  # change to Staging/Production as needed
# model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage}")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        logging.info(f"Loading model: {model_name}, stage: {model_stage}")
        model = mlflow.pyfunc.load_model(model_uri="models/trained/churn")
        data = request.get_json()
        df = pd.DataFrame([data])
        prediction = model.predict(df)
        model_version = getattr(model.metadata, "run_id", "unknown")
        return jsonify({
            "prediction": int(prediction[0]),
            "model_version": model_version
        })
    except Exception as e:
        logging.error(e, exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

