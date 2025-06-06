from flask import Flask, request, jsonify
import pandas as pd
import mlflow.pyfunc
import logging

app = Flask(__name__)
mlflow.set_tracking_uri("http://localhost:5000")

# load latest model from MLflow registry
model_name = "ChurnModel"
model_stage = "None"  # change to Staging/Production as needed
# model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage}")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        logging.info(f"Loading model: {model_name}, stage: {model_stage}")
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage}")

        data = request.get_json()
        df = pd.DataFrame([data])
        prediction = model.predict(df)
        
        # dynamically get latest model version
        model_version = model.metadata.run_id if hasattr(model, "metadata") and hasattr(model.metadata, "run_id") else "unknown"

        return jsonify({
            "prediction": int(prediction[0]),
            "model_version": model_version
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
