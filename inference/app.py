# inference_api/app.py
from flask import Flask, request, jsonify
import pandas as pd
import mlflow.pyfunc

app = Flask(__name__)
mlflow.set_tracking_uri("http://localhost:5000")

# Load latest model from MLflow registry
model_name = "ChurnModel"
model_stage = "None"  # use 'Staging' or 'Production' later if you use stages
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage}")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        prediction = model.predict(df)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
