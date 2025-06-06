import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import mlflow
import mlflow.xgboost
import xgboost as xgb
import os

# === Load and Prepare Data ===
df = pd.read_csv("data/processed/train.csv")  # assume this has already been cleaned
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# === Start MLflow Run ===
mlflow.set_tracking_uri("http://localhost:5000")  # MLflow server URI
mlflow.set_experiment("Churn-Detection-Drift")    # Creates or switches to this experiment

with mlflow.start_run():
    ratio_of_0_to_1 = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        eval_metric="logloss",
        scale_pos_weight=ratio_of_0_to_1
    )
    
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds)

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Log parameters
    mlflow.log_params(model.get_params())

    # Log model
    mlflow.xgboost.log_model(model, artifact_path="model")

    # Register model
    result = mlflow.register_model(
        model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
        name="ChurnModel"
    )

    print(f"Model registered with name: ChurnModel and version: {result.version}")

# Evaluate on training data
train_preds = model.predict(X_train)
print("Training Classification Report:")
print(classification_report(y_train, train_preds))

# Evaluate on validation data
val_preds = model.predict(X_val)
print("Validation Classification Report:")
print(classification_report(y_val, val_preds))

# Test the model with the specific input
test_input = pd.DataFrame([{
    "age": 40,
    "balance": 837.33,
    "num_transactions": 3,
    "days_active": 334
}])
test_prediction = model.predict(test_input)
print("Test Input Prediction:", test_prediction[0])
