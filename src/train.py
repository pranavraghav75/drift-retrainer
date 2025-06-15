# src/train.py

import os
import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ─── Configure MLflow Tracking URI ───────────────────────────────────────────────
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000")
mlflow.set_tracking_uri(TRACKING_URI)

print("Starting training...")

# ─── Load and split data ────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/train.csv")  # assume this is already clean
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")
print("Starting MLflow run...")

# ─── Train and log model ────────────────────────────────────────────────────────
mlflow.set_experiment("Churn-Detection-Drift")

with mlflow.start_run():
    # handle class imbalance
    ratio_of_0_to_1 = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        eval_metric="logloss",
        scale_pos_weight=ratio_of_0_to_1,
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds)

    # Log metrics & params
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_params(model.get_params())

    # Log model artifact
    mlflow.xgboost.log_model(model, artifact_path="model")

    # Register model
    run_id = mlflow.active_run().info.run_id
    result = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name="ChurnModel"
    )
    print(f"Model registered as ChurnModel version {result.version}")

# ─── Print reports locally ──────────────────────────────────────────────────────
train_preds = model.predict(X_train)
print("Training Classification Report:")
print(classification_report(y_train, train_preds))

val_preds = model.predict(X_val)
print("Validation Classification Report:")
print(classification_report(y_val, val_preds))