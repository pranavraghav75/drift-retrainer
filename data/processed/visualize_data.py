import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv("data/processed/train.csv")
inference_df = pd.read_csv("data/processed/latest_inference.csv")

# plot distributions
features = ["age", "balance", "num_transactions", "days_active"]
for feature in features:
    plt.figure(figsize=(8, 4))
    plt.hist(train_df[feature], bins=30, alpha=0.5, label="Train", density=True)
    plt.hist(inference_df[feature], bins=30, alpha=0.5, label="Inference", density=True)
    plt.title(f"Distribution of {feature}")
    plt.legend()
    plt.show()