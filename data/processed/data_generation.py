import pandas as pd
import numpy as np
import os

# original small dataset values
data = {
    "age": [36, 60, 28, 58, 18, 40, 64, 57, 49, 41],
    "balance": [1136.98, 1115.77, 1052.3, 863.82, 1006.86, 837.33, 1147.41, 901.34, 820.55, 984.87],
    "num_transactions": [5, 6, 5, 5, 4, 3, 4, 6, 6, 4],
    "days_active": [186, 202, 152, 120, 96, 334, 343, 55, 134, 316],
    "target": [0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
}

df_original = pd.DataFrame(data)

# generate a larger dataset by expanding the original data
n_samples = 1000
df_large = pd.DataFrame({
    "age": np.random.randint(18, 70, size=n_samples),
    "balance": np.round(np.random.normal(loc=1000, scale=150, size=n_samples), 2),
    "num_transactions": np.random.randint(1, 10, size=n_samples),
    "days_active": np.random.randint(30, 365, size=n_samples),
})

df_large["target"] = np.where(
    (df_large["balance"] > 1000) & (df_large["num_transactions"] < 5), 1, 0
)

output_path = "data/processed/train.csv"
if os.path.exists('train.csv'):
    os.remove('train.csv')

df_large.to_csv(output_path, index=False)

# print("first 5 rows:")
# print(df_large.head())

#  ===== data drift generation =====
n_samples = 100
drift_data = pd.DataFrame({
    "age": np.random.randint(20, 70, n_samples),
    "balance": np.random.uniform(1500, 2500, n_samples),  # shifted higher
    "num_transactions": np.random.randint(1, 10, n_samples),
    "days_active": np.random.randint(30, 365, n_samples)
})

drift_data["target"] = np.where(
    (drift_data["balance"] > 2000) & (drift_data["num_transactions"] < 5), 1, 0
)

output_path = "data/processed/latest_inference.csv"
if os.path.exists('latest_inference.csv'):
    os.remove('latest_inference.csv')

drift_data.to_csv(output_path, index=False)