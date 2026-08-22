import numpy as np
import pandas as pd

np.random.seed(42)

data = pd.read_csv("data/coffe_dataset.csv")

X0 = data[data["overcooked"] == 0][["temperature", "time"]].values
X1 = data[data["overcooked"] == 1][["temperature", "time"]].values

n = 250

mean0, std0 = X0.mean(axis=0), X0.std(axis=0)
mean1, std1 = X1.mean(axis=0), X1.std(axis=0)

gen0 = np.random.normal(loc=mean0, scale=std0 * 0.5, size=(n, 2))
gen0[:, 0] = np.clip(gen0[:, 0], 100, 250)
gen0[:, 1] = np.clip(gen0[:, 1], 3, 18)
gen0 = np.round(gen0, 1)

gen1 = np.random.normal(loc=mean1, scale=std1 * 0.5, size=(n, 2))
gen1[:, 0] = np.clip(gen1[:, 0], 200, 450)
gen1[:, 1] = np.clip(gen1[:, 1], 14, 40)
gen1 = np.round(gen1, 1)

new_data = pd.DataFrame(
    np.vstack([
        np.column_stack([gen0, np.zeros(n, dtype=int)]),
        np.column_stack([gen1, np.ones(n, dtype=int)])
    ]),
    columns=["temperature", "time", "overcooked"]
)

combined = pd.concat([data, new_data], ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

combined.to_csv("data/coffe_dataset.csv", index=False)
print(f"Original: {len(data)} rows | Generated: {len(new_data)} rows | Total: {len(combined)} rows")
