import numpy as np
import pandas as pd

np.random.seed(42)

n = 250

# ==================================================
# [1, 0] -> CAT
# ==================================================

cat_weight = np.random.normal(
    loc=4.5,
    scale=0.7,
    size=n
)

cat_size = np.random.normal(
    loc=35,
    scale=5,
    size=n
)

cat_weight = np.clip(cat_weight, 2.5, 6.5)
cat_size = np.clip(cat_size, 25, 45)

cat = np.column_stack([
    cat_weight,
    cat_size,
    np.ones(n, dtype=int),
    np.zeros(n, dtype=int)
])


# ==================================================
# [0, 1] -> DOG
# ==================================================

dog_weight = np.random.normal(
    loc=15,
    scale=2.5,
    size=n
)

dog_size = np.random.normal(
    loc=65,
    scale=8,
    size=n
)

dog_weight = np.clip(dog_weight, 9, 22)
dog_size = np.clip(dog_size, 45, 80)

dog = np.column_stack([
    dog_weight,
    dog_size,
    np.zeros(n, dtype=int),
    np.ones(n, dtype=int)
])


# ==================================================
# [1, 1] -> CAT + DOG
# ==================================================

both_weight = np.random.normal(
    loc=10,
    scale=1.5,
    size=n
)

both_size = np.random.normal(
    loc=50,
    scale=6,
    size=n
)

both_weight = np.clip(both_weight, 7, 14)
both_size = np.clip(both_size, 38, 65)

both = np.column_stack([
    both_weight,
    both_size,
    np.ones(n, dtype=int),
    np.ones(n, dtype=int)
])


# ==================================================
# [0, 0] -> NEITHER
# ==================================================

neither_weight = np.random.normal(
    loc=2,
    scale=0.5,
    size=n
)

neither_size = np.random.normal(
    loc=15,
    scale=3,
    size=n
)

neither_weight = np.clip(neither_weight, 0.5, 3)
neither_size = np.clip(neither_size, 8, 22)

neither = np.column_stack([
    neither_weight,
    neither_size,
    np.zeros(n, dtype=int),
    np.zeros(n, dtype=int)
])


# ==================================================
# Combine dataset
# ==================================================

dataset = np.vstack([
    cat,
    dog,
    both,
    neither
])


# ==================================================
# Create DataFrame
# ==================================================

data = pd.DataFrame(
    dataset,
    columns=[
        "weight",
        "size",
        "cat",
        "dog"
    ]
)


# Round feature values
data["weight"] = data["weight"].round(2)
data["size"] = data["size"].round(2)


# Shuffle dataset
data = data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ==================================================
# Save CSV
# ==================================================

data.to_csv(
    "data/animals_multilabel.csv",
    index=False
)


# ==================================================
# Information
# ==================================================

print(f"Generated {len(data)} samples")

print("\nLabel combinations:")
print(
    data[["cat", "dog"]].value_counts()
)

print("\nFirst 10 samples:")
print(data.head(10))