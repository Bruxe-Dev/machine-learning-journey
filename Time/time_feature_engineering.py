import pandas as pd 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import TimeSeriesSplit

bike_sharing = fetch_openml("Bike_Sharing_Demand",version=2,as_frame=True)
df = bike_sharing.frame

y = df["count"]/df["count"].max()

x = df.drop("count", axis="columns")

print(f"x features: {x.shape}")

x["weather"] = (
    x["weather"]
    .astype(object)
    .replace(to_replace="heavy_rain",value="rain")
    .astype("category")
)

tsc_cv = TimeSeriesSplit(
    n_splits=5,
    gap=48,
    test_size=1000,
    max_train_size=10000
)