import pandas as pd 
from sklearn.datasets import fetch_openml

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