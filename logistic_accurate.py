import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Dense

data = pd.read_csv("data/coffee_roast_dataset.csv")
print(data.head(10))

X = data[["temperature","time"]].values

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

X = (X - X_mean) / X_std

Y = data["overcooked"].values.reshape(-1,1)

model = Sequential([
    Input(shape=(2,)),
    Dense(units=25, activation= "relu"),
    Dense(units=15, activation= "relu"),
    Dense(units=10, activation= "relu"),
    Dense(units=1, activation="linear")
])

model.summary()

model.compile (
    optimizer = tf.keras.optimizers.SGD(
        learning_rate = 0.01
    ),
    loss=BinaryCrossentropy(from_logits = True),
    metrics = ["accuracy"],
)

model.fit(
    X,
    Y,
    epochs = 1000,
    verbose = 1
)

logits = model(X)

f_x = tf.nn.sigmoid(logits)

print("Predictions: ")
print(f_x)

for layer in model.layers:
    print(layer.get_weights())