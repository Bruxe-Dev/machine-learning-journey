import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy

data = pd.read_csv("data/animals_multilabel.csv")
print(data.head(10))

X = data[["weight","size"]].values

X_mean = X.mean(axis=0)
x_std = X.std(axis=0)

X = (X - X_mean)/x_std

Y = data[["cat","dog"]].values

model = Sequential([
    Input(shape=(2,)),
    Dense(units=50, activation="relu"),
    Dense(units=25, activation="relu"),
    Dense(units=10, activation="relu"),
    Dense(units=2, name="output_layer")
])

model.summary()

model.compile(
    loss = BinaryCrossentropy(from_logits = True),
    optimizer = tf.keras.optimizers.SGD(
        learning_rate = 0.001
    ),
    metrics=["accuracy"]
)

model.fit(
    X,
    Y,
    epochs=1500,
    verbose=1
)

logits = model(X)

P = tf.sigmoid(logits)
predictions = (P >=0.5).astype(int)

print("\n" + "=" * 50)
print("FINAL RESULTS")
print("=" * 50)

output_layer = model.get_layer('output_layer')
W, b = output_layer.get_weights()

print("\nFinal Weights:")
print(W)

print("\nFinal Bias:")
print(b)

print("\nFinal Z:")
print(logits)

print("\nFinal Probabilities:")
print(P)

print("\nPredictions:")
print(predictions)

print("\nActual Labels:")
print(Y)

label_accuracy = np.mean(predictions == Y)

print(f"\nLabel Accuracy: {label_accuracy * 100:.2f}%")

sample_accuracy = np.mean(
    np.all(predictions == Y, axis=1)
)

print(f"Exact Match Accuracy: {sample_accuracy * 100:.2f}%")