import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import Input
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

data = pd.read_csv("data/coffee_roast_dataset.csv")

print("The read data")
print(data)

X = data[["temperature","time"]].values

Y = data["overcooked"].values
Y = Y.reshape(-1,1)

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

X = (X - X_mean) / X_std

model = Sequential([
    Input(shape=(2,)),

    Dense(25, activation="relu"),

    Dense(15, activation="relu"),

    Dense(5, activation="relu"),

    Dense(1, activation="sigmoid")
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.SGD(
        learning_rate=0.01
    ),

    loss="binary_crossentropy",

    metrics=["accuracy"]
)

history = model.fit(
    X,
    Y,
    epochs=1000,
    verbose=1
)

predictions = model.predict(X)

classes = (predictions >= 0.5).astype(int)


print("\nPredictions Made:")
print(predictions)

print("\nClasses:")
print(classes)

# Peformance plot
plt.plot(history.history["loss"])

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Training Loss")
plt.savefig("plots-binary-classification/training_loss.png")
plt.show()

# Accuracyu plot
plt.plot(history.history["accuracy"])

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title("Training Accuracy")
plt.savefig("plots-binary-classification/training_accuracy.png")
plt.show()