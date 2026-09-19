import tensorflow as tf
import pandas as pd 
from tensorflow.keras.datasets import fashion_mnist 
from tensorflow.keras import Input,Sequential
from tensorflow.keras.layers import Dense, Flatten

(X_train,Y_train),(X_test,Y_test) = fashion_mnist.load_data()
print(f"Shape: {X_train.shape}")
print(f"Shape: {Y_train.shape}")
print(f"Shape: {X_test.shape}")
print(f"Shape: {Y_test.shape}")

x_train = X_train[:5000]
Y_train = Y_train[:5000]

#x_test = X_test[:2000]

x_train_norm = x_train.astype("float32")/ 255.0
x_test_norm = x_test.astype("float32")/ 255.0

model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(units=128,activation="relu",kernel_regularizer=L2(0.01)),
    Dense(units=64, activation = "relu", kernel_regularizer=L2(0.01)),
    Dense(units=10)
])

model.summary()

model.compile(
    optimizer = tf.keras.optimizers.Adam(
        learning_rate = 1e-3
    ),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ["accuracy"]
)