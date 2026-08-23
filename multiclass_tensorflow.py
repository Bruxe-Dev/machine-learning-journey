import tensorflow as tf 
import numpy as np 

from tensorflow.keras import Input
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import Flatten
from tensorflow.keras.datasets import mnist 

(X_train, Y_train), (X_test, Y_test)= mnist.load_data()
print(X_train.shape)
print(Y_train.shape)

# Normalize the images 

X_train = X_train / 255.0
X_test = X_test / 255.0

# Make a model 

model  = Sequential ([
    Flatten(input_shape=(28,28)),
    Dense(units=128, activation ="relu"),
    Dense(units=64, activation = "relu"),
    Dense(units=10)
])

model.summary()

model.compile(
    optimizer = tf.keras.optimizers.Adam(
        learning_rate = 0.001
    ),
    loss = tf.keras.losses.SparseCategorialCrossentropy,
    metrics = ["accuracy"]
)

model.fit (
    X_train,
    Y_train,
    epoch = 20,
    batch_size = 128,
    validation_data = (X_test,Y_test)
)