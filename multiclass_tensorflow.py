import tensorflow as tf 
import numpy as np 

from tensorflow.keras import Input
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras.datasets import mnist 

(X_train, Y_train), (X_test, Y_test)= mnist.load_data()
print(X_train.shape)
print(Y_train.shape)

# Normalize the images 

X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape the Normalized images (flatten)

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.reshape[0], -1)

# Make a model 

model  = Sequential ([
    Input(shape=(784,)),
    Dense(units=128, activation ="relu"),
    Dense(units=64, activation = "relu"),
    Dense(units=10)
])