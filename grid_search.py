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