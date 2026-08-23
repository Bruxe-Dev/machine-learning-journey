import tensorflow as tf 
import pandas as pd 
from tensorflow.keras import Input
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.datasets import mnist 

def relu_activation(z):
    return np.maximum(0,z)

def relu_derivative(z):
    return (z > 0).astype(float)