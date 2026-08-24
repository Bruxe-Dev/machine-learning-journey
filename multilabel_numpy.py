import numpy as np 

def sigmoid(z):
    return 1/(1+np.exp(-z))

def binaryCrossentropy(y,p):
    epsilon = 1e-15
    p = np.clip(p, epsilon, 1-epsilon)

    return  -(y *np.log(p) + (1-y) * np.log(1-p))

X = np.array([
    [2, 1],
    [1, 2],
    [8, 9],
    [9, 8]
])

Y = np.array([
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0]
])
 
W = np.array([
    [0.5, 0.2],
    [0.3, 0.8]
])

b = np.array([0.1,0.2])

Z = X @ W + b 

p = sigmoid(Z)

print("Z:")
print(Z)

print("P:")
print(p)

loss = binaryCrossentropy(Y,p)

print("Loss: ")
print(loss)

cost = np.mean(loss)
print("cost: ")
print(cost)
#predictions = (p > 0.5).astype(int)