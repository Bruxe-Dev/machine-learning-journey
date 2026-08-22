import numpy as np
import pandas as pd

alpha = 0.01
epochs = 1000

def sigmoid_activation(z):
    return 1 / (1 + np.exp(-z))


def dense(A_in, W, b):

    Z = np.matmul(A_in, W) + b

    A_out = sigmoid_activation(Z)

    return A_out

def forward_prop(X, W1, b1, W2, b2, W3, b3, W4, b4):

    # Hidden Layer 1
    a1 = dense(X, W1, b1)

    # Hidden Layer 2
    a2 = dense(a1, W2, b2)

    # Hidden Layer 3
    a3 = dense(a2, W3, b3)

    # Output Layer
    a4 = dense(a3, W4, b4)

    return a1, a2, a3, a4


def compute_cost(Y, a4):

    m = Y.shape[0]

    cost = -(1 / m) * np.sum(
        Y * np.log(a4) +
        (1 - Y) * np.log(1 - a4)
    )

    return cost

def backward_prop(
    X, Y,
    a1, a2, a3, a4,
    W1, W2, W3, W4
):

    m = Y.shape[0]

    dz4 = a4 - Y

    dW4 = (1/m) * (a3.T @ dz4)

    db4 = (1 / m) * np.sum(
        dz4,
        axis=0
    )


    da3 = dz4 @ W4.T

    dz3 = da3 * a3 * (1 - a3)

    dW3 = (1 / m) * (a2.T @ dz3)

    db3 = (1 / m) * np.sum(
        dz3,
        axis=0
    )

    da2 = dz3 @ W3.T

    dz2 = da2 * a2 * (1 - a2)

    dW2 = (1 / m) * (a1.T @ dz2)

    db2 = (1 / m) * np.sum(
        dz2,
        axis=0
    )

    da1 = dz2 @ W2.T

    dz1 = da1 * a1 * (1 - a1)

    dW1 = (1 / m) * (X.T @ dz1)

    db1 = (1 / m) * np.sum(
        dz1,
        axis=0
    )


    return (
        dW1, db1,
        dW2, db2,
        dW3, db3,
        dW4, db4
    )

def gradient_descent(
    W1, b1,
    W2, b2,
    W3, b3,
    W4, b4,

    dW1, db1,
    dW2, db2,
    dW3, db3,
    dW4, db4,

    alpha
):

    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1

    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3


    W4 = W4 - alpha * dW4
    b4 = b4 - alpha * db4


    return (
        W1, b1,
        W2, b2,
        W3, b3,
        W4, b4
    )

W1 = np.array([
    [1, -3, 5, 2],
    [2,  4, -6, 1]
], dtype=float)

b1 = np.array([
    1, -1, 2, 0
], dtype=float)


W2 = np.array([
    [ 2, -1,  3,  4],
    [ 1,  2, -2,  1],
    [-3,  1,  4, -2],
    [ 2,  3, -1,  1]
], dtype=float)

b2 = np.array([
    1, 0, -1, 2
], dtype=float)


W3 = np.array([
    [ 1,  2, -1],
    [-2,  3,  1],
    [ 4, -1,  2],
    [ 1,  2,  3]
], dtype=float)

b3 = np.array([
    0, 1, -1
], dtype=float)


W4 = np.array([
    [2],
    [-1],
    [3]
], dtype=float)

b4 = np.array([
    0
], dtype=float)


data = pd.read_csv(
    "data/coffee_roast_dataset.csv"
)

X = data[
    ["temperature", "time"]
].values

Y = data[
    "overcooked"
].values.reshape(-1, 1)

X_mean = X.mean(axis=0)

X_std = X.std(axis=0)

X = (X - X_mean) / X_std

for epoch in range(epochs):

    a1, a2, a3, a4 = forward_prop(
        X,
        W1, b1,
        W2, b2,
        W3, b3,
        W4, b4
    )


    cost = compute_cost(
        Y,
        a4
    )

    (
        dW1, db1,
        dW2, db2,
        dW3, db3,
        dW4, db4
    ) = backward_prop(
        X,
        Y,
        a1, a2, a3, a4,
        W1, W2, W3, W4
    )

    (
        W1, b1,
        W2, b2,
        W3, b3,
        W4, b4
    ) = gradient_descent(

        W1, b1,
        W2, b2,
        W3, b3,
        W4, b4,

        dW1, db1,
        dW2, db2,
        dW3, db3,
        dW4, db4,

        alpha
    )

    if epoch % 100 == 0:

        print(
            f"Epoch: {epoch} | Cost: {cost}"
        )

_, _, _, predictions = forward_prop(
    X,
    W1, b1,
    W2, b2,
    W3, b3,
    W4, b4
)

classes = (
    predictions >= 0.5
).astype(int)


print("\nPredictions:")

print(predictions)


print("\nClasses:")

print(classes)