import numpy as np 

X = np.array([
    [1, 2],
    [2, 4],
    [3, 6],
    [4, 8],
    [5, 10]
], dtype=float)

y = np.array([
    3,
    6,
    9,
    12,
    15
], dtype=float)


w, b = lasso_regression(
    X,
    y,
    learning_rate=0.01,
    lambda_=0.1,
    epochs=2000
)

print("Weights:", w)
print("Bias:", b)

def compute_cost(X, y, w, b, lambda_):
    m = len(y)

    predictions = X @ w + b
    errors = predictions - y

    # MSE
    mse = (1 / (2 * m)) * np.sum(errors ** 2)

    # L1 regularization
    l1_penalty = (lambda_ / m) * np.sum(np.abs(w))

    return mse + l1_penalty


def compute_gradient(X, y, w, b, lambda_):
    m = len(y)

    # Predictions
    predictions = X @ w + b

    # Error
    errors = predictions - y

    # MSE gradient
    dw = (1 / m) * (X.T @ errors)

    # L1 gradient
    dw += (lambda_ / m) * np.sign(w)

    # Bias gradient
    db = (1 / m) * np.sum(errors)

    return dw, db


def lasso_regression(X, y, learning_rate, lambda_, epochs):

    m, n = X.shape

    # Initialize parameters
    w = np.zeros(n)
    b = 0.0

    for epoch in range(epochs):

        # Calculate gradients
        dw, db = compute_gradient(
            X, y, w, b, lambda_
        )

        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db

        # Print cost occasionally
        if epoch % 100 == 0:
            cost = compute_cost(
                X, y, w, b, lambda_
            )

            print(f"Epoch {epoch}: Cost = {cost:.4f}")

    return w, b