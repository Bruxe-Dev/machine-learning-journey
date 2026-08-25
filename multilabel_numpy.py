import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(y, p):
    epsilon = 1e-15

    # Prevent log(0)
    p = np.clip(p, epsilon, 1 - epsilon)

    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))

    return loss


X = np.array([
    [2, 1],
    [1, 2],
    [8, 9],
    [9, 8]
], dtype=float)


# Two labels:
#
# Label 1 → Dog
# Label 2 → Cat

Y = np.array([
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0]
], dtype=float)

W = np.array([
    [0.5, 0.2],
    [0.3, 0.8]
], dtype=float)

b = np.array([0.1, 0.2], dtype=float)

learning_rate = 0.01
epochs = 1000

m = X.shape[0]

for epoch in range(epochs):

    Z = np.matmul(X, W) + b

    P = sigmoid(Z)


    loss = binary_cross_entropy(Y, P)

    cost = np.mean(loss)

    error = P - Y

    dW = (1 / m) * np.matmul(X.T, error)

    db = (1 / m) * np.sum(error, axis=0)


    W -= learning_rate * dW

    b -= learning_rate * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Cost: {cost:.6f}")

Z = X @ W + b

P = sigmoid(Z)

predictions = (P >= 0.5).astype(int)

print("\n" + "=" * 50)
print("FINAL RESULTS")
print("=" * 50)
print("\nFinal Weights:")
print(W)

print("\nFinal Bias:")
print(b)

print("\nFinal Z:")
print(Z)

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