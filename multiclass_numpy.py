import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


# ============================================================
# ACTIVATION FUNCTIONS
# ============================================================

def relu_activation(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def softmax(z):
    # Numerical stability
    z = z - np.max(z, axis=1, keepdims=True)

    exp_z = np.exp(z)

    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# ============================================================
# FORWARD PROPAGATION
# ============================================================

def forward_propagation(X, W1, b1, W2, b2, W3, b3):

    # First hidden layer
    Z1 = X @ W1 + b1
    A1 = relu_activation(Z1)

    # Second hidden layer
    Z2 = A1 @ W2 + b2
    A2 = relu_activation(Z2)

    # Output layer
    Z3 = A2 @ W3 + b3

    return Z1, A1, Z2, A2, Z3


# ============================================================
# LOSS FUNCTION
# ============================================================

def sparse_categorical_crossentropy(probabilities, labels):

    batch_size = probabilities.shape[0]

    # Get probability assigned to the correct class
    correct_class_probabilities = probabilities[
        np.arange(batch_size),
        labels
    ]

    # Prevent log(0)
    correct_class_probabilities = np.clip(
        correct_class_probabilities,
        1e-12,
        1.0
    )

    losses = -np.log(correct_class_probabilities)

    return np.mean(losses)


# ============================================================
# BACKWARD PROPAGATION
# ============================================================

def backward_propagation(
    X,
    Y,
    Z1, A1,
    Z2, A2,
    Z3,
    probabilities,
    W2,
    W3
):

    batch_size = X.shape[0]

    # --------------------------------------------------------
    # Output layer
    # --------------------------------------------------------

    dZ3 = probabilities.copy()

    dZ3[np.arange(batch_size), Y] -= 1

    # Average gradient across batch
    dZ3 /= batch_size

    dW3 = A2.T @ dZ3

    db3 = np.sum(
        dZ3,
        axis=0,
        keepdims=True
    )

    # --------------------------------------------------------
    # Second hidden layer
    # --------------------------------------------------------

    dA2 = dZ3 @ W3.T

    dZ2 = dA2 * relu_derivative(Z2)

    dW2 = A1.T @ dZ2

    db2 = np.sum(
        dZ2,
        axis=0,
        keepdims=True
    )

    # --------------------------------------------------------
    # First hidden layer
    # --------------------------------------------------------

    dA1 = dZ2 @ W2.T

    dZ1 = dA1 * relu_derivative(Z1)

    dW1 = X.T @ dZ1

    db1 = np.sum(
        dZ1,
        axis=0,
        keepdims=True
    )

    return (
        dW1, db1,
        dW2, db2,
        dW3, db3
    )


# ============================================================
# GRADIENT DESCENT
# ============================================================

def gradient_descent(
    W1, b1,
    W2, b2,
    W3, b3,
    dW1, db1,
    dW2, db2,
    dW3, db3,
    learning_rate
):

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3

    return (
        W1, b1,
        W2, b2,
        W3, b3
    )


# ============================================================
# ACCURACY
# ============================================================

def calculate_accuracy(X, Y, W1, b1, W2, b2, W3, b3):

    _, _, _, _, logits = forward_propagation(
        X,
        W1, b1,
        W2, b2,
        W3, b3
    )

    probabilities = softmax(logits)

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    accuracy = np.mean(
        predictions == Y
    )

    return accuracy


# ============================================================
# LOAD MNIST
# ============================================================

print("Loading MNIST dataset...")

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


print("\nOriginal shapes:")
print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)
print("X_test :", X_test.shape)
print("Y_test :", Y_test.shape)


# ============================================================
# NORMALIZE
# ============================================================

X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0


# ============================================================
# FLATTEN IMAGES
# ============================================================

X_train = X_train.reshape(
    X_train.shape[0],
    -1
)

X_test = X_test.reshape(
    X_test.shape[0],
    -1
)


print("\nAfter flattening:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# ============================================================
# INITIALIZE PARAMETERS
# ============================================================

np.random.seed(42)


# Layer 1
W1 = np.random.randn(784, 128) * np.sqrt(2 / 784)
b1 = np.zeros((1, 128))


# Layer 2
W2 = np.random.randn(128, 64) * np.sqrt(2 / 128)
b2 = np.zeros((1, 64))


# Layer 3 / Output
W3 = np.random.randn(64, 10) * np.sqrt(2 / 64)
b3 = np.zeros((1, 10))


# ============================================================
# TRAINING SETTINGS
# ============================================================

learning_rate = 0.01

batch_size = 128

epochs = 20


print("\nTraining settings:")
print("Training examples:", len(X_train))
print("Batch size:", batch_size)
print("Epochs:", epochs)
print("Learning rate:", learning_rate)


# ============================================================
# STORE TRAINING HISTORY
# ============================================================

loss_history = []

accuracy_history = []

test_accuracy_history = []


# ============================================================
# TRAINING LOOP
# ============================================================

print("\n==============================")
print("STARTING TRAINING")
print("==============================\n")


for epoch in range(epochs):

    # --------------------------------------------------------
    # Shuffle dataset
    # --------------------------------------------------------

    indices = np.random.permutation(
        len(X_train)
    )

    X_train = X_train[indices]
    Y_train = Y_train[indices]


    total_loss = 0.0

    num_batches = 0


    # --------------------------------------------------------
    # Mini-batch training
    # --------------------------------------------------------

    for start in range(
        0,
        len(X_train),
        batch_size
    ):

        end = start + batch_size

        X_batch = X_train[start:end]

        Y_batch = Y_train[start:end]


        # ----------------------------------------------------
        # Forward propagation
        # ----------------------------------------------------

        Z1, A1, Z2, A2, Z3 = forward_propagation(
            X_batch,
            W1, b1,
            W2, b2,
            W3, b3
        )


        # ----------------------------------------------------
        # Softmax
        # ----------------------------------------------------

        probabilities = softmax(Z3)


        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = sparse_categorical_crossentropy(
            probabilities,
            Y_batch
        )

        total_loss += loss

        num_batches += 1


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        (
            dW1, db1,
            dW2, db2,
            dW3, db3
        ) = backward_propagation(
            X_batch,
            Y_batch,
            Z1, A1,
            Z2, A2,
            Z3,
            probabilities,
            W2,
            W3
        )


        # ----------------------------------------------------
        # Gradient descent
        # ----------------------------------------------------

        (
            W1, b1,
            W2, b2,
            W3, b3
        ) = gradient_descent(
            W1, b1,
            W2, b2,
            W3, b3,
            dW1, db1,
            dW2, db2,
            dW3, db3,
            learning_rate
        )


    # ========================================================
    # END OF EPOCH
    # ========================================================

    average_loss = (
        total_loss / num_batches
    )


    # Training accuracy
    train_accuracy = calculate_accuracy(
        X_train,
        Y_train,
        W1, b1,
        W2, b2,
        W3, b3
    )


    # Test accuracy
    test_accuracy = calculate_accuracy(
        X_test,
        Y_test,
        W1, b1,
        W2, b2,
        W3, b3
    )


    # Store history
    loss_history.append(
        average_loss
    )

    accuracy_history.append(
        train_accuracy
    )

    test_accuracy_history.append(
        test_accuracy
    )


    # Print progress
    print(
        f"Epoch {epoch + 1:02d}/{epochs} | "
        f"Loss: {average_loss:.4f} | "
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}% | "
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n==============================")
print("TRAINING COMPLETE")
print("==============================")

print(
    f"Final Training Accuracy: "
    f"{accuracy_history[-1] * 100:.2f}%"
)

print(
    f"Final Test Accuracy: "
    f"{test_accuracy_history[-1] * 100:.2f}%"
)

print(
    f"Final Loss: "
    f"{loss_history[-1]:.4f}"
)


# ============================================================
# PLOT LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, epochs + 1),
    loss_history,
    marker="o"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training Loss")

plt.grid(True)

plt.savefig(
    "training_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PLOT ACCURACY
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, epochs + 1),
    accuracy_history,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    range(1, epochs + 1),
    test_accuracy_history,
    marker="o",
    label="Test Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("MNIST Training vs Test Accuracy")

plt.legend()

plt.grid(True)

plt.savefig(
    "training_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()