# Machine Learning Journey

My hands-on journey learning Machine Learning and Artificial Intelligence — starting from
fundamental ML concepts and building up to neural networks implemented **from scratch with
Python and NumPy**.

This is a learning repository, not a polished library. The goal is to understand how these
algorithms actually work internally instead of relying entirely on high-level frameworks.

---

## What This Repository Is

A record of my progression through ML/AI fundamentals, where every important component —
forward propagation, loss functions, backpropagation, gradient descent — is written by hand
before (or instead of) calling a framework to do it for me.

The current milestone is a **multiclass neural network built only with NumPy**, trained on the
MNIST handwritten digit dataset.

---

## Learning Progression

1. **Binary classification with Keras/TensorFlow** (`logistic.py`) — first contact with
   `Sequential` models, `Dense` layers, training loops, and evaluation on a coffee-roasting
   dataset.
2. **Logits-based output layer** (`logistic_accurate.py`) — moving from `sigmoid` activations
   in the final layer to a linear output with `BinaryCrossentropy(from_logits=True)`, which is
   the more numerically stable, recommended design.
3. **Peeling back the framework** (`logistic_numpy.py`) — re-implementing the same kind of
   network in pure NumPy: dense layers with sigmoid activation, forward propagation, binary
   cross-entropy cost, backpropagation, and gradient descent updates.
4. **Multiclass classification from scratch** (`multiclass_numpy.py`) — the current project:
   a fully vectorized 3-layer ReLU + softmax network trained on MNIST with mini-batch
   gradient descent.

---

## Machine Learning Concepts Studied

Concepts I have studied so far on this path:

- Linear regression and logistic regression
- Cost / loss functions (squared error, binary cross-entropy)
- Gradient descent
- Feature scaling and z-score normalization *(used in the code for the coffee dataset)*
- Polynomial regression and regularization *(studied conceptually — not yet implemented in
  this repo)*
- Sigmoid activation and decision boundaries
- Vectorization

## Neural Network Concepts Studied

- Neurons, layers, and Dense layers
- Forward propagation
- Activation functions: sigmoid, ReLU; also studied Leaky ReLU and Swish
- Softmax for multiclass outputs
- Sparse categorical cross-entropy
- Backward propagation, derivatives, and gradient calculation
- Weight and bias updates via gradient descent
- Mini-batch gradient descent

---

## Current Project: MNIST From Scratch (NumPy Only)

`multiclass_numpy.py` implements, manually, everything needed to train a multiclass
classifier on MNIST:

| Component | Implementation |
|---|---|
| Forward propagation | Manual matrix multiplications through 3 layers |
| Hidden activations | ReLU + its derivative |
| Output activation | Softmax (with max-subtraction for numerical stability) |
| Loss | Sparse categorical cross-entropy (clipped to avoid `log(0)`) |
| Backpropagation | Hand-derived gradients for all weights and biases |
| Optimization | Mini-batch gradient descent (batch size 128, lr 0.01, 20 epochs) |
| Initialization | He initialization: `randn * sqrt(2 / fan_in)` |
| Training hygiene | Per-epoch shuffling, loss history, train/test accuracy tracking |

### Architecture

```
Input (784 pixels)
        │
   Dense(128) + ReLU          ← He init
        │
   Dense(64)  + ReLU          ← He init
        │
   Dense(10)  → logits
        │
     Softmax                   ← probabilities over digits 0–9
```

The output layer produces raw logits that are passed through softmax; the loss is computed
from those probabilities against integer class labels.

### Results

Training runs track average epoch loss plus training and test accuracy each epoch. On this
setup the network learns the task well within 20 epochs — running the script reproduces the
loss/accuracy curves.

---

## Why Implement Algorithms From Scratch?

Frameworks like TensorFlow/Keras are excellent, but they hide the mechanics. By writing
forward propagation, backpropagation, softmax, cross-entropy, and gradient descent myself,
I can actually answer questions like:

- Why does the softmax need max-subtraction for numerical stability?
- Where exactly does the ReLU derivative appear in the backward pass?
- How do gradients flow from the output layer back to the first hidden layer?
- What do weight shapes really mean in a vectorized implementation?

---

## Technologies

Only what the code actually uses:

- **Python**
- **NumPy** — all from-scratch implementations
- **Pandas** — loading the coffee-roasting CSV dataset
- **Matplotlib** — loss and accuracy plots
- **TensorFlow/Keras** — MNIST dataset loader and the earlier framework-stage models
- **Git & GitHub** — versioning this journey

---

## Repository Structure

```
.
├── logistic_accurate.py    # Keras model with logits + BinaryCrossentropy(from_logits=True)
├── logistic.py             # First Keras binary classifier + performance plots
├── logistic_numpy.py       # Binary classification NN implemented from scratch in NumPy
├── multiclass_numpy.py     # MNIST multiclass NN implemented from scratch in NumPy
├── generate_data.py        # Synthetic data augmentation for the coffee dataset
├── data/
│   └── coffee_roast_dataset.csv
└── requirements.txt
```

> Note: `plots/` and generated `.png` files are ignored — they are regenerable outputs of the
> scripts.

## Running

```bash
pip install -r requirements.txt

# From-scratch MNIST network (downloads MNIST via keras.datasets on first run)
python multiclass_numpy.py

# Earlier experiments (require data/coffee_roast_dataset.csv)
python logistic_numpy.py
python logistic.py
python logistic_accurate.py
```

Run scripts from the repository root, as they use relative paths.

---

## What's Next

- Implementing regularization and polynomial features in code (moving them from theory to practice)
- Trying other optimizers (e.g., Adam) and comparing against plain gradient descent
- Experimenting with learning-rate schedules, batch sizes, and deeper architectures
- Convolutional neural networks for image data
- Eventually porting what I've learned into PyTorch while keeping my NumPy implementations as reference

---

*This repository grows as I learn — it intentionally shows the messy, incremental path of
understanding ML/AI from the inside out.*
