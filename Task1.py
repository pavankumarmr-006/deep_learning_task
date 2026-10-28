
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Activation functions
# -----------------------------

def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    # Clip values to avoid exponential overflow
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# -----------------------------
# 2. Loss functions
# -----------------------------

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def cross_entropy(y_true, y_pred):
    # Binary cross-entropy
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )


# -----------------------------
# 3. XOR dataset
# -----------------------------

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float64)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=np.float64)


# -----------------------------
# 4. Initialize 2-4-1 network
# -----------------------------

np.random.seed(42)

W1 = np.random.randn(2, 4) * np.sqrt(2 / 2)
b1 = np.zeros((1, 4))

W2 = np.random.randn(4, 1) * np.sqrt(2 / 4)
b2 = np.zeros((1, 1))


# -----------------------------
# 5. Forward pass
# -----------------------------

def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = relu(z1)

    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)

    cache = (X, z1, a1, a2)
    return a2, cache


# -----------------------------
# 6. Training with backprop
# -----------------------------

def train(lr=0.5, epochs=5000):
    W1 = np.random.randn(2, 4) * np.sqrt(2 / 2)
    b1 = np.zeros((1, 4))
    W2 = np.random.randn(4, 1) * np.sqrt(2 / 4)
    b2 = np.zeros((1, 1))

    losses = []
    diverged = False

    for epoch in range(epochs):
        # Forward pass
        pred, cache = forward(X, W1, b1, W2, b2)
        X_cache, z1, a1, a2 = cache

        loss = cross_entropy(y, pred)

        if not np.isfinite(loss):
            diverged = True
            break

        losses.append(loss)

        # BCE + sigmoid: dL/dz2 = (pred - y) / N
        dz2 = (a2 - y) / len(X)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Backpropagate through ReLU
        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X_cache.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Gradient descent update
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        # Detect runaway parameters as well
        if not all(np.all(np.isfinite(p))
                   for p in (W1, b1, W2, b2)):
            diverged = True
            break

    pred, _ = forward(X, W1, b1, W2, b2)

    return pred, losses, (W1, b1, W2, b2), diverged


# -----------------------------
# 7. Train at required LR
# -----------------------------

pred, losses, params, diverged = train(lr=0.5, epochs=5000)

print("Softmax:", np.round(softmax(
    np.array([2.0, 1.0, 0.1])
), 3))

print("Final BCE:", losses[-1] if losses else "Diverged")
print("XOR probabilities:", np.round(pred.flatten(), 3))
print("XOR classes:", (pred >= 0.5).astype(int).flatten())


# -----------------------------
# 8. Plot required loss curve
# -----------------------------

plt.figure(figsize=(8, 5))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Binary cross-entropy")
plt.title("XOR training loss (learning rate = 0.5)")
plt.grid(True)
plt.tight_layout()
plt.savefig("xor_loss.png", dpi=150)
plt.show()


# -----------------------------
# 9. Compare learning rates
# -----------------------------

plt.figure(figsize=(9, 5))

for lr in [0.01, 0.5, 5.0]:
    pred_lr, loss_lr, _, diverged_lr = train(
        lr=lr, epochs=5000
    )

    if loss_lr:
        plt.plot(
            loss_lr,
            label=f"lr={lr}" + (" (stopped)" if diverged_lr else "")
        )

    print(
        f"lr={lr}:",
        "diverged" if diverged_lr else "completed",
        "final loss =", loss_lr[-1] if loss_lr else "N/A",
        "predictions =", np.round(pred_lr.flatten(), 3)
    )

plt.xlabel("Epoch")
plt.ylabel("Binary cross-entropy")
plt.title("Effect of learning rate on XOR training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("learning_rate_comparison.png", dpi=150)
plt.show()
