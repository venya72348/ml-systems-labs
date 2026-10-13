"""Two-hidden-layer neural network implemented from scratch with NumPy.

Architecture: 64 -> W -> ReLU -> W -> ReLU -> 10 -> softmax
"""
import numpy as np


def init_params(width, rng, n_in=64, n_out=10, std=0.01):
    """Weights ~ N(0, 0.01) (std = 0.01), biases = 0."""
    sizes = [n_in, width, width, n_out]
    params = {}
    for i in range(3):
        params[f"W{i + 1}"] = rng.normal(0.0, std, size=(sizes[i], sizes[i + 1]))
        params[f"b{i + 1}"] = np.zeros(sizes[i + 1])
    return params


def relu(z):
    return np.maximum(0.0, z)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(params, X):
    """Forward pass layer by layer. Returns probabilities (batch, 10) and a cache."""
    z1 = X @ params["W1"] + params["b1"]
    a1 = relu(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    a2 = relu(z2)
    z3 = a2 @ params["W3"] + params["b3"]
    p = softmax(z3)
    cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "p": p}
    return p, cache


def cross_entropy(p, Y):
    """Mean cross-entropy between predicted probabilities p and one-hot labels Y."""
    return float(-np.mean(np.sum(Y * np.log(np.clip(p, 1e-12, 1.0)), axis=1)))


def backward(params, cache, Y):
    """Analytic gradients via the chain rule.

    softmax + cross-entropy:  dL/dz3 = (p - Y) / batch
    linear layer z = a @ W + b:  dL/dW = a.T @ dz,  dL/db = sum(dz),  dL/da = dz @ W.T
    ReLU:  dL/dz = dL/da * (z > 0)
    """
    n = Y.shape[0]
    X, z1, a1, z2, a2, p = (cache[k] for k in ("X", "z1", "a1", "z2", "a2", "p"))

    dz3 = (p - Y) / n
    grads = {"W3": a2.T @ dz3, "b3": dz3.sum(axis=0)}

    da2 = dz3 @ params["W3"].T
    dz2 = da2 * (z2 > 0)
    grads["W2"] = a1.T @ dz2
    grads["b2"] = dz2.sum(axis=0)

    da1 = dz2 @ params["W2"].T
    dz1 = da1 * (z1 > 0)
    grads["W1"] = X.T @ dz1
    grads["b1"] = dz1.sum(axis=0)
    return grads


def sgd_step(params, grads, lr):
    for k in params:
        params[k] -= lr * grads[k]


def accuracy(params, X, Y):
    p, _ = forward(params, X)
    return float(np.mean(p.argmax(axis=1) == Y.argmax(axis=1)))