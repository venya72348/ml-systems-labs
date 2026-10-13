"""Sanity checks: output shape, softmax rows, initial loss, numerical gradient check."""
import math
from pathlib import Path

import numpy as np

from data import SEED, load_data, set_seeds
from nn import backward, cross_entropy, forward, init_params, sgd_step

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
EPS = 1e-5
lines = []


def log(msg=""):
    print(msg)
    lines.append(msg)


set_seeds()
X_train, X_test, Y_train, Y_test = load_data()
log(f"Train {X_train.shape}, test {X_test.shape}")

for width in (16, 64):
    rng = np.random.default_rng(SEED)
    params = init_params(width, rng)
    log(f"\n===== Width W = {width} =====")
    log("Shapes: " + ", ".join(f"{k}{v.shape}" for k, v in params.items()))

    # --- forward checks ---
    p, _ = forward(params, X_train[:32])
    log(f"Output shape: {p.shape}  (expected (32, 10))")
    max_dev = np.abs(p.sum(axis=1) - 1).max()
    log(f"Max |row sum - 1|: {max_dev:.2e}  -> {'PASS' if max_dev < 1e-6 else 'FAIL'}")

    # --- loss sanity check ---
    p_all, _ = forward(params, X_train)
    loss0 = cross_entropy(p_all, Y_train)
    log(f"Untrained loss: {loss0:.4f}  (ln(10) = {math.log(10):.4f})")

    # --- gradient check ---
    # Train 50 epochs first (lr=0.1): at N(0, 0.01) init many gradients are ~1e-7,
    # so float64 round-off in the numerical slope dominates the relative error.
    for _ in range(50):
        for i in range(0, len(X_train), 32):
            _, c = forward(params, X_train[i:i + 32])
            sgd_step(params, backward(params, c, Y_train[i:i + 32]), 0.1)

    Xb, Yb = X_train[:100], Y_train[:100]
    _, cache = forward(params, Xb)
    grads = backward(params, cache, Yb)

    pick = np.random.default_rng(SEED)
    passed = 0
    log("Gradient check (10 random weights, eps = 1e-5):")
    for _ in range(10):
        k = pick.choice(["W1", "W2", "W3"])
        idx = tuple(int(pick.integers(0, s)) for s in params[k].shape)
        old = params[k][idx]
        params[k][idx] = old + EPS
        lp = cross_entropy(forward(params, Xb)[0], Yb)
        params[k][idx] = old - EPS
        lm = cross_entropy(forward(params, Xb)[0], Yb)
        params[k][idx] = old
        num = (lp - lm) / (2 * EPS)
        ana = grads[k][idx]
        denom = abs(ana) + abs(num)
        rel = 0.0 if denom < 1e-12 else abs(ana - num) / denom
        ok = rel < 1e-6
        passed += ok
        log(f"  {k}{idx}: analytic={ana:+.6e} numeric={num:+.6e} rel_err={rel:.2e} {'OK' if ok else 'X'}")
    log(f"Passed {passed}/10 -> {'PASS' if passed >= 9 else 'FAIL'} (need >= 9)")

(RESULTS / "checks.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nSaved to {RESULTS / 'checks.txt'}")