"""
Task 5: train a tiny MLP on make_moons with our engine and with PyTorch.
Architecture: 2 -> 8 (ReLU) -> 1 (sigmoid), BCE loss, full-batch gradient descent.
Both implementations start from the SAME initial weights and data, so their
loss curves should match; then we compare wall-clock training time
(median of 5 runs).
"""
import json
import os
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
from sklearn.datasets import make_moons

from engine import Value, zero_grad

N_SAMPLES = 200
EPOCHS = 100
LR = 0.5
HIDDEN = 8
RUNS = 5
SEED = 42


# ------------------------------------------------------------------ data/init
def get_data():
    X, y = make_moons(n_samples=N_SAMPLES, noise=0.1, random_state=SEED)
    return X.astype(np.float64), y.astype(np.float64)


def init_weights():
    """He-style init, shared by both implementations."""
    rng = np.random.default_rng(SEED)
    W1 = rng.normal(0, np.sqrt(2 / 2), size=(HIDDEN, 2))
    b1 = np.zeros(HIDDEN)
    W2 = rng.normal(0, np.sqrt(1 / HIDDEN), size=(1, HIDDEN))
    b2 = np.zeros(1)
    return W1, b1, W2, b2


# ------------------------------------------------------------------ our engine
class EngineMLP:
    def __init__(self, W1, b1, W2, b2):
        self.W1 = [[Value(w) for w in row] for row in W1]
        self.b1 = [Value(b) for b in b1]
        self.W2 = [Value(w) for w in W2[0]]
        self.b2 = Value(b2[0])

    def parameters(self):
        ps = [w for row in self.W1 for w in row] + self.b1 + self.W2 + [self.b2]
        return ps

    def __call__(self, x):
        h = [(sum((wi * xi for wi, xi in zip(row, x)), b)).relu()
             for row, b in zip(self.W1, self.b1)]
        z = sum((w * hi for w, hi in zip(self.W2, h)), self.b2)
        return z.sigmoid()


def bce(p, y):
    # -(y log p + (1-y) log(1-p)); only one term is non-zero for y in {0,1}
    return -(p.log()) if y == 1.0 else -((1 - p).log())


def train_engine(X, y, init):
    model = EngineMLP(*init)
    params = model.parameters()
    losses = []
    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        zero_grad(params)
        loss = sum((bce(model(xi), yi) for xi, yi in zip(X.tolist(), y.tolist())), Value(0.0))
        loss = loss * (1.0 / len(X))
        loss.backward()
        for p in params:
            p.data -= LR * p.grad
        losses.append(loss.data)
    elapsed = time.perf_counter() - t0
    preds = [1.0 if model(xi).data > 0.5 else 0.0 for xi in X.tolist()]
    acc = float(np.mean(np.array(preds) == y))
    return elapsed, losses, acc


# ------------------------------------------------------------------ PyTorch
def train_torch(X, y, init, dtype=torch.float64):
    W1, b1, W2, b2 = init
    model = torch.nn.Sequential(
        torch.nn.Linear(2, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, 1), torch.nn.Sigmoid(),
    ).to(dtype)
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(W1)); model[0].bias.copy_(torch.tensor(b1))
        model[2].weight.copy_(torch.tensor(W2)); model[2].bias.copy_(torch.tensor(b2))
    Xt = torch.tensor(X, dtype=dtype)
    yt = torch.tensor(y, dtype=dtype).unsqueeze(1)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    loss_fn = torch.nn.BCELoss()
    losses = []
    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    elapsed = time.perf_counter() - t0
    with torch.no_grad():
        acc = ((model(Xt) > 0.5).to(dtype) == yt).float().mean().item()
    return elapsed, losses, acc


def main():
    torch.set_num_threads(1)  # fair single-thread comparison on such a tiny model
    X, y = get_data()
    init = init_weights()
    print(f"make_moons: {N_SAMPLES} samples, {EPOCHS} epochs full-batch GD, lr={LR}, "
          f"2->{HIDDEN}->1, ReLU hidden, sigmoid out, BCE")

    eng_times, tor_times = [], []
    for r in range(RUNS):
        te, le, acc_e = train_engine(X, y, init)
        tt, lt, acc_t = train_torch(X, y, init)
        eng_times.append(te); tor_times.append(tt)
        print(f"run {r+1}: engine {te:7.3f}s   torch {tt:7.4f}s")

    max_loss_diff = max(abs(a - b) for a, b in zip(le, lt))
    res = {
        "n_samples": N_SAMPLES, "epochs": EPOCHS, "lr": LR, "runs": RUNS,
        "engine_times_s": eng_times, "torch_times_s": tor_times,
        "engine_median_s": statistics.median(eng_times),
        "torch_median_s": statistics.median(tor_times),
        "speedup_torch_over_engine": statistics.median(eng_times) / statistics.median(tor_times),
        "engine_final_loss": le[-1], "torch_final_loss": lt[-1],
        "engine_accuracy": acc_e, "torch_accuracy": acc_t,
        "max_abs_loss_diff_over_training": max_loss_diff,
        "engine_loss_curve": le, "torch_loss_curve": lt,
    }
    print(f"\nmedian time  engine={res['engine_median_s']:.3f}s  torch={res['torch_median_s']:.4f}s"
          f"  -> PyTorch is {res['speedup_torch_over_engine']:.0f}x faster")
    print(f"loss  epoch1: engine={le[0]:.6f} torch={lt[0]:.6f} | final: engine={le[-1]:.6f} torch={lt[-1]:.6f}")
    print(f"max |loss diff| over all epochs = {max_loss_diff:.2e}")
    print(f"train accuracy  engine={acc_e:.3f}  torch={acc_t:.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_DIR / "moons.json", "w") as fh:
        json.dump(res, fh, indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(le, label="our engine")
        plt.plot(lt, "--", label="PyTorch")
        plt.xlabel("epoch"); plt.ylabel("BCE loss"); plt.title("make_moons training loss")
        plt.legend(); plt.tight_layout(); plt.savefig(RESULTS_DIR / "moons_loss.png", dpi=120)
        print("Saved results/moons.json, results/moons_loss.png")
    except ImportError:
        print("Saved results/moons.json (matplotlib not installed, no plot)")


if __name__ == "__main__":
    main()
