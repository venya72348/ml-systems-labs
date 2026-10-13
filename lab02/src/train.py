"""Train the network for every (learning rate, width) pair and plot loss curves."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import SEED, load_data, set_seeds
from nn import accuracy, backward, cross_entropy, forward, init_params, sgd_step

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

LEARNING_RATES = [0.001, 0.01, 0.1]
WIDTHS = [16, 64]
EPOCHS = 50
BATCH = 32
STYLES = {0.001: ":", 0.01: "--", 0.1: "-"}  # 0.001 and 0.01 overlap, so use line styles


def train(width, lr, X_train, Y_train, X_test, Y_test):
    rng = np.random.default_rng(SEED)
    params = init_params(width, rng)
    history = []
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(X_train))          # shuffle every epoch
        batch_losses = []
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            p, cache = forward(params, X_train[idx])
            batch_losses.append(cross_entropy(p, Y_train[idx]))
            sgd_step(params, backward(params, cache, Y_train[idx]), lr)
        train_loss = float(np.mean(batch_losses))
        test_loss = cross_entropy(forward(params, X_test)[0], Y_test)
        history.append({"width": width, "lr": lr, "epoch": epoch,
                        "train_loss": train_loss, "test_loss": test_loss})
        if epoch % 10 == 0 or epoch == 1:
            print(f"W={width:<3} lr={lr:<6} epoch {epoch:>2}: "
                  f"train_loss={train_loss:.4f} test_loss={test_loss:.4f}")
    return params, history


def main():
    set_seeds()
    X_train, X_test, Y_train, Y_test = load_data()
    all_history, summary = [], []

    for width in WIDTHS:
        for lr in LEARNING_RATES:
            params, history = train(width, lr, X_train, Y_train, X_test, Y_test)
            all_history += history
            summary.append({
                "width": width, "lr": lr,
                "final_train_loss": round(history[-1]["train_loss"], 4),
                "final_test_loss": round(history[-1]["test_loss"], 4),
                "test_accuracy": round(accuracy(params, X_test, Y_test), 4),
            })

    hist_df = pd.DataFrame(all_history)
    sum_df = pd.DataFrame(summary)
    hist_df.to_csv(RESULTS / "training_log.csv", index=False)
    sum_df.to_csv(RESULTS / "final_accuracy.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, width in zip(axes, WIDTHS):
        for lr in LEARNING_RATES:
            h = hist_df[(hist_df.width == width) & (hist_df.lr == lr)]
            ax.plot(h.epoch, h.train_loss, label=f"lr={lr}", linestyle=STYLES[lr], linewidth=2)
        ax.set_title(f"Hidden width W = {width}")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Training cross-entropy loss")
    fig.suptitle("Loss curves (batch 32, 50 epochs, SGD)")
    fig.tight_layout()
    fig.savefig(RESULTS / "loss_curves.png", dpi=150)

    print("\n=== Final results ===")
    print(sum_df.to_string(index=False))
    print(f"\nSaved loss_curves.png, training_log.csv, final_accuracy.csv to {RESULTS}")


if __name__ == "__main__":
    main()