"""Lab 7, task 1: train the oversized (~1M params) baseline CNN on Fashion-MNIST and record
accuracy, parameter count, on-disk size, latency (batch 1 and batch 256), MACs and energy.

    python lab07/src/baseline.py
Outputs: models/baseline.pt, results/baseline.json, results/baseline_history.csv
"""
import pandas as pd

from common import (DEVICE, MODELS_DIR, RES, Net, disk_sizes, energy_meter, evaluate, hardware_info,
                    load_cfg, load_data, macs, measure_latency, n_params, save_json, save_model,
                    set_seed, thop_macs, train)


def main():
    cfg = load_cfg()
    set_seed(cfg["seed"])
    data = load_data(cfg)
    t = cfg["teacher"]
    model = Net(t["widths"]).to(DEVICE)
    print(f"baseline widths={t['widths']} params={n_params(model):,} device={DEVICE}")

    with energy_meter(cfg, "train_baseline") as energy:
        history = train(model, data, t["epochs"], t["lr"], t["batch_size"], cfg["seed"], tag="baseline")
    test_acc, test_loss = evaluate(model, *data["test"])
    path = MODELS_DIR / "baseline.pt"
    save_model(model, path)

    res = {
        "model": "baseline", "widths": t["widths"], "params": n_params(model),
        "test_acc": test_acc, "test_loss": test_loss, "val_acc": history[-1]["val_acc"],
        "file_size_kb": path.stat().st_size / 1024, **disk_sizes(model),
        **measure_latency(model, cfg), **macs(model), "thop_macs": thop_macs(model),
        "train_energy": energy, "epochs": t["epochs"], "hardware": hardware_info(),
    }
    save_json(res, "baseline.json")
    pd.DataFrame(history).to_csv(RES / "baseline_history.csv", index=False)
    print({k: v for k, v in res.items() if k not in ("hardware",)})


if __name__ == "__main__":
    main()
