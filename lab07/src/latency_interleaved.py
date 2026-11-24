"""Lab 7, task 3 (control): interleaved CPU latency of the key variants.

latency_size.py measures variants one after another, so slow drift of the laptop (turbo
boost, temperature, background load) can masquerade as a speed difference. Here the models
are measured in ROUNDS: in every round each model gets the same number of timed runs, in a
rotated order, and we report the median over rounds. Drift then hits all models equally.

    python lab07/src/latency_interleaved.py
Output: results/latency_interleaved.csv
"""
import statistics

import pandas as pd
import torch

from common import CKPT, MODELS_DIR, RES, latency_ms, load_cfg, load_model, n_nonzero

MODELS = [("baseline (dense)", MODELS_DIR / "baseline.pt"),
          ("global-pruned 90% (masked, dense kernels)", CKPT / "unstructured_global_0.9.pt"),
          ("global-pruned 95% (masked, dense kernels)", CKPT / "unstructured_global_0.95.pt"),
          ("structured-pruned 50% (physically smaller)", CKPT / "structured_0.5.pt"),
          ("student (100k)", MODELS_DIR / "student_plain.pt")]
ROUNDS = 12


def main():
    cfg = load_cfg()
    torch.set_num_threads(cfg["latency"]["threads"])
    models = [(n, load_model(p).cpu().eval()) for n, p in MODELS]
    per = {n: {"b1": [], "b256": []} for n, _ in models}
    for r in range(ROUNDS):
        order = models[r % len(models):] + models[:r % len(models)]   # rotate start each round
        for n, m in order:
            per[n]["b1"].append(latency_ms(m, 1, runs=60, warmup=10))
            per[n]["b256"].append(latency_ms(m, 256, runs=4, warmup=1))
        print(f"round {r + 1}/{ROUNDS}", flush=True)
    rows = []
    for n, m in models:
        b1, b256 = per[n]["b1"], per[n]["b256"]
        rows.append({"model": n, "nonzero_params": n_nonzero(m),
                     "latency_b1_ms_median": statistics.median(b1),
                     "latency_b1_ms_min": min(b1), "latency_b1_ms_max": max(b1),
                     "latency_b256_ms_median": statistics.median(b256),
                     "latency_b256_ms_min": min(b256), "latency_b256_ms_max": max(b256)})
    df = pd.DataFrame(rows)
    base = df.iloc[0]
    df["speedup_b1"] = base["latency_b1_ms_median"] / df["latency_b1_ms_median"]
    df["speedup_b256"] = base["latency_b256_ms_median"] / df["latency_b256_ms_median"]
    df.to_csv(RES / "latency_interleaved.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
