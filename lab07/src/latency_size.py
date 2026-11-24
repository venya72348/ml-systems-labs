"""Lab 7, task 3: latency and on-disk size of every variant, plus a sparse-kernel benchmark.

1. For baseline + every pruned variant: latency at batch 1 and batch 256 (warm-up, median),
   dense vs effective MACs, and size as dense .pt, zipped .pt, COO, CSR(int32), zipped CSR.
2. Micro-benchmark of the layer that holds 90% of the parameters (fc1, 6272 x 144):
   dense F.linear vs torch.sparse CSR / COO matmul at each sparsity, to show when (if ever)
   sparse kernels beat dense BLAS on this CPU.

    python lab07/src/latency_size.py
Outputs: results/latency_size.csv, results/sparse_matmul.csv, results/latency_size.png, results/sparse_matmul.png
"""
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

from common import (CKPT, MODELS_DIR, RES, disk_sizes, evaluate, load_cfg, load_data, load_model, macs,
                    measure_latency, n_nonzero, n_params)


def variants(cfg):
    yield "baseline", "baseline", 0.0, MODELS_DIR / "baseline.pt"
    for mode in ["global", "layer"]:
        for s in cfg["pruning"]["sparsities"]:
            if s > 0:
                yield f"unstructured_{mode}_{s}", f"unstructured_{mode}", s, CKPT / f"unstructured_{mode}_{s}.pt"
    for r in cfg["pruning"]["structured_ratios"]:
        yield f"structured_{r}", "structured", r, CKPT / f"structured_{r}.pt"


def bench(fn, runs, warmup=10):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


@torch.inference_mode()
def sparse_matmul_benchmark(cfg):
    rows = []
    for s in [0.0] + [s for s in cfg["pruning"]["sparsities"] if s > 0] + [0.99]:
        if s in (0.0, 0.99):   # 0: baseline weights; 0.99: extra point, baseline masked to 99% in fc1
            w = load_model(MODELS_DIR / "baseline.pt").fc1.weight.detach().cpu().clone()
            if s:
                thr = torch.kthvalue(w.abs().flatten(), int(s * w.numel())).values
                w[w.abs() <= thr] = 0
        else:
            w = load_model(CKPT / f"unstructured_layer_{s}.pt").fc1.weight.detach().cpu().clone()
        w_csr, w_coo = w.to_sparse_csr(), w.to_sparse()
        for bs in [1, 256]:
            x = torch.randn(bs, w.shape[1])
            xt = x.t().contiguous()
            runs = 300 if bs == 1 else 100
            rows.append({
                "fc1_sparsity": float((w == 0).float().mean()), "batch": bs,
                "dense_ms": bench(lambda: F.linear(x, w), runs),
                "csr_ms": bench(lambda: torch.sparse.mm(w_csr, xt), runs),
                "coo_ms": bench(lambda: torch.sparse.mm(w_coo, xt), runs),
            })
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df["csr_speedup_vs_dense"] = df["dense_ms"] / df["csr_ms"]
    return df


def main():
    cfg = load_cfg()
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg["latency"]["threads"])   # all timing here is on the CPU
    data = load_data(cfg)
    rows = []
    for name, method, target, path in variants(cfg):
        m = load_model(path)
        acc, _ = evaluate(m, *data["test"])
        row = {"variant": name, "method": method, "target": target, "params": n_params(m),
               "nonzero_params": n_nonzero(m), "test_acc": acc, **macs(m), **measure_latency(m, cfg),
               "file_size_kb": path.stat().st_size / 1024, **disk_sizes(m)}
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "latency_size.csv", index=False)

    sm = sparse_matmul_benchmark(cfg)
    sm.to_csv(RES / "sparse_matmul.csv", index=False)
    plot(df, sm, cfg)


def plot(df, sm, cfg):
    B = cfg["latency"]["big_batch"]
    base = df[df["variant"] == "baseline"].iloc[0]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for method, st, c in [("unstructured_global", "o-", "C0"), ("unstructured_layer", "s-", "C1"),
                          ("structured", "^-", "C2")]:
        d = pd.concat([df[df["variant"] == "baseline"], df[df["method"] == method]])
        x = 100 * (1 - d["nonzero_params"] / base["params"])
        ax[0].plot(x, d["latency_b1_ms"], st, color=c, label=method)
        ax[1].plot(x, d["latency_bbig_ms"], st, color=c, label=method)
    ax[0].set_title("Latency, batch 1 (median ms)")
    ax[1].set_title(f"Latency, batch {B} (median ms)")
    for a in ax[:2]:
        a.set_xlabel("weights removed / zero (%)"); a.set_ylim(bottom=0); a.grid(alpha=0.3); a.legend(fontsize=8)
    d = df[df["method"].isin(["baseline", "unstructured_global"])].copy()
    x = 100 * (1 - d["nonzero_params"] / base["params"])
    for col, lab in [("size_dense_kb", "dense .pt"), ("size_dense_zip_kb", "dense .pt + zip"),
                     ("size_coo_kb", "COO (int64 idx)"), ("size_csr_kb", "CSR (int32 idx)"),
                     ("size_csr_zip_kb", "CSR + zip")]:
        ax[2].plot(x, d[col], "o-", label=lab)
    ds = df[df["method"] == "structured"]
    ax[2].plot(100 * (1 - ds["params"] / base["params"]), ds["size_dense_kb"], "^--", color="k",
               label="structured, dense .pt")
    ax[2].set_title("On-disk size, global unstructured (KB)")
    ax[2].set_xlabel("weights removed / zero (%)"); ax[2].grid(alpha=0.3); ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "latency_size.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, bs in zip(ax, [1, 256]):
        d = sm[sm["batch"] == bs]
        a.plot(d["fc1_sparsity"] * 100, d["dense_ms"], "o-", label="dense F.linear (MKL)")
        a.plot(d["fc1_sparsity"] * 100, d["csr_ms"], "s-", label="torch.sparse.mm CSR")
        a.plot(d["fc1_sparsity"] * 100, d["coo_ms"], "^-", label="torch.sparse.mm COO")
        a.set_yscale("log"); a.set_xlabel("fc1 sparsity (%)"); a.set_ylabel("median ms (log)")
        a.set_title(f"fc1 (6272x144) matmul, batch {bs}"); a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "sparse_matmul.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
