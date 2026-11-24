"""Lab 7, task 5: summary by the Ch. 9 efficiency pillars (data, model, compute, energy).

Rows: baseline, best global-pruned, best structured-pruned, plain student, distilled student.
"Best" pruned = the most compressed variant whose test accuracy is within 1 pp of the baseline.
Energy: codecarbon around (a) each training stage (recorded by the earlier scripts) and
(b) inference over the 10k test set, measured here.

    python lab07/src/summary.py
Outputs: results/summary_pillars.csv, results/inference_energy.csv, results/summary_tradeoff.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from common import (CKPT, MODELS_DIR, RES, batches, energy_meter, load_cfg, load_data, load_model,
                    read_json, thop_macs)

N_TRAIN = 55_000


def pick_best(df, method, base_acc, tol_pp=1.0):
    d = df[(df["method"] == method) & (df["test_acc"] >= base_acc - tol_pp / 100)]
    if d.empty:  # nothing within tolerance -> least compressed variant
        d = df[df["method"] == method].sort_values("param_sparsity").head(1)
    return d.sort_values("param_sparsity").iloc[-1]


@torch.inference_mode()
def inference_energy(model, x, passes, cfg, label):
    """Inference on the CPU (deployment target): GPU not used, so no GPU energy term."""
    model = model.cpu().eval()
    x = x.cpu()
    for xb, _ in batches(x, x[:, 0, 0, 0], 256, shuffle=False):   # warm-up pass
        model(xb)
    with energy_meter(cfg, label, gpu_active=False) as en:
        for _ in range(passes):
            for xb, _ in batches(x, x[:, 0, 0, 0], 256, shuffle=False):
                model(xb)
    n = passes * len(x)
    return {"inference_time_s": en["time_s"], "inference_ms_per_image": 1000 * en["time_s"] / n,
            "inference_kwh": en["energy_kwh"], "inference_mj_per_image": en["energy_kwh"] * 3.6e9 / n,
            "energy_source": en["energy_source"]}


def main():
    cfg = load_cfg()
    torch.set_num_threads(cfg["latency"]["threads"])
    data = load_data(cfg, device=torch.device("cpu"))
    base = read_json("baseline.json")
    pr = pd.read_csv(RES / "pruning.csv")
    ls = pd.read_csv(RES / "latency_size.csv").set_index("variant")
    dc = pd.read_csv(RES / "distill_compare.csv").set_index("model")
    dd = pd.read_csv(RES / "distill.csv")
    dt = read_json("distill_teacher.json")
    T = cfg["teacher"]["epochs"]
    FT = cfg["pruning"]["finetune_epochs"]
    SE = cfg["student"]["epochs"]
    base_e = base["train_energy"]["energy_kwh"]

    g = pick_best(pr, "unstructured_global", base["test_acc"])
    s = pick_best(pr, "structured", base["test_acc"])
    plain = dd[(dd["mode"] == "plain") & (dd["seed"] == cfg["seed"])].iloc[0]
    kd = dd[(dd["mode"] == "kd") & (dd["seed"] == cfg["seed"])].iloc[0]

    entries = [
        ("baseline (teacher)", MODELS_DIR / "baseline.pt", base["test_acc"], ls.loc["baseline"],
         T * N_TRAIN, "labels only", base_e, base_e),
        (f"global-pruned {g['target']:.0%}", CKPT / f"{g['variant']}.pt", g["test_acc"], ls.loc[g["variant"]],
         (T + FT) * N_TRAIN, "labels; starts from baseline", g["ft_energy_kwh"], base_e + g["ft_energy_kwh"]),
        (f"structured-pruned {s['target']:.0%} filters", CKPT / f"{s['variant']}.pt", s["test_acc"],
         ls.loc[s["variant"]], (T + FT) * N_TRAIN, "labels; starts from baseline", s["ft_energy_kwh"],
         base_e + s["ft_energy_kwh"]),
        ("student, no teacher", MODELS_DIR / "student_plain.pt", plain["test_acc"],
         dc.loc["student plain"], SE * N_TRAIN, "labels only", plain["train_energy_kwh"],
         plain["train_energy_kwh"]),
        ("student, distilled", MODELS_DIR / "student_kd.pt", kd["test_acc"], dc.loc["student distilled"],
         SE * N_TRAIN + N_TRAIN, "labels + teacher soft targets (T=4)", kd["train_energy_kwh"],
         base_e + dt["teacher_logits_energy"]["energy_kwh"] + kd["train_energy_kwh"]),
    ]
    rows, inf_rows = [], []
    # Latency: prefer the interleaved control (all models measured in alternating rounds, so
    # laptop drift cancels); otherwise the sequential measurement of latency_size / distill.
    inter = pd.read_csv(RES / "latency_interleaved.csv") if (RES / "latency_interleaved.csv").exists() else None
    inter_key = {"baseline.pt": "baseline (dense)",
                 "unstructured_global_0.9.pt": "global-pruned 90% (masked, dense kernels)",
                 "unstructured_global_0.95.pt": "global-pruned 95% (masked, dense kernels)",
                 "structured_0.5.pt": "structured-pruned 50% (physically smaller)",
                 "student_plain.pt": "student (100k)",
                 "student_kd.pt": "student (100k)"}   # same architecture -> same latency

    def lat(path, m):
        key = inter_key.get(path.name)
        if inter is not None and key in set(inter["model"]):
            r = inter[inter["model"] == key].iloc[0]
            return r["latency_b1_ms_median"], r["latency_b256_ms_median"], "interleaved control"
        return m["latency_b1_ms"], m["latency_bbig_ms"], "sequential"

    for name, path, acc, m, samples, signal, own_e, cum_e in entries:
        model = load_model(path)
        ie = inference_energy(model, data["test"][0], cfg["energy"]["inference_passes"], cfg, name)
        inf_rows.append({"model": name, **ie})
        rows.append({
            "model": name, "test_acc": acc,
            # DATA pillar
            "train_samples_processed": samples, "supervision": signal,
            # MODEL pillar
            "params": int(m["params"]), "nonzero_params": int(m["nonzero_params"]),
            "size_dense_kb": m["size_dense_kb"], "size_best_kb": min(m["size_dense_zip_kb"], m["size_csr_zip_kb"]),
            # COMPUTE pillar
            "dense_macs": int(m["dense_macs"]), "effective_macs": int(m["effective_macs"]),
            "thop_macs": thop_macs(model),
            "latency_b1_ms": lat(path, m)[0], "latency_b256_ms": lat(path, m)[1],
            "latency_source": lat(path, m)[2],
            # ENERGY pillar
            "train_energy_own_kwh": own_e, "train_energy_cumulative_kwh": cum_e,
            "inference_mj_per_image": ie["inference_mj_per_image"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(RES / "summary_pillars.csv", index=False)
    pd.DataFrame(inf_rows).to_csv(RES / "inference_energy.csv", index=False)
    print(df.round(4).T.to_string())

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for _, r in df.iterrows():
        for a, col in zip(ax, ["size_best_kb", "latency_b1_ms", "effective_macs"]):
            a.scatter(r[col], r["test_acc"] * 100, s=60)
            a.annotate(r["model"], (r[col], r["test_acc"] * 100), textcoords="offset points",
                       xytext=(5, 4), fontsize=7)
    for a, t in zip(ax, ["smallest on-disk size (KB, log)", "latency batch 1 (ms)", "effective MACs / image (log)"]):
        a.set_xlabel(t); a.set_ylabel("test accuracy (%)"); a.grid(alpha=0.3)
    ax[0].set_xscale("log"); ax[2].set_xscale("log")
    fig.tight_layout()
    fig.savefig(RES / "summary_tradeoff.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
