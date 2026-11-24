"""Lab 7, task 4: knowledge distillation from the ~1M-param baseline into a ~100k student.

loss = alpha * CE(student, y) + (1 - alpha) * T^2 * KL(softmax(teacher/T) || softmax(student/T)),
T = 4, alpha = 0.5, implemented with log_softmax + nn.KLDivLoss(reduction="batchmean")
(see common.kd_loss). The same student is trained with and without the teacher: same
architecture, same initial weights (same seed), same batches, same optimizer and epochs.
Teacher logits are computed once in eval mode at the start of each KD run (no augmentation is
used, so this is identical to running the teacher inside the loop, just cheaper).

    python lab07/src/distill.py --mode kd --seed 42     # one run (run_all.py does this)
    python lab07/src/distill.py --aggregate             # tables, size/latency comparison, plot
Outputs: results/distill/<mode>_<seed>.json, results/distill.csv, results/distill_history.csv,
         results/distill_compare.csv, results/distill_seed_stats.csv, results/distill_curves.png,
         models/student_plain.pt, models/student_kd.pt (seed 42)
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import (DEVICE, MODELS_DIR, RES, Net, disk_sizes, energy_meter, evaluate, load_cfg, load_data,
                    load_model, macs, measure_latency, n_nonzero, n_params, predict_logits, save_json,
                    save_model, set_seed, train)


def run_one(cfg, mode, seed):
    S, D = cfg["student"], cfg["distill"]
    data = load_data(cfg)
    t_logits, teacher_info = None, {}
    if mode == "kd":
        teacher = load_model(MODELS_DIR / "baseline.pt")
        with energy_meter(cfg, "teacher_logits") as en_t:
            t_logits = predict_logits(teacher, data["train"][0])
        teacher_info = {"teacher_train_acc": (t_logits.argmax(1) == data["train"][1]).float().mean().item(),
                        "teacher_test_acc": evaluate(teacher, *data["test"])[0], "teacher_logits_energy": en_t}
        del teacher
    set_seed(seed)
    model = Net(S["widths"]).to(DEVICE)          # identical init for plain and kd (same seed)
    with energy_meter(cfg, f"student_{mode}_{seed}") as en:
        hist = train(model, data, S["epochs"], S["lr"], S["batch_size"], seed, teacher_logits=t_logits,
                     T=D["T"], alpha=D["alpha"], tag=f"{mode} s{seed}")
    acc, loss = evaluate(model, *data["test"])
    train_acc, _ = evaluate(model, *data["train"])
    row = {"mode": mode, "seed": seed, "params": n_params(model), "test_acc": acc, "test_loss": loss,
           "val_acc": hist[-1]["val_acc"], "train_acc_eval": train_acc, "train_time_s": en["time_s"],
           "train_energy_kwh": en["energy_kwh"], **teacher_info, "energy": en, "history": hist}
    if seed == cfg["seed"]:
        save_model(model, MODELS_DIR / f"student_{mode}.pt")
    (RES / "distill").mkdir(parents=True, exist_ok=True)
    (RES / "distill" / f"{mode}_{seed}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print({k: v for k, v in row.items() if k not in ("history", "energy")}, flush=True)


def aggregate(cfg):
    D = cfg["distill"]
    runs = [json.loads((RES / "distill" / f"{m}_{s}.json").read_text(encoding="utf-8"))
            for s in D["seeds"] for m in ["plain", "kd"]]
    drop = ("history", "energy", "teacher_logits_energy")
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in drop} for r in runs])
    df.to_csv(RES / "distill.csv", index=False)
    hist = pd.DataFrame([{"mode": r["mode"], "seed": r["seed"], **h} for r in runs for h in r["history"]])
    hist.to_csv(RES / "distill_history.csv", index=False)
    kd0 = next(r for r in runs if r["mode"] == "kd" and r["seed"] == cfg["seed"])
    save_json({k: kd0[k] for k in ("teacher_train_acc", "teacher_test_acc", "teacher_logits_energy")}
              | {"T": D["T"], "alpha": D["alpha"]}, "distill_teacher.json")

    data = load_data(cfg)
    cmp = []
    for name, path in [("teacher (baseline)", MODELS_DIR / "baseline.pt"),
                       ("student plain", MODELS_DIR / "student_plain.pt"),
                       ("student distilled", MODELS_DIR / "student_kd.pt")]:
        m = load_model(path)
        cmp.append({"model": name, "params": n_params(m), "nonzero_params": n_nonzero(m),
                    "test_acc": evaluate(m, *data["test"])[0], **macs(m),
                    "file_size_kb": path.stat().st_size / 1024, **disk_sizes(m), **measure_latency(m, cfg)})
    pd.DataFrame(cmp).to_csv(RES / "distill_compare.csv", index=False)
    agg = df.groupby("mode")["test_acc"].agg(["mean", "std", "min", "max"]).reset_index()
    paired = df.pivot(index="seed", columns="mode", values="test_acc")
    paired["kd_minus_plain_pp"] = 100 * (paired["kd"] - paired["plain"])
    agg.to_csv(RES / "distill_seed_stats.csv", index=False)
    paired.reset_index().to_csv(RES / "distill_paired.csv", index=False)
    print(pd.DataFrame(cmp).round(3).to_string(index=False))
    print(agg, paired, sep="\n")
    plot(hist, df, kd0["teacher_test_acc"], cfg["seed"])


def plot(h, df, t_acc, seed0):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for mode, c in [("plain", "C0"), ("kd", "C3")]:
        lab = "student, no teacher" if mode == "plain" else "student, distilled"
        for s in sorted(h["seed"].unique()):
            d = h[(h["mode"] == mode) & (h["seed"] == s)]
            lw, al = (2, 1) if s == seed0 else (0.8, 0.45)
            ax[0].plot(d["epoch"], d["val_acc"] * 100, color=c, lw=lw, alpha=al, label=lab if s == seed0 else None)
            ax[1].plot(d["epoch"], d["val_loss"], color=c, lw=lw, alpha=al, label=lab if s == seed0 else None)
    ax[0].axhline(t_acc * 100, color="gray", ls="--", lw=1, label="teacher (test)")
    ax[0].set_title("Validation accuracy (%), thick = seed 42")
    ax[0].set_ylim(max(80, h["val_acc"].min() * 100 - 1), t_acc * 100 + 1.5)
    ax[1].set_title("Validation cross-entropy (hard labels)")
    for a in ax[:2]:
        a.set_xlabel("epoch"); a.grid(alpha=0.3); a.legend(fontsize=8)
    for i, mode in enumerate(["plain", "kd"]):
        d = df[df["mode"] == mode]
        ax[2].scatter([i] * len(d), d["test_acc"] * 100, color=f"C{0 if mode == 'plain' else 3}", s=40)
        ax[2].errorbar(i + 0.15, d["test_acc"].mean() * 100, yerr=d["test_acc"].std() * 100, fmt="D", color="k")
    ax[2].set_xticks([0, 1], ["no teacher", "distilled"]); ax[2].set_xlim(-0.5, 1.5)
    ax[2].set_title("Test accuracy per seed (mean +- std)"); ax[2].set_ylabel("%"); ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "distill_curves.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["plain", "kd"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg()
    if args.aggregate:
        aggregate(cfg)
    else:
        run_one(cfg, args.mode, args.seed)


if __name__ == "__main__":
    main()
