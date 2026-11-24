"""Lab 6: run a study (a list of config overrides) through train.py and aggregate results.

    python lab06/src/studies.py base            # 10-epoch base run (configs/base.yaml as is)
    python lab06/src/studies.py profile         # task 2
    python lab06/src/studies.py num_workers     # task 3
    python lab06/src/studies.py batch_size      # task 4
    python lab06/src/studies.py schedules       # task 5
    python lab06/src/studies.py grad_accum      # task 6
    python lab06/src/studies.py resume          # task 7
    python lab06/src/studies.py all
    python lab06/src/studies.py <study> --analyze-only   # rebuild tables/plots from saved runs

Each run is a separate `python train.py --config configs/base.yaml --set ...` process, so every
experiment uses exactly the same training script and starts from a clean process (clean seeds,
clean memory measurement). Tables (CSV) and plots (PNG) are written to lab06/results/.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

LAB_DIR = Path(__file__).resolve().parents[1]
SRC = LAB_DIR / "src"
CFG = LAB_DIR / "configs"
RES = LAB_DIR / "results"
RUNS = RES / "runs"
STUDIES = ["base", "profile", "num_workers", "batch_size", "schedules", "grad_accum", "resume"]


def fmt(v):
    return "null" if v is None else (str(v).lower() if isinstance(v, bool) else str(v))


# Rough wall-clock model for the up-front estimate (seconds per full epoch + per-process start-up).
# Calibrated on this laptop (GTX 1050 Ti): preloaded ~3.5 s/epoch, start-up+eval ~30 s per process;
# per-image pipeline guessed from the CPU-side transform cost measured on CPU (~17 s of 37 s).
SEC_PER_EPOCH = {"per_image": 22.0, "preloaded": 4.0}
PROC_OVERHEAD_S = 30.0


def study_spec(study):
    if study == "base":
        return {"common": {}, "runs": [{"name": "base", "set": {}}]}
    return yaml.safe_load((CFG / f"study_{study}.yaml").read_text(encoding="utf-8"))


def run_overrides(spec, r):
    return {**spec.get("common", {}), **r["set"], "run.name": r["name"]}


def already_done(ov):
    """Resume-safe skip: a run is done when its summary.json exists. The first phase of an
    interrupt/resume pair only needs a summary; a normal run needs a non-interrupted one."""
    p = RUNS / ov["run.name"] / "summary.json"
    if not p.exists():
        return False
    if ov.get("train.stop_after_epoch"):
        return True
    return not json.loads(p.read_text(encoding="utf-8"))["interrupted"]


def estimate_s(ov):
    base = yaml.safe_load((CFG / "base.yaml").read_text(encoding="utf-8"))
    epochs = ov.get("train.epochs", base["train"]["epochs"])
    if ov.get("train.stop_after_epoch"):
        epochs = ov["train.stop_after_epoch"]
    elif ov.get("train.resume_from"):
        epochs -= 3
    if ov.get("train.max_steps_per_epoch"):
        epochs *= ov["train.max_steps_per_epoch"] / 430
    pipe = "preloaded" if ov.get("data.preload", base["data"]["preload"]) else "per_image"
    return PROC_OVERHEAD_S + epochs * SEC_PER_EPOCH[pipe]


def plan(studies):
    todo = []
    for st in studies:
        spec = study_spec(st)
        for r in spec["runs"]:
            ov = run_overrides(spec, r)
            todo.append((st, ov, already_done(ov)))
    return todo


WATCHDOG_LIMIT_MB = 6000


def run_watched(cmd):
    """Run one experiment process; every 0.5 s check how much system memory has been consumed
    since it started (covers the child, its DataLoader workers and profiler post-processing,
    without over-counting shared DLL pages like summed RSS). Kill it above ~6 GB or when free
    memory drops under 1 GB."""
    import psutil
    avail0 = psutil.virtual_memory().available / 2**20
    proc = subprocess.Popen(cmd, cwd=LAB_DIR)
    peak, killed = 0.0, False
    while proc.poll() is None:
        avail = psutil.virtual_memory().available / 2**20
        peak = max(peak, avail0 - avail)
        if (peak > WATCHDOG_LIMIT_MB or avail < 1024) and not killed:
            print(f"!! memory watchdog: +{peak:.0f} MB, {avail:.0f} MB free -> killing", flush=True)
            proc.kill()
            killed = True
        time.sleep(0.5)
    return proc.returncode, peak, killed


def run_study(study):
    """Each run is its own `python train.py` process: memory is fully released between runs and
    a crashed run is recorded and skipped instead of killing the rest of the study."""
    spec = study_spec(study)
    log_lines = []
    for r in spec["runs"]:
        overrides = run_overrides(spec, r)
        if already_done(overrides):
            print(f"-- skip {r['name']} (results exist)", flush=True)
            continue
        cmd = [sys.executable, "-W", "ignore", str(SRC / "train.py"), "--config", str(CFG / "base.yaml")]
        for k, v in overrides.items():
            cmd += ["--set", f"{k}={fmt(v)}"]
        print(">>", " ".join(cmd[3:]), flush=True)
        t0 = time.perf_counter()
        rc, watchdog_peak, killed = run_watched(cmd)
        ok = rc == 0 and already_done(overrides)
        summ_p = RUNS / r["name"] / "summary.json"
        peak = json.loads(summ_p.read_text(encoding="utf-8")).get("peak_rss_incl_workers_mb") if ok else None
        status = "ok" if ok else ("KILLED (memory watchdog)" if killed else f"FAILED rc={rc}")
        log_lines.append({"study": study, "run": r["name"], "status": status,
                          "wall_s": round(time.perf_counter() - t0, 1), "peak_rss_mb": peak,
                          "watchdog_peak_sys_increase_mb": round(watchdog_peak),
                          "overrides": json.dumps(overrides)})
        print(f"<< {r['name']}: {log_lines[-1]['status']} in {log_lines[-1]['wall_s']} s, "
              f"peak RSS {peak} MB", flush=True)
        if not ok and r.get("stop_study_on_fail", "resume" in study):
            print(f"!! stopping study {study}: later runs depend on this one", flush=True)
            break
    if log_lines:
        path = RES / "wallclock.csv"
        old = pd.read_csv(path) if path.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame(log_lines)]).to_csv(path, index=False)
    return all(l["status"] == "ok" for l in log_lines)


def hist(name):
    return pd.read_csv(RUNS / name / "history.csv")


def steps(name):
    p = RUNS / name / "steps.csv"
    return pd.read_csv(p) if p.stat().st_size else pd.DataFrame()


def summ(name):
    return json.loads((RUNS / name / "summary.json").read_text(encoding="utf-8"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(RES / name, dpi=130)
    plt.close(fig)


def write_table(df, name):
    df.to_csv(RES / f"{name}.csv", index=False)
    print(f"\n== {name} ==\n{df.to_string(index=False)}")


# ---------------------------------------------------------------- analyses
def analyze_base():
    h, s = hist("base"), summ("base")
    df = h[["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time_s", "throughput_sps"]]
    write_table(df.round(4), "base_history")
    st = steps("base")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(st["epoch"], st["train_loss"], lw=0.8, alpha=0.5, label="train (every 20 steps)")
    ax[0].plot(h["epoch"], h["val_loss"], "s-", label="val")
    ax[0].set_title("Loss")
    ax[1].plot(h["epoch"], h["train_acc"] * 100, "o-", label="train")
    ax[1].plot(h["epoch"], h["val_acc"] * 100, "s-", label="val")
    ax[1].set_title(f"Accuracy (%)   test = {s['test_acc']*100:.2f}%")
    ax[2].plot(h["epoch"], h["lr"], "o-")
    ax[2].set_title("LR (cosine)")
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=0.3)
    ax[0].legend(); ax[1].legend()
    save(fig, "base_curves.png")


def analyze_profile():
    """Main table: manual perf_counter timers over the FULL epoch (phase_nw0 / phase_nw4).
    Evidence: torch.profiler over the first 150 steps (torchprof_nw0)."""
    phases = [("loader start (spawn workers)", "loader_start_s"),
              ("data_loading", "data_time_s"), ("forward", "forward_s"), ("backward", "backward_s"),
              ("optimizer_step", "optimizer_s"), ("other (loop, metrics, .item())", "other_s")]
    rows = []
    for run in ["phase_nw0", "phase_nw4"]:
        h = hist(run).iloc[0]
        for ph, col in phases:
            rows.append({"run": run, "phase": ph, "time_s": h[col], "share_pct": 100 * h[col] / h["epoch_time_s"]})
        rows.append({"run": run, "phase": "TOTAL epoch", "time_s": h["epoch_time_s"], "share_pct": 100.0})
    df = pd.DataFrame(rows)
    write_table(df.round(3), "profile_breakdown")

    tp = pd.read_csv(RUNS / "torchprof_nw0" / "profile_breakdown.csv")
    write_table(tp.round(3), "torchprof_breakdown_150steps")
    top = pd.read_csv(RUNS / "torchprof_nw0" / "profile_top_ops.csv")
    write_table(top.round(4), "torchprof_top_ops_150steps")

    fig, ax = plt.subplots(1, 3, figsize=(17, 3.9))
    colors = ["#8172B3", "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8C8C8C"]
    panels = [(df[(df["run"] == r) & (df["phase"] != "TOTAL epoch")], t) for r, t in
              [("phase_nw0", "full epoch, num_workers=0 (manual timers)"),
               ("phase_nw4", "full epoch, num_workers=4 (manual timers)")]]
    tpp = tp.rename(columns={"phase": "phase"})
    panels.append((tpp, "torch.profiler, first 150 steps, num_workers=0"))
    xmax = max(p[0]["share_pct"].max() for p in panels) * 1.4
    for a, (d, title) in zip(ax, panels):
        cols = colors if len(d) == 6 else colors[1:]   # same colour per phase in every panel
        a.barh(d["phase"].str.slice(0, 28), d["share_pct"], color=cols[: len(d)])
        for y, (v, t) in enumerate(zip(d["share_pct"], d["time_s"])):
            a.text(v + 0.5, y, f"{v:.1f}%  ({t:.2f}s)", va="center", fontsize=8)
        a.invert_yaxis(); a.set_xlim(0, xmax)
        a.set_title(title + f"\nwall {d['time_s'].sum():.1f} s", fontsize=9)
        a.set_xlabel("share of wall time (%)"); a.grid(axis="x", alpha=0.3)
    save(fig, "profile_breakdown.png")


def analyze_num_workers():
    rows = []
    for nw in [0, 1, 2, 4, 8]:
        h = hist(f"nw_{nw}")
        s = summ(f"nw_{nw}")
        e1, e2 = h.iloc[0], h.iloc[1]
        rows.append({
            "num_workers": nw,
            "epoch1_time_s": e1["epoch_time_s"], "first_batch_s": e1["first_batch_s"],
            "epoch2_time_s": e2["epoch_time_s"],
            "epoch2_throughput_sps": e2["throughput_sps"],
            "epoch2_data_wait_s": e2["data_time_s"], "epoch2_compute_s": e2["compute_time_s"],
            "data_wait_pct": 100 * e2["data_time_s"] / e2["epoch_time_s"],
            "peak_rss_incl_workers_mb": s["peak_rss_incl_workers_mb"],
        })
    df = pd.DataFrame(rows)
    df["speedup_vs_0"] = df["epoch2_throughput_sps"] / df.loc[0, "epoch2_throughput_sps"]
    write_table(df.round(3), "num_workers")

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    x = df["num_workers"].astype(str)
    ax[0].plot(x, df["epoch2_throughput_sps"], "o-")
    ax[0].set_title("Throughput, steady state (epoch 2)"); ax[0].set_ylabel("samples/s")
    ax[1].bar(x, df["epoch2_compute_s"], label="compute (fwd+bwd+step)")
    ax[1].bar(x, df["epoch2_data_wait_s"], bottom=df["epoch2_compute_s"], label="waiting for data")
    ax[1].set_title("Epoch-2 time split"); ax[1].set_ylabel("s"); ax[1].legend()
    ax[2].bar(x, df["first_batch_s"], color="#C44E52")
    ax[2].set_title("Time to first batch (worker start-up)"); ax[2].set_ylabel("s")
    for a in ax:
        a.set_xlabel("num_workers"); a.grid(alpha=0.3)
    save(fig, "num_workers.png")


def analyze_batch_size():
    rows = []
    for bs in [32, 64, 128, 256, 512]:
        h, s = hist(f"bs_{bs}"), summ(f"bs_{bs}")
        rows.append({
            "batch_size": bs,
            "steps_per_epoch": -(-55000 // bs),
            "throughput_sps": h["throughput_sps"].mean(),
            "epoch_time_s": h["epoch_time_s"].mean(),
            "device": s["hardware"]["device"],
            "peak_train_mem_mb": s["peak_memory_mb"],      # GPU: cuda.max_memory_allocated (training only)
            "peak_process_rss_mb": s.get("peak_rss_mb"),
            "val_acc_ep5": s["final_val_acc"],
            "test_acc_ep5": s["test_acc"],
            "train_loss_ep5": h["train_loss"].iloc[-1],
        })
    df = pd.DataFrame(rows)
    write_table(df.round(4), "batch_size")

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    x = df["batch_size"]
    ax[0].plot(x, df["throughput_sps"], "o-"); ax[0].set_title("Throughput"); ax[0].set_ylabel("samples/s")
    ax[1].plot(x, df["peak_train_mem_mb"], "o-", label="GPU max_memory_allocated (training)")
    ax[1].set_title("Peak training memory"); ax[1].set_ylabel("MB"); ax[1].legend()
    ax[2].plot(x, df["val_acc_ep5"] * 100, "o-", label="val")
    ax[2].plot(x, df["test_acc_ep5"] * 100, "s-", label="test")
    ax[2].set_title("Accuracy after 5 epochs"); ax[2].set_ylabel("%"); ax[2].legend()
    for a in ax:
        a.set_xscale("log", base=2); a.set_xticks(x, [str(v) for v in x])
        a.set_xlabel("batch size"); a.grid(alpha=0.3)
    save(fig, "batch_size.png")


def analyze_schedules():
    names = [f"sched_{o}_{s}" for o in ["sgd", "adam"] for s in ["step", "cosine", "plateau"]]
    rows = []
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for n in names:
        h, s = hist(n), summ(n)
        _, opt, sch = n.split("_")
        rows.append({"optimizer": opt, "scheduler": sch, "lr0": s["config"]["optim"]["lr"],
                     "final_val_acc": s["final_val_acc"], "best_val_acc": s["best_val_acc"],
                     "test_acc": s["test_acc"], "final_train_loss": h["train_loss"].iloc[-1],
                     "final_val_loss": h["val_loss"].iloc[-1], "final_lr": h["lr"].iloc[-1]})
        ls = "-" if opt == "adam" else "--"
        ax[0].plot(h["epoch"], h["val_acc"] * 100, ls, marker="o", ms=3, label=f"{opt}+{sch}")
        ax[1].plot(h["epoch"], h["train_loss"], ls, marker="o", ms=3, label=f"{opt}+{sch}")
        ax[2].plot(h["epoch"], h["lr"] / h["lr"].iloc[0], ls, marker="o", ms=3, label=f"{opt}+{sch}")
    ax[0].set_title("Validation accuracy (%)")
    ax[1].set_title("Train loss")
    ax[2].set_title("LR / initial LR (LR used in epoch)")
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    save(fig, "schedules.png")
    write_table(pd.DataFrame(rows).round(4), "schedules")


def analyze_grad_accum():
    a, b = "accum_real256", "accum_64x4"
    ha, hb, sa, sb = hist(a), hist(b), summ(a), summ(b)
    sta, stb = steps(a), steps(b)
    m = sta.merge(stb, on="step", suffixes=("_real", "_accum"))
    rows = [
        {"run": "real batch 256", "optimizer_steps_per_epoch": 215,
         "final_train_loss": ha["train_loss"].iloc[-1], "final_val_loss": ha["val_loss"].iloc[-1],
         "final_val_acc": sa["final_val_acc"], "test_acc": sa["test_acc"],
         "throughput_sps": ha["throughput_sps"].mean(), "peak_train_mem_mb": sa["peak_memory_mb"]},
        {"run": "batch 64 x accum 4", "optimizer_steps_per_epoch": 215,
         "final_train_loss": hb["train_loss"].iloc[-1], "final_val_loss": hb["val_loss"].iloc[-1],
         "final_val_acc": sb["final_val_acc"], "test_acc": sb["test_acc"],
         "throughput_sps": hb["throughput_sps"].mean(), "peak_train_mem_mb": sb["peak_memory_mb"]},
    ]
    write_table(pd.DataFrame(rows).round(5), "grad_accum")
    diff = pd.DataFrame([{
        "max_abs_diff_step_loss": (m["train_loss_real"] - m["train_loss_accum"]).abs().max(),
        "mean_abs_diff_step_loss": (m["train_loss_real"] - m["train_loss_accum"]).abs().mean(),
        "max_abs_diff_val_acc_pp": ((ha["val_acc"] - hb["val_acc"]).abs() * 100).max(),
        "max_abs_diff_val_loss": (ha["val_loss"] - hb["val_loss"]).abs().max(),
    }])
    write_table(diff, "grad_accum_diff")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(m["epoch_real"], m["train_loss_real"], lw=1.6, label="real batch 256")
    ax[0].plot(m["epoch_accum"], m["train_loss_accum"], "--", lw=1.2, label="64 x accum 4")
    ax[0].set_title("Train loss per optimizer step (every 5 steps)"); ax[0].set_ylim(0.1, 1.0)
    ax[1].plot(ha["epoch"], ha["val_loss"], "o-", label="real batch 256")
    ax[1].plot(hb["epoch"], hb["val_loss"], "x--", ms=9, label="64 x accum 4")
    ax[1].set_title("Validation loss")
    ax[2].plot(ha["epoch"], ha["val_acc"] * 100, "o-", label="real batch 256")
    ax[2].plot(hb["epoch"], hb["val_acc"] * 100, "x--", ms=9, label="64 x accum 4")
    ax[2].set_title("Validation accuracy (%)")
    for x in ax:
        x.set_xlabel("epoch"); x.grid(alpha=0.3); x.legend()
    save(fig, "grad_accum.png")


def analyze_resume():
    hf, hs = hist("resume_full"), hist("resume_split")
    sf, ss = steps("resume_full"), steps("resume_split")
    cols = ["epoch", "train_loss", "val_loss", "val_acc", "lr"]
    m = hf[cols].merge(hs[cols + ["resumed"]], on="epoch", suffixes=("_full", "_split"))
    for c in ["train_loss", "val_loss", "val_acc", "lr"]:
        m[f"absdiff_{c}"] = (m[f"{c}_full"] - m[f"{c}_split"]).abs()
    write_table(m, "resume")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(sf["epoch"], sf["train_loss"], lw=2.2, alpha=0.6, label="uninterrupted")
    pre, post = ss[ss["epoch"] <= 3], ss[ss["epoch"] > 3]
    ax[0].plot(pre["epoch"], pre["train_loss"], "--", color="k", lw=1, label="interrupted run (epochs 1-3)")
    ax[0].plot(post["epoch"], post["train_loss"], "--", color="C3", lw=1, label="resumed run (epochs 4-6)")
    ax[0].set_title("Train loss per step"); ax[0].set_ylim(0.15, 1.0)
    for a, c, t in [(ax[1], "val_loss", "Validation loss"), (ax[2], "val_acc", "Validation accuracy")]:
        a.plot(hf["epoch"], hf[c], "o-", lw=2.2, ms=9, alpha=0.6, label="uninterrupted")
        a.plot(hs["epoch"][:3], hs[c][:3], "x--", color="k", ms=8, label="interrupted (1-3)")
        a.plot(hs["epoch"][2:], hs[c][2:], "x--", color="C3", ms=8, label="resumed (4-6)")
        a.set_title(t)
    for a in ax:
        a.axvline(3, color="gray", ls=":", lw=1)
        a.set_xlabel("epoch"); a.grid(alpha=0.3); a.legend(fontsize=8)
    save(fig, "resume.png")


ANALYZE = {"base": analyze_base, "profile": analyze_profile, "num_workers": analyze_num_workers,
           "batch_size": analyze_batch_size, "schedules": analyze_schedules,
           "grad_accum": analyze_grad_accum, "resume": analyze_resume}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study", choices=STUDIES + ["all"])
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    studies = STUDIES if args.study == "all" else [args.study]
    if not args.analyze_only:
        todo = plan(studies)
        pending = [(st, ov) for st, ov, done in todo if not done]
        total = sum(estimate_s(ov) for _, ov in pending)
        print(f"== plan: {len(pending)} runs to do, {len(todo) - len(pending)} already done; "
              f"estimated wall time ~{total / 60:.0f} min", flush=True)
        for st, ov in pending:
            print(f"   {st:12s} {ov['run.name']:20s} ~{estimate_s(ov) / 60:.1f} min", flush=True)
    for st in studies:
        if not args.analyze_only:
            run_study(st)
        try:
            ANALYZE[st]()
        except Exception as e:  # missing runs -> report, keep going with the other studies
            print(f"!! analysis of {st} failed: {e!r}", flush=True)


if __name__ == "__main__":
    main()
