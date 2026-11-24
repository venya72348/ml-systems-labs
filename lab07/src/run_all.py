"""Lab 7 runner: every experiment is a separate Python process.

    python lab07/src/run_all.py            # run everything that is not done yet
    python lab07/src/run_all.py --dry-run  # only print the plan and the runtime estimate

* resume-safe: an experiment is skipped when its result file already exists in results/
  (delete the file to force a rerun); aggregation steps are cheap and always re-run;
* isolation: memory is released between runs, a crashed run is logged and the rest continue
  (later stages that need its output then fail visibly in the log instead of silently);
* RSS watchdog: the runner samples the child's RSS (incl. its children) every 0.5 s, logs the
  peak and kills the child if it exceeds rss_limit_mb from configs/lab07.yaml.
Log: results/runlog.csv
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import psutil
import yaml

SRC = Path(__file__).resolve().parent
LAB = SRC.parent
RES = LAB / "results"
CFG = yaml.safe_load((LAB / "configs" / "lab07.yaml").read_text(encoding="utf-8"))

# seconds per training epoch on this machine (GTX 1050 Ti, data preloaded on the GPU),
# from a calibration run; used only for the up-front estimate
SEC_PER_EPOCH = {"teacher": 11.5, "student": 5.0}
PROC_S = 15.0   # interpreter + torch + CUDA start-up per process
MAX_ATTEMPTS = 3   # 1 run + up to 2 retries after a native crash


def stages():
    sys_py = [sys.executable, "-X", "faulthandler", "-W", "ignore"]   # faulthandler: dump native-crash stacks
    T, P, S = CFG["teacher"], CFG["pruning"], CFG["student"]
    out = [("baseline", sys_py + [str(SRC / "baseline.py")], RES / "baseline.json",
            PROC_S + T["epochs"] * SEC_PER_EPOCH["teacher"] + 20)]
    tags = [f"unstructured_{m}_{s}" for m in ["global", "layer"] for s in P["sparsities"]
            if not (m == "layer" and s == 0)] + [f"structured_{r}" for r in P["structured_ratios"]]
    for t in tags:
        out.append((f"prune {t}", sys_py + [str(SRC / "prune.py"), "--variant", t],
                    RES / "pruning" / f"{t}.json", PROC_S + P["finetune_epochs"] * SEC_PER_EPOCH["teacher"]))
    out.append(("prune aggregate", sys_py + [str(SRC / "prune.py"), "--aggregate"], None, PROC_S))
    out.append(("latency + size", sys_py + [str(SRC / "latency_size.py")], RES / "latency_size.csv", 240))
    out.append(("latency interleaved (control)", sys_py + [str(SRC / "latency_interleaved.py")],
                RES / "latency_interleaved.csv", 150))
    for seed in CFG["distill"]["seeds"]:
        for mode in ["plain", "kd"]:
            out.append((f"distill {mode} seed {seed}",
                        sys_py + [str(SRC / "distill.py"), "--mode", mode, "--seed", str(seed)],
                        RES / "distill" / f"{mode}_{seed}.json", PROC_S + S["epochs"] * SEC_PER_EPOCH["student"]))
    out.append(("distill aggregate", sys_py + [str(SRC / "distill.py"), "--aggregate"], None, PROC_S + 20))
    out.append(("summary", sys_py + [str(SRC / "summary.py")], RES / "summary_pillars.csv", 120))
    return out


def run_watched(cmd, limit_mb):
    proc = subprocess.Popen(cmd, cwd=LAB)
    ps = psutil.Process(proc.pid)
    peak, killed = 0.0, False
    while proc.poll() is None:
        try:
            rss = ps.memory_info().rss + sum(c.memory_info().rss for c in ps.children(recursive=True))
            peak = max(peak, rss / 2**20)
        except psutil.Error:
            pass
        if peak > limit_mb and not killed:
            print(f"!! RSS {peak:.0f} MB > {limit_mb} MB, killing", flush=True)
            proc.kill()
            killed = True
        time.sleep(0.5)
    return proc.returncode, peak, killed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    plan = stages()
    todo = [s for s in plan if s[2] is None or not s[2].exists()]
    est = sum(s[3] for s in todo)
    print(f"== Lab 7 plan: {len(todo)} steps to run, {len(plan) - len(todo)} already done; "
          f"estimated wall time ~{est / 60:.0f} min", flush=True)
    for name, _, out, sec in plan:
        state = "done" if (out is not None and out.exists()) else f"~{sec / 60:.1f} min"
        print(f"   {name:32s} {state}", flush=True)
    if args.dry_run:
        return
    RES.mkdir(parents=True, exist_ok=True)
    log = []
    for name, cmd, out, _ in todo:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            print(f"\n>> {name} (attempt {attempt})", flush=True)
            t0 = time.perf_counter()
            rc, peak, killed = run_watched(cmd, CFG["rss_limit_mb"])
            ok = rc == 0 and (out is None or out.exists())
            status = "ok" if ok else ("KILLED (RSS limit)" if killed else f"FAILED rc={rc}")
            log.append({"step": name, "attempt": attempt, "status": status,
                        "wall_s": round(time.perf_counter() - t0, 1), "peak_rss_mb": round(peak)})
            print(f"<< {name}: {status} in {log[-1]['wall_s']} s, peak RSS {peak:.0f} MB", flush=True)
            path = RES / "runlog.csv"
            old = pd.read_csv(path) if path.exists() else pd.DataFrame()
            pd.concat([old, pd.DataFrame(log[-1:])]).to_csv(path, index=False)
            # retry only native crashes (Windows NTSTATUS codes such as 0xC0000409 = abort/fail-fast);
            # Python exceptions (rc=1) and watchdog kills are real problems and are not retried
            if ok or killed or not (rc is not None and (rc > 255 or rc < 0)):
                break
    final = {l["step"]: l for l in log}          # last attempt per step
    bad = [l for l in final.values() if l["status"] != "ok"]
    retried = sum(1 for l in log if l["attempt"] > 1)
    print(f"\n== finished: {len(final) - len(bad)} ok, {len(bad)} failed, {retried} retries", flush=True)
    for b in bad:
        print("   FAILED:", b, flush=True)


if __name__ == "__main__":
    main()
