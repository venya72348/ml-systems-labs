"""Lab 6: config-driven training pipeline for the Lab 3 CNN on Fashion-MNIST.

One script for every experiment. Behaviour is controlled only by a YAML config plus
`--set key=value` overrides (dotted keys, values parsed as YAML):

    python lab06/src/train.py --config lab06/configs/base.yaml
    python lab06/src/train.py --config lab06/configs/base.yaml --set optim.name=sgd --set optim.lr=0.05

Features: fixed seeds, full checkpoints (model + optimizer + scheduler + epoch + RNG state),
resume, gradient accumulation, torch.profiler with record_function labels, MLflow logging
(params, per-epoch metrics, config/plot/checkpoint artifacts), timing and peak memory.
"""
import argparse
import copy
import csv
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import torch.nn as nn
import yaml
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import (BatchSampler, DataLoader, RandomSampler, SequentialSampler, TensorDataset,
                              random_split)
from torchvision import datasets, transforms

LAB_DIR = Path(__file__).resolve().parents[1]
PROFILE_LABELS = ["data_loading", "forward", "backward", "optimizer_step"]


# ---------------------------------------------------------------- config
def set_by_path(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node:
            raise KeyError(f"unknown config section '{k}' in '{dotted}'")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"unknown config key '{dotted}'")
    node[keys[-1]] = value


def load_config(path, overrides=()):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for ov in overrides:
        key, _, raw = ov.partition("=")
        set_by_path(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def lab_path(p):
    p = Path(p)
    return p if p.is_absolute() else (LAB_DIR / p).resolve()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))   # union, first-seen order
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- model (copied from lab03/src/lab03.py)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.net(x)


MODELS = {"cnn_lab3": CNN}


# ---------------------------------------------------------------- reproducibility / system
def seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def hardware_info(device):
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_logical": psutil.cpu_count(),
        "cpu_physical": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": str(device),
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def rss_mb():
    return psutil.Process().memory_info().rss / 2**20


class RssSampler:
    """Background thread that records the max RSS of this process (+ DataLoader workers)
    while training runs. Process-lifetime peaks (peak_wset / ru_maxrss) would also include
    one-off spikes from imports or the profiler, so we sample only the training phase."""

    def check(self):
        """Called every training step: abort the run before it can exhaust system RAM.

        Summed RSS over-counts on Windows: every DataLoader worker re-imports torch and the
        (shared, file-backed) CUDA DLL pages appear in each worker's RSS. The guard therefore
        uses the drop in SYSTEM available memory since the run started (= the physical memory
        this run really added) plus a hard floor on free memory."""
        if self.limit_mb and (self.peak_sys_increase > self.limit_mb or self.min_avail_mb < 1024):
            raise MemoryError(f"run footprint {self.peak_sys_increase:.0f} MB (limit {self.limit_mb}) / "
                              f"system free {self.min_avail_mb:.0f} MB; reduce this experiment's footprint")

    def __init__(self, track_children, limit_mb=None, interval_s=0.05, children_refresh_s=2.0):
        import threading
        self.limit_mb = limit_mb
        self.track_children = track_children
        self.interval_s = interval_s
        self.children_refresh_s = children_refresh_s
        self.peak_main = 0.0
        self.peak_total = 0.0
        self.avail0 = psutil.virtual_memory().available / 2**20
        self.peak_sys_increase = 0.0
        self.min_avail_mb = self.avail0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        # memory_info() costs ~7 us, but children() scans the whole process table (~8 ms on
        # Windows), so the worker list is refreshed rarely and only when workers exist.
        proc = psutil.Process()
        kids, last_refresh, tick_n = [], -1e9, 0
        while not self._stop.is_set():
            now = time.perf_counter()
            if self.track_children and now - last_refresh > self.children_refresh_s:
                try:
                    kids = proc.children(recursive=True)
                except psutil.Error:
                    kids = []
                last_refresh = now
            main = proc.memory_info().rss
            kid_rss = 0
            for c in kids:
                try:
                    kid_rss += c.memory_info().rss
                except psutil.Error:
                    pass
            tick_n += 1
            if tick_n % 10 == 0:   # virtual_memory() costs ~3 ms on Windows -> every 0.5 s
                avail = psutil.virtual_memory().available / 2**20
                self.min_avail_mb = min(self.min_avail_mb, avail)
                self.peak_sys_increase = max(self.peak_sys_increase, self.avail0 - avail)
            self.peak_main = max(self.peak_main, main / 2**20)
            self.peak_total = max(self.peak_total, (main + kid_rss) / 2**20)
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


# ---------------------------------------------------------------- data
MEAN, STD = 0.2860, 0.3530


def build_datasets(cfg):
    """data.preload=false: original torchvision pipeline, PIL image -> ToTensor -> Normalize
    per sample, every epoch (used by the profiling and num_workers studies so the data-loading
    cost is real). data.preload=true: decode + normalise the whole dataset ONCE into a
    TensorDataset in RAM (all other studies). Same split, same values (same float32 ops)."""
    root = lab_path(cfg["data"]["root"])
    n_val = cfg["data"]["val_size"]
    split_gen = torch.Generator().manual_seed(cfg["data"]["split_seed"])
    if not cfg["data"]["preload"]:
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])
        full_train = datasets.FashionMNIST(root, train=True, download=True, transform=tf)
        test_set = datasets.FashionMNIST(root, train=False, download=True, transform=tf)
        train_set, val_set = random_split(full_train, [len(full_train) - n_val, n_val], generator=split_gen)
        return train_set, val_set, test_set

    def tensors(train):
        ds = datasets.FashionMNIST(root, train=train, download=True)
        x = ds.data.unsqueeze(1).float().div(255.0).sub(MEAN).div(STD)
        return x, ds.targets.clone()

    x, y = tensors(True)
    perm = torch.randperm(len(x), generator=split_gen).tolist()   # identical to random_split
    tr, va = perm[: len(x) - n_val], perm[len(x) - n_val:]
    xte, yte = tensors(False)
    return (TensorDataset(x[tr], y[tr]), TensorDataset(x[va], y[va]), TensorDataset(xte, yte))


def make_loader(ds, cfg, shuffle, generator=None, batch_size=None, num_workers=None):
    d = cfg["data"]
    nw = d["num_workers"] if num_workers is None else num_workers
    bs = batch_size or d["batch_size"]
    common = dict(num_workers=nw, pin_memory=d["pin_memory"],
                  persistent_workers=bool(d["persistent_workers"]) and nw > 0)
    if isinstance(ds, TensorDataset):
        # Fetch a whole batch with one vectorised index (tensor[list_of_indices]) instead of
        # 128 single-sample lookups + collate. RandomSampler(generator) gives exactly the same
        # permutation as DataLoader(shuffle=True, generator), so batch contents are identical.
        sampler = RandomSampler(ds, generator=generator) if shuffle else SequentialSampler(ds)
        return DataLoader(ds, sampler=BatchSampler(sampler, bs, drop_last=False), batch_size=None, **common)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, generator=generator, **common)


# ---------------------------------------------------------------- optim
def build_optimizer(model, cfg):
    o = cfg["optim"]
    if o["name"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=o["lr"], weight_decay=o["weight_decay"])
    if o["name"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=o["lr"], momentum=o["momentum"],
                               weight_decay=o["weight_decay"])
    raise ValueError(f"unknown optimizer {o['name']}")


def build_scheduler(opt, cfg):
    s = cfg["sched"]
    name = s["name"]
    if name == "none":
        return None
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=s["step_size"], gamma=s["gamma"])
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=s["t_max"] or cfg["train"]["epochs"])
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=s["factor"],
                                                          patience=s["patience"])
    raise ValueError(f"unknown scheduler {name}")


# ---------------------------------------------------------------- train / eval
@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    loss_sum = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += loss_fn(out, y).item() * y.numel()
        correct += (out.argmax(1) == y).sum().item()
        total += y.numel()
    return loss_sum / total, correct / total


def train_one_epoch(model, loader, opt, loss_fn, device, cfg, epoch, global_step, rss_guard=None):
    """One pass over the training set. Returns (stats dict, step-loss log, new global_step)."""
    t = cfg["train"]
    accum = t["grad_accum_steps"]
    n_batches = len(loader)
    if t["max_steps_per_epoch"]:
        n_batches = min(n_batches, t["max_steps_per_epoch"] * accum)

    model.train()
    opt.zero_grad(set_to_none=True)
    loss_sum = correct = seen = 0
    # manual phase timers over the whole epoch; with train.sync_timers on a GPU we synchronise
    # before reading the clock, otherwise asynchronous kernels would be billed to the next phase
    sync = (lambda: torch.cuda.synchronize()) if (device.type == "cuda" and t["sync_timers"]) else (lambda: None)
    phase = {"data_loading": 0.0, "forward": 0.0, "backward": 0.0, "optimizer_step": 0.0}
    step_log, group_losses = [], []

    def tick():
        sync()
        return time.perf_counter()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()   # training-only peak (eval at batch 512 excluded)
    t0 = tick()
    it = iter(loader)  # with num_workers>0 this spawns the worker processes
    loader_start_s = tick() - t0
    first_batch_s = None
    for i in range(n_batches):
        if rss_guard is not None:
            rss_guard.check()
        ta = tick()
        with record_function("data_loading"):
            x, y = next(it)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        tb = tick()
        phase["data_loading"] += tb - ta
        if first_batch_s is None:
            first_batch_s = tb - t0

        # micro-batches in the current accumulation group (the last group may be shorter)
        group_size = min(accum, n_batches - (i // accum) * accum)
        with record_function("forward"):
            out = model(x)
            loss = loss_fn(out, y)
        tc = tick()
        phase["forward"] += tc - tb
        with record_function("backward"):
            (loss / group_size).backward()
        td = tick()
        phase["backward"] += td - tc
        group_losses.append(loss.item())

        if (i + 1) % accum == 0 or i + 1 == n_batches:
            te = tick()
            with record_function("optimizer_step"):
                opt.step()
                opt.zero_grad(set_to_none=True)
            phase["optimizer_step"] += tick() - te
            global_step += 1
            if global_step % t["log_every"] == 0:
                step_log.append({"step": global_step, "epoch": epoch - 1 + (i + 1) / n_batches,
                                 "train_loss": float(np.mean(group_losses))})
            group_losses = []

        loss_sum += loss.item() * y.numel()
        correct += (out.argmax(1) == y).sum().item()
        seen += y.numel()

    if device.type == "cuda":
        torch.cuda.synchronize()
    epoch_time = time.perf_counter() - t0
    stats = {
        "train_loss": loss_sum / seen,
        "train_acc": correct / seen,
        "epoch_time_s": epoch_time,
        "data_time_s": phase["data_loading"],
        "compute_time_s": epoch_time - phase["data_loading"],
        "forward_s": phase["forward"],
        "backward_s": phase["backward"],
        "optimizer_s": phase["optimizer_step"],
        "loader_start_s": loader_start_s,          # iter(loader): worker start-up (0 for num_workers=0)
        "other_s": epoch_time - sum(phase.values()) - loader_start_s,
        "train_peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0,
        "first_batch_s": first_batch_s,
        "throughput_sps": seen / epoch_time,
        "samples": seen,
    }
    return stats, step_log, global_step


def save_checkpoint(path, model, opt, sched, epoch, global_step, history, step_history, cfg, mlflow_run_id):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict() if sched is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "rng": get_rng_state(),
        "history": history,
        "step_history": step_history,
        "config": cfg,
        "mlflow_run_id": mlflow_run_id,
    }, path)


def plot_history(history, step_history, path, title):
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    if step_history:
        ax[0].plot([s["epoch"] for s in step_history], [s["train_loss"] for s in step_history],
                   alpha=0.4, lw=0.8, label="train (per step)")
    ax[0].plot(ep, [h["train_loss"] for h in history], "o-", label="train (epoch)")
    ax[0].plot(ep, [h["val_loss"] for h in history], "s-", label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, [h["train_acc"] * 100 for h in history], "o-", label="train")
    ax[1].plot(ep, [h["val_acc"] * 100 for h in history], "s-", label="val")
    ax[1].set_title("Accuracy (%)"); ax[1].set_xlabel("epoch"); ax[1].legend()
    ax[2].plot(ep, [h["lr"] for h in history], "o-")
    ax[2].set_title("Learning rate"); ax[2].set_xlabel("epoch")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def profile_tables(prof, epoch_wall_s, row_limit):
    """Split the profiled epoch into the four record_function labels."""
    events = {e.key: e for e in prof.key_averages()}
    rows = []
    for lab in PROFILE_LABELS:
        e = events.get(lab)
        rows.append({"phase": lab, "calls": e.count if e is not None else 0,
                     "time_s": (e.cpu_time_total / 1e6) if e is not None else 0.0})
    labelled = sum(r["time_s"] for r in rows)
    rows.append({"phase": "other (loop overhead, metrics, profiler)", "calls": 0,
                 "time_s": max(epoch_wall_s - labelled, 0.0)})
    for r in rows:
        r["share_pct"] = 100 * r["time_s"] / epoch_wall_s
    top_ops = []
    for e in sorted(prof.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True):
        if e.key in PROFILE_LABELS or e.key.startswith("ProfilerStep"):
            continue
        top_ops.append({"op": e.key, "calls": e.count, "self_cpu_s": e.self_cpu_time_total / 1e6,
                        "self_share_pct": 100 * e.self_cpu_time_total / 1e6 / epoch_wall_s})
        if len(top_ops) >= row_limit:
            break
    return rows, top_ops


# ---------------------------------------------------------------- main loop
def run(cfg):
    cfg = copy.deepcopy(cfg)
    name = cfg["run"]["name"]
    seed_everything(cfg["run"]["seed"], cfg["run"]["deterministic"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hw = hardware_info(device)
    print(f"[{name}] device={device} threads={torch.get_num_threads()}", flush=True)

    results_dir = lab_path(cfg["outputs"]["results_dir"]) / name
    artifacts_dir = lab_path(cfg["outputs"]["artifacts_dir"]) / name
    ckpt_dir = lab_path(cfg["checkpoint"]["dir"]) / name
    for d in (results_dir, artifacts_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_set, val_set, test_set = build_datasets(cfg)
    loader_gen = torch.Generator()
    train_loader = make_loader(train_set, cfg, shuffle=True, generator=loader_gen)
    # evaluation loads in the main process: data.num_workers only applies to the training loader.
    # (Persistent eval workers would triple the worker count; on Windows each worker re-imports
    # CUDA torch, and with num_workers=8 this pushed free RAM down to 1.1 GB.)
    val_loader = make_loader(val_set, cfg, shuffle=False, batch_size=512, num_workers=0)
    test_loader = make_loader(test_set, cfg, shuffle=False, batch_size=512, num_workers=0)

    model = MODELS[cfg["model"]["name"]]().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg)
    loss_fn = nn.CrossEntropyLoss()

    start_epoch, global_step = 1, 0
    history, step_history = [], []
    mlflow_run_id = None
    resume = cfg["train"]["resume_from"]
    if resume:
        # load to CPU: RNG states must stay CPU ByteTensors; load_state_dict moves weights/optimizer
        # state to the model's device
        ckpt = torch.load(lab_path(resume), map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        if sched is not None and ckpt["scheduler"] is not None:
            sched.load_state_dict(ckpt["scheduler"])
        set_rng_state(ckpt["rng"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        history, step_history = ckpt["history"], ckpt["step_history"]
        mlflow_run_id = ckpt["mlflow_run_id"]
        print(f"[{name}] resumed from {resume} at epoch {start_epoch}", flush=True)

    # ---- MLflow
    ml = None
    if cfg["mlflow"]["enabled"]:
        import mlflow
        ml = mlflow
        ml.set_tracking_uri(lab_path(cfg["mlflow"]["tracking_uri"]).as_uri())
        ml.set_experiment(cfg["run"]["experiment"])
        ml.start_run(run_id=mlflow_run_id, run_name=None if mlflow_run_id else name)
        mlflow_run_id = ml.active_run().info.run_id
        if resume:
            ml.set_tag("resumed_at_epoch", start_epoch)
        else:
            ml.log_params({k: ("null" if v is None else v) for k, v in flatten(cfg).items()})
            ml.log_params({"n_params": n_params, "device": str(device),
                           "torch_threads": torch.get_num_threads()})
            ml.set_tags({"study": name.split("_")[0], "cpu": hw["processor"]})

    rss_before = rss_mb()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    epochs = cfg["train"]["epochs"]
    stop_after = cfg["train"]["stop_after_epoch"]
    interrupted = False
    profile_rows = None
    sampler = RssSampler(track_children=cfg["data"]["num_workers"] > 0,
                         limit_mb=cfg["run"]["rss_limit_mb"]).__enter__()
    for epoch in range(start_epoch, epochs + 1):
        loader_gen.manual_seed(cfg["run"]["seed"] + epoch)  # data order depends only on (seed, epoch)
        if cfg["profile"]["enabled"] and epoch == start_epoch:
            # CPU activity only by default: CUPTI (CUDA activity) buffers made a 150-step trace use ~6 GB.
            # With train.sync_timers the GPU time still lands inside the labelled CPU ranges.
            acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if (device.type == "cuda" and
                                                                        cfg["profile"]["cuda_activity"]) else [])
            # operator-level evidence only: run with train.max_steps_per_epoch (~150 steps) so the
            # event buffer stays small; the full-epoch phase table comes from the manual timers
            with profile(activities=acts, record_shapes=False, profile_memory=False, with_stack=False) as prof:
                stats, steps, global_step = train_one_epoch(model, train_loader, opt, loss_fn, device,
                                                            cfg, epoch, global_step, sampler)
            profile_rows, top_ops = profile_tables(prof, stats["epoch_time_s"], cfg["profile"]["row_limit"])
            write_csv(results_dir / "profile_breakdown.csv", profile_rows)
            write_csv(results_dir / "profile_top_ops.csv", top_ops)
            (artifacts_dir / "profiler_table.txt").write_text(
                prof.key_averages().table(sort_by="cpu_time_total", row_limit=30), encoding="utf-8")
        else:
            stats, steps, global_step = train_one_epoch(model, train_loader, opt, loss_fn, device,
                                                        cfg, epoch, global_step, sampler)
        val_loss, val_acc = evaluate(model, val_loader, device, loss_fn)
        lr = opt.param_groups[0]["lr"]
        if sched is not None:
            if cfg["sched"]["name"] == "plateau":
                sched.step(val_loss)
            else:
                sched.step()
        row = {"epoch": epoch, **stats, "val_loss": val_loss, "val_acc": val_acc, "lr": lr,
               "resumed": bool(resume)}
        history.append(row)
        step_history.extend(steps)
        print(f"[{name}] epoch {epoch}/{epochs} train_loss {stats['train_loss']:.4f} "
              f"val_acc {val_acc:.4f} lr {lr:.2e} time {stats['epoch_time_s']:.1f}s "
              f"({stats['throughput_sps']:.0f} samples/s)", flush=True)
        if ml:
            ml.log_metrics({k: float(v) for k, v in row.items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "epoch"},
                           step=epoch)

        if epoch % cfg["checkpoint"]["save_every"] == 0 or epoch == epochs or epoch == stop_after:
            save_checkpoint(ckpt_dir / "last.pt", model, opt, sched, epoch, global_step,
                            history, step_history, cfg, mlflow_run_id)
        if stop_after and epoch == stop_after:
            interrupted = True
            print(f"[{name}] simulated interruption after epoch {epoch}", flush=True)
            break

    sampler.__exit__()
    # GPU: peak allocated by the training loop (max over epochs); CPU: peak RSS during training
    peak_mem = max(h.get("train_peak_gpu_mb", 0.0) for h in history if h["epoch"] >= start_epoch)         if device.type == "cuda" else sampler.peak_main
    test_loss, test_acc = evaluate(model, test_loader, device, loss_fn)

    write_csv(results_dir / "history.csv", history)
    write_csv(results_dir / "steps.csv", step_history)
    with open(artifacts_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    plot_path = artifacts_dir / "curves.png"
    plot_history(history, step_history, plot_path, name)

    timed = [h for h in history if h["epoch"] >= start_epoch]
    summary = {
        "name": name,
        "epochs_done": history[-1]["epoch"],
        "interrupted": interrupted,
        "n_params": n_params,
        "final_val_acc": history[-1]["val_acc"],
        "best_val_acc": max(h["val_acc"] for h in history),
        "test_acc": test_acc,
        "test_loss": test_loss,
        "mean_epoch_time_s": float(np.mean([h["epoch_time_s"] for h in timed])),
        "mean_throughput_sps": float(np.mean([h["throughput_sps"] for h in timed])),
        "peak_memory_mb": peak_mem,
        "memory_kind": "cuda.max_memory_allocated during training epochs (eval excluded)" if device.type == "cuda" else "peak RSS of main process during training (sampled every 50 ms)",
        "peak_rss_mb": sampler.peak_main,
        "peak_rss_incl_workers_mb": sampler.peak_total,
        "peak_system_mem_increase_mb": sampler.peak_sys_increase,
        "min_system_available_mb": sampler.min_avail_mb,
        "rss_before_train_mb": rss_before,
        "mlflow_run_id": mlflow_run_id,
        "hardware": hw,
        "config": cfg,
    }
    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if ml:
        ml.log_metrics({"test_acc": test_acc, "test_loss": test_loss, "peak_memory_mb": peak_mem})
        ml.log_artifact(str(artifacts_dir / "config.yaml"))
        ml.log_artifact(str(plot_path))
        ml.log_artifact(str(results_dir / "history.csv"))
        if profile_rows is not None:
            ml.log_artifact(str(results_dir / "profile_breakdown.csv"))
            ml.log_artifact(str(artifacts_dir / "profiler_table.txt"))
        if cfg["mlflow"]["log_checkpoint"]:
            ml.log_artifact(str(ckpt_dir / "last.pt"), artifact_path="checkpoint")
        ml.end_run()
    print(f"[{name}] done: test_acc {test_acc:.4f}, peak mem {peak_mem:.0f} MB, "
          f"peak RSS {sampler.peak_total:.0f} MB", flush=True)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(LAB_DIR / "configs" / "base.yaml"))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set data.batch_size=256")
    args = ap.parse_args(argv)
    run(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
