"""Lab 7 shared code: data, model family, training loop (optional mask / distillation),
evaluation, latency, size, MACs and energy measurement.

Imported by the five task scripts in this folder; not meant to be run directly.
"""
import contextlib
import copy
import io
import json
import platform
import random
import statistics
import time
import zipfile
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

LAB_DIR = Path(__file__).resolve().parents[1]
RES = LAB_DIR / "results"
MODELS_DIR = LAB_DIR / "models"          # small final models kept in git (< 5 MB)
CKPT = LAB_DIR / "checkpoints"           # every pruned variant (git-ignored)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_TDP_W = 75.0          # GTX 1050 Ti board power limit (upper-bound estimate)


def load_cfg():
    with open(LAB_DIR / "configs" / "lab07.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def hardware_info():
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_logical": psutil.cpu_count(),
        "cpu_physical": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": str(DEVICE),
    }


def save_json(obj, name):
    RES.mkdir(parents=True, exist_ok=True)
    (RES / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def read_json(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- data (pre-loaded tensors)
def load_data(cfg, device=None):
    """Fashion-MNIST as normalised float tensors held in RAM (same split as Lab 3/6).

    Pre-loading once instead of decoding PIL images every epoch is the 'data' efficiency
    lever measured in Lab 6 (the per-sample transform was a large part of the epoch there)."""
    from torchvision import datasets
    root = (LAB_DIR / cfg["data"]["root"]).resolve()
    tr = datasets.FashionMNIST(root, train=True, download=True)
    te = datasets.FashionMNIST(root, train=False, download=True)

    def prep(ds):
        x = ds.data.float().div(255.0).sub(0.2860).div(0.3530).unsqueeze(1)
        return x, ds.targets.clone()

    x, y = prep(tr)
    n_val = cfg["data"]["val_size"]
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(cfg["data"]["split_seed"]))
    tr_idx, va_idx = perm[: len(x) - n_val], perm[len(x) - n_val:]  # identical to random_split
    xte, yte = prep(te)
    dev = DEVICE if device is None else device   # whole dataset (~190 MB) lives on the GPU when training there
    return {"train": (x[tr_idx].to(dev), y[tr_idx].to(dev)), "val": (x[va_idx].to(dev), y[va_idx].to(dev)),
            "test": (xte.to(dev), yte.to(dev))}


def batches(x, y, bs, shuffle, seed=None, extra=None):
    n = len(x)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed)) if shuffle else torch.arange(n)
    idx = idx.to(x.device)
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        if extra is None:
            yield x[j], y[j]
        else:
            yield x[j], y[j], extra[j]


# ---------------------------------------------------------------- model family
class Net(nn.Module):
    """conv(1->c1) -> conv(c1->c2) -> pool -> conv(c2->c3) -> pool -> fc(c3*7*7 -> h) -> fc(h -> 10).

    widths=[32,64,128,144] is the ~1M-param teacher, [16,32,64,24] the ~100k student.
    Structured pruning produces other widths of the same family (physically smaller layers)."""

    def __init__(self, widths):
        super().__init__()
        c1, c2, c3, h = widths
        self.widths = list(widths)
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1)
        self.fc1 = nn.Linear(c3 * 7 * 7, h)
        self.fc2 = nn.Linear(h, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # 28 -> 14
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)   # 14 -> 7
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)


PRUNABLE = ["conv1", "conv2", "conv3", "fc1", "fc2"]


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def n_nonzero(model):
    return sum(int((p != 0).sum()) for p in model.parameters())


def load_model(path):
    ck = torch.load(path, map_location=DEVICE)
    m = Net(ck["widths"]).to(DEVICE)
    m.load_state_dict(ck["state_dict"])
    return m


def save_model(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"widths": model.widths, "state_dict": model.state_dict()}, path)


# ---------------------------------------------------------------- train / eval
@torch.no_grad()
def evaluate(model, x, y, bs=1000):
    model.eval()
    correct = 0
    loss = 0.0
    for xb, yb in batches(x, y, bs, shuffle=False):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        out = model(xb)
        loss += F.cross_entropy(out, yb, reduction="sum").item()
        correct += (out.argmax(1) == yb).sum().item()
    return correct / len(x), loss / len(x)


@torch.no_grad()
def predict_logits(model, x, bs=1000):
    model.eval()
    return torch.cat([model(x[i:i + bs]) for i in range(0, len(x), bs)])   # stays on x.device


_KL = nn.KLDivLoss(reduction="batchmean")


def kd_loss(student_logits, teacher_logits, labels, T, alpha):
    """alpha * CE(student, y) + (1 - alpha) * T^2 * KL(p_teacher^T || p_student^T).

    nn.KLDivLoss takes the *log*-probabilities of the model being trained as `input` and the
    *probabilities* of the target distribution as `target`, and returns
    sum p_t * (log p_t - log p_s) = KL(teacher || student) - the Hinton et al. (2015) loss.
    'batchmean' divides the sum by the batch size, which is the correct per-sample KL
    ('mean' would also divide by the 10 classes). T^2 keeps the gradient scale of the soft
    term comparable to the hard term (soft-target gradients shrink as 1/T^2)."""
    ce = F.cross_entropy(student_logits, labels)
    soft = _KL(F.log_softmax(student_logits / T, dim=1), F.softmax(teacher_logits / T, dim=1))
    return alpha * ce + (1 - alpha) * (T ** 2) * soft, ce.detach(), soft.detach()


def train(model, data, epochs, lr, bs, seed, masks=None, teacher_logits=None, T=4.0, alpha=0.5,
          cosine=True, tag=""):
    """Generic loop. masks: {layer: 0/1 tensor} kept fixed (pruned weights stay exactly zero).
    teacher_logits: pre-computed teacher outputs for the train set -> distillation loss."""
    xtr, ytr = data["train"]
    xva, yva = data["val"]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if cosine else None
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        tot = correct = 0
        loss_sum = 0.0
        for b in batches(xtr, ytr, bs, shuffle=True, seed=seed * 1000 + ep, extra=teacher_logits):
            xb, yb = b[0].to(DEVICE), b[1].to(DEVICE)
            out = model(xb)
            if teacher_logits is None:
                loss = F.cross_entropy(out, yb)
            else:
                loss, _, _ = kd_loss(out, b[2].to(DEVICE), yb, T, alpha)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if masks is not None:
                with torch.no_grad():
                    for n, m in masks.items():
                        getattr(model, n).weight.mul_(m)
            loss_sum += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            tot += len(yb)
        if sched:
            sched.step()
        va_acc, va_loss = evaluate(model, xva, yva)
        row = {"epoch": ep, "train_loss": loss_sum / tot, "train_acc": correct / tot,
               "val_loss": va_loss, "val_acc": va_acc, "epoch_time_s": time.perf_counter() - t0}
        history.append(row)
        print(f"  {tag} epoch {ep}/{epochs} loss {row['train_loss']:.4f} val_acc {va_acc:.4f} "
              f"({row['epoch_time_s']:.1f}s)", flush=True)
    return history


# ---------------------------------------------------------------- latency / size / compute
@torch.inference_mode()
def latency_ms(model, batch, runs, warmup):
    """Median wall time of one forward pass ON THE CPU (the deployment target of this lab),
    whatever device the model was trained on."""
    model = copy.deepcopy(model).cpu().eval()
    x = torch.randn(batch, 1, 28, 28)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def measure_latency(model, cfg):
    torch.set_num_threads(cfg["latency"]["threads"])
    L = cfg["latency"]
    return {"latency_b1_ms": latency_ms(model, 1, L["b1_runs"], L["b1_warmup"]),
            "latency_bbig_ms": latency_ms(model, L["big_batch"], L["big_runs"], L["big_warmup"])}


def _bytes_of_save(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def _zip_bytes(raw):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("model.pt", raw)
    return buf.getvalue()


def disk_sizes(model):
    """KB on disk of the same weights in several formats.

    dense      torch.save(state_dict)                 4 B per weight, zeros included
    dense_zip  the dense file DEFLATE-compressed      runs of zeros compress well
    coo        weights as tensor.to_sparse()          int64 index per dim per non-zero + fp32 value
    csr        2-D CSR, int32 indices                 int32 column + fp32 value per non-zero (+ row ptrs)
    csr_zip    the CSR file DEFLATE-compressed
    """
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    dense = _bytes_of_save(sd)
    coo_sd, csr_sd = {}, {}
    for k, v in sd.items():
        if k.endswith("weight"):
            coo_sd[k] = v.to_sparse()
            w2 = v.reshape(v.shape[0], -1).to_sparse_csr()
            csr_sd[k] = {"shape": tuple(v.shape), "crow": w2.crow_indices().int(),
                         "col": w2.col_indices().int(), "val": w2.values()}
        else:
            coo_sd[k] = v
            csr_sd[k] = v
    csr = _bytes_of_save(csr_sd)
    return {"size_dense_kb": len(dense) / 1024, "size_dense_zip_kb": len(_zip_bytes(dense)) / 1024,
            "size_coo_kb": len(_bytes_of_save(coo_sd)) / 1024, "size_csr_kb": len(csr) / 1024,
            "size_csr_zip_kb": len(_zip_bytes(csr)) / 1024}


def macs(model):
    """Per-sample multiply-accumulates of conv/linear layers (bias, ReLU, pool ignored).

    dense_macs counts every weight, effective_macs only non-zero weights: the work an ideal
    sparse kernel would do. A dense kernel always does dense_macs."""
    out_hw = {}
    hooks = [getattr(model, n).register_forward_hook(
        lambda m, i, o, n=n: out_hw.__setitem__(n, o.shape[2] * o.shape[3] if o.dim() == 4 else 1))
        for n in PRUNABLE]
    with torch.no_grad():
        model.eval()
        model(torch.zeros(1, 1, 28, 28, device=DEVICE))
    for h in hooks:
        h.remove()
    dense = eff = 0
    for n in PRUNABLE:
        w = getattr(model, n).weight
        dense += w.numel() * out_hw[n]
        eff += int((w != 0).sum()) * out_hw[n]
    return {"dense_macs": dense, "effective_macs": eff}


def thop_macs(model):
    """thop cross-check of dense MACs (on a fresh copy, thop adds buffers to the model)."""
    from thop import profile
    m = Net(model.widths).to(DEVICE)
    total, _ = profile(m, inputs=(torch.zeros(1, 1, 28, 28, device=DEVICE),), verbose=False)
    return int(total)


# ---------------------------------------------------------------- energy
@contextlib.contextmanager
def energy_meter(cfg, label, gpu_active=None):
    """codecarbon OfflineEmissionsTracker around a code block.

    On this laptop (Windows, no RAPL / Intel Power Gadget) codecarbon cannot read CPU power
    counters and falls back to a constant CPU power = 50% of TDP (22.5 W for the i7-8750H) plus
    a RAM power estimate, so its kWh value is effectively wall time x fixed power.
    Its NVML probe crashes on the GTX 1050 Ti (the card exposes no power readings), so GPU
    tracking is switched off - all Lab 7 work runs on the CPU anyway. If codecarbon fails
    entirely we fall back to 22.5 W x time and say so in 'energy_source'."""
    rec = {"label": label}
    gpu_active = (DEVICE.type == "cuda") if gpu_active is None else gpu_active
    tracker = None
    try:
        import logging
        import codecarbon.emissions_tracker as et
        et.gpu.is_gpu_details_available = lambda: False
        from codecarbon import OfflineEmissionsTracker
        tracker = OfflineEmissionsTracker(country_iso_code=cfg["energy"]["country_iso_code"],
                                          measure_power_secs=1, save_to_file=False, log_level="error")
        logging.getLogger("codecarbon").setLevel(logging.ERROR)
        tracker.start()
    except Exception as e:
        rec["codecarbon_error"] = repr(e)
        tracker = None
    t0 = time.perf_counter()
    try:
        yield rec
    finally:
        rec["time_s"] = time.perf_counter() - t0
        d = None
        if tracker is not None:
            try:
                tracker.stop()
                d = tracker.final_emissions_data
            except Exception as e:
                rec["codecarbon_error"] = repr(e)
        if d is not None:
            rec.update(cpu_ram_energy_kwh=d.energy_consumed, cpu_energy_kwh=d.cpu_energy,
                       ram_energy_kwh=d.ram_energy, cpu_power_w=d.cpu_power,
                       energy_source="codecarbon 2.4.2 (CPU = constant 50% TDP model, no power counters)")
        else:
            rec.update(cpu_ram_energy_kwh=22.5 * rec["time_s"] / 3.6e6,
                       energy_source="ESTIMATE: 22.5 W x wall time (codecarbon unavailable)")
        # GPU: the GTX 1050 Ti reports no power over NVML, so codecarbon cannot see it. We add a
        # clearly labelled upper-bound estimate: board power limit (75 W) x wall time.
        rec["gpu_energy_kwh_est"] = GPU_TDP_W * rec["time_s"] / 3.6e6 if gpu_active else 0.0
        rec["energy_kwh"] = rec["cpu_ram_energy_kwh"] + rec["gpu_energy_kwh_est"]
        if gpu_active:
            rec["energy_source"] += " + GPU ESTIMATE 75 W x time"
        intensity = (d.emissions / d.energy_consumed) if (d is not None and d.energy_consumed) else 0.0
        rec["grid_kg_co2_per_kwh"] = intensity      # codecarbon offline value for Poland
        rec["co2_kg"] = rec["energy_kwh"] * intensity
