"""Lab 7, task 2: magnitude pruning of the baseline.

Unstructured: rank weights by |w|, zero the smallest ones either GLOBALLY (one threshold for
all conv/linear weights) or LAYER-WISE (same sparsity in every layer), for sparsities
0..0.95; then fine-tune 3 epochs with the mask fixed (pruned weights re-zeroed after every
optimizer step, so they can never grow back).

Structured: remove whole conv filters (and fc1 hidden units) with the smallest L2 norm at
10/25/50% and physically rebuild thinner layers (the next layer loses the matching input
channels), then fine-tune 3 epochs.

    python lab07/src/prune.py --variant unstructured_global_0.9   # one variant (run_all.py does this)
    python lab07/src/prune.py --aggregate                        # tables + plot from results/pruning/*.json
Outputs: results/pruning.csv, results/pruning_layer_sparsity.csv, results/accuracy_vs_sparsity.png,
         checkpoints/<variant>.pt (git-ignored)
"""
import argparse
import copy
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from common import (CKPT, MODELS_DIR, PRUNABLE, RES, Net, energy_meter, evaluate, load_cfg, load_data,
                    load_model, n_nonzero, n_params, read_json, save_model, set_seed, train)


# ---------------------------------------------------------------- unstructured
def magnitude_masks(model, sparsity, mode):
    W = {n: getattr(model, n).weight.detach() for n in PRUNABLE}
    if sparsity == 0:
        return {n: torch.ones_like(w) for n, w in W.items()}
    if mode == "global":
        scores = torch.cat([w.abs().flatten() for w in W.values()])
        k = int(round(sparsity * scores.numel()))
        thr = torch.kthvalue(scores, k).values          # k-th smallest |w| over ALL layers
        return {n: (w.abs() > thr).float() for n, w in W.items()}
    masks = {}
    for n, w in W.items():                               # same sparsity in every layer
        k = int(round(sparsity * w.numel()))
        thr = torch.kthvalue(w.abs().flatten(), k).values if k > 0 else -1.0
        masks[n] = (w.abs() > thr).float()
    return masks


def apply_masks(model, masks):
    with torch.no_grad():
        for n, m in masks.items():
            getattr(model, n).weight.mul_(m)


def layer_sparsity(model):
    return {n: float((getattr(model, n).weight == 0).float().mean()) for n in PRUNABLE}


# ---------------------------------------------------------------- structured
def _top_by_l2(weight, keep):
    norms = weight.detach().flatten(1).norm(dim=1)       # one L2 norm per filter / neuron
    return norms.topk(keep).indices.sort().values


def structured_prune(model, ratio):
    """Return a NEW, physically smaller Net keeping the (1-ratio) highest-L2 filters per layer."""
    c1, c2, c3, h = model.widths
    keep = lambda c: max(1, int(round(c * (1 - ratio))))
    i1 = _top_by_l2(model.conv1.weight, keep(c1))
    i2 = _top_by_l2(model.conv2.weight, keep(c2))
    i3 = _top_by_l2(model.conv3.weight, keep(c3))
    ih = _top_by_l2(model.fc1.weight, keep(h))
    new = Net([len(i1), len(i2), len(i3), len(ih)]).to(model.conv1.weight.device)
    cols = (i3[:, None] * 49 + torch.arange(49, device=i3.device)).flatten()  # fc1 inputs of kept channels
    with torch.no_grad():
        new.conv1.weight.copy_(model.conv1.weight[i1]);           new.conv1.bias.copy_(model.conv1.bias[i1])
        new.conv2.weight.copy_(model.conv2.weight[i2][:, i1]);    new.conv2.bias.copy_(model.conv2.bias[i2])
        new.conv3.weight.copy_(model.conv3.weight[i3][:, i2]);    new.conv3.bias.copy_(model.conv3.bias[i3])
        new.fc1.weight.copy_(model.fc1.weight[ih][:, cols]);      new.fc1.bias.copy_(model.fc1.bias[ih])
        new.fc2.weight.copy_(model.fc2.weight[:, ih]);            new.fc2.bias.copy_(model.fc2.bias)
    return new


# ---------------------------------------------------------------- main
def variant_tags(cfg):
    P = cfg["pruning"]
    tags = [f"unstructured_{m}_{s}" for m in ["global", "layer"] for s in P["sparsities"]
            if not (m == "layer" and s == 0)]          # layer-wise s=0 == global s=0 (dense control)
    return tags + [f"structured_{r}" for r in P["structured_ratios"]]


def run_variant(cfg, tag):
    """Prune + fine-tune ONE variant (own process, called by run_all.py). Writes results/pruning/<tag>.json."""
    P = cfg["pruning"]
    data = load_data(cfg)
    base = load_model(MODELS_DIR / "baseline.pt")
    base_params = n_params(base)
    base_acc = read_json("baseline.json")["test_acc"]
    if tag.startswith("unstructured"):
        _, mode, s = tag.split("_")
        target = float(s)
        model = copy.deepcopy(base)
        masks = magnitude_masks(model, target, mode)
        apply_masks(model, masks)
        method = f"unstructured_{mode}"
    else:
        target = float(tag.split("_")[1])
        model, masks, method = structured_prune(base, target), None, "structured"
    acc_before, _ = evaluate(model, *data["test"])
    set_seed(cfg["seed"])
    with energy_meter(cfg, f"finetune_{tag}") as en:
        hist = train(model, data, P["finetune_epochs"], P["finetune_lr"], P["batch_size"], cfg["seed"],
                     masks=masks, cosine=False, tag=tag)
    acc, _ = evaluate(model, *data["test"])
    save_model(model, CKPT / f"{tag}.pt")
    nz = n_nonzero(model)
    removed = 1 - (nz if masks is not None else n_params(model)) / base_params
    row = {"variant": tag, "method": method, "target": target, "params": n_params(model),
           "nonzero_params": nz, "param_sparsity": removed, "test_acc_before_ft": acc_before,
           "test_acc": acc, "val_acc": hist[-1]["val_acc"], "acc_drop_pp": 100 * (base_acc - acc),
           "ft_time_s": en["time_s"], "ft_energy_kwh": en["energy_kwh"],
           "widths": "x".join(map(str, model.widths)), "layer_sparsity": layer_sparsity(model),
           "history": hist, "energy": en}
    (RES / "pruning").mkdir(parents=True, exist_ok=True)
    (RES / "pruning" / f"{tag}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print({k: v for k, v in row.items() if k not in ("history", "energy")}, flush=True)


def aggregate(cfg):
    base_acc = read_json("baseline.json")["test_acc"]
    rows, layer_rows = [], []
    for tag in variant_tags(cfg):
        r = json.loads((RES / "pruning" / f"{tag}.json").read_text(encoding="utf-8"))
        rows.append({k: v for k, v in r.items() if k not in ("history", "energy", "layer_sparsity")})
        layer_rows.append({"variant": tag, **r["layer_sparsity"]})
        if tag == "unstructured_global_0.0":       # also list it as the layer-wise s=0 point
            rows.append({**rows[-1], "variant": "unstructured_layer_0.0", "method": "unstructured_layer"})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "pruning.csv", index=False)
    pd.DataFrame(layer_rows).to_csv(RES / "pruning_layer_sparsity.csv", index=False)
    plot(df, base_acc)
    print(df.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", help="one variant tag, e.g. unstructured_global_0.9 or structured_0.25")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg()
    if args.list:
        print(*variant_tags(cfg), sep="\n")
    elif args.aggregate:
        aggregate(cfg)
    elif args.variant:
        run_variant(cfg, args.variant)
    else:
        for tag in variant_tags(cfg):
            run_variant(cfg, tag)
        aggregate(cfg)


def plot(df, base_acc):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    styles = {"unstructured_global": ("o-", "C0", "unstructured, global"),
              "unstructured_layer": ("s-", "C1", "unstructured, layer-wise"),
              "structured": ("^-", "C2", "structured (filters removed)")}
    for a, ylim in [(ax[0], None), (ax[1], (base_acc * 100 - 4, base_acc * 100 + 1))]:
        for m, (st, c, lab) in styles.items():
            d = df[df["method"] == m].sort_values("param_sparsity")
            if m == "structured":
                d = pd.concat([df[df["variant"] == "unstructured_global_0.0"], d])
            a.plot(d["param_sparsity"] * 100, d["test_acc"] * 100, st, color=c, label=f"{lab}, fine-tuned")
            a.plot(d["param_sparsity"] * 100, d["test_acc_before_ft"] * 100, st.replace("-", ":"),
                   color=c, alpha=0.45, label=f"{lab}, before fine-tune")
        a.axhline(base_acc * 100, color="gray", ls="--", lw=1, label="baseline")
        a.set_xlabel("weights removed / zero (% of baseline params)")
        a.set_ylabel("test accuracy (%)")
        a.grid(alpha=0.3)
        if ylim:
            a.set_ylim(*ylim)
            a.set_title("Zoom: within 4 pp of baseline")
        else:
            a.set_title("Accuracy vs sparsity")
    for _, r in df[df["method"] == "structured"].iterrows():
        ax[0].annotate(f"{int(r['target']*100)}% filters", (r["param_sparsity"] * 100, r["test_acc"] * 100),
                       textcoords="offset points", xytext=(-10, -14), fontsize=7, color="C2")
    ax[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(RES / "accuracy_vs_sparsity.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
