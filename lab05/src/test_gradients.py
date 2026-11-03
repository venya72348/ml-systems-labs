"""
Task 4: verify the engine against PyTorch autograd and numerical gradients on
    ((a*b + c)**2).relu().sum()
Requirement: max absolute difference < 1e-6.

Also checks each op individually and the gradient-accumulation case (a node
used more than once in the graph).
"""
import json
import os
from pathlib import Path
import random

import torch

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

from engine import Value, vsum

TOL = 1e-6
N = 6          # length of the a, b, c vectors
EPS = 1e-6     # step for central finite differences
SEED = 0

torch.set_default_dtype(torch.float64)


def f_engine(a, b, c):
    """((a*b + c)**2).relu().sum() on lists of Value."""
    return vsum([((ai * bi + ci) ** 2).relu() for ai, bi, ci in zip(a, b, c)])


def f_float(a, b, c):
    """Same function on plain floats (for numerical gradients)."""
    return sum(max((ai * bi + ci) ** 2, 0.0) for ai, bi, ci in zip(a, b, c))


def numerical_grads(a, b, c):
    """Central differences: (f(x+eps) - f(x-eps)) / (2 eps) for every input."""
    inputs = [list(a), list(b), list(c)]
    grads = []
    for vec in inputs:
        g = []
        for i in range(len(vec)):
            orig = vec[i]
            vec[i] = orig + EPS
            fp = f_float(*inputs)
            vec[i] = orig - EPS
            fm = f_float(*inputs)
            vec[i] = orig
            g.append((fp - fm) / (2 * EPS))
        grads.append(g)
    return grads


def main_check():
    rng = random.Random(SEED)
    a_f = [rng.uniform(-2, 2) for _ in range(N)]
    b_f = [rng.uniform(-2, 2) for _ in range(N)]
    c_f = [rng.uniform(-2, 2) for _ in range(N)]

    # ours
    a = [Value(x, label=f"a{i}") for i, x in enumerate(a_f)]
    b = [Value(x, label=f"b{i}") for i, x in enumerate(b_f)]
    c = [Value(x, label=f"c{i}") for i, x in enumerate(c_f)]
    out = f_engine(a, b, c)
    out.backward()
    ours = [[v.grad for v in a], [v.grad for v in b], [v.grad for v in c]]

    # PyTorch
    ta = torch.tensor(a_f, requires_grad=True)
    tb = torch.tensor(b_f, requires_grad=True)
    tc = torch.tensor(c_f, requires_grad=True)
    tout = ((ta * tb + tc) ** 2).relu().sum()
    tout.backward()
    theirs = [ta.grad.tolist(), tb.grad.tolist(), tc.grad.tolist()]

    # numerical
    num = numerical_grads(a_f, b_f, c_f)

    def maxdiff(x, y):
        return max(abs(p - q) for vx, vy in zip(x, y) for p, q in zip(vx, vy))

    res = {
        "function": "((a*b + c)**2).relu().sum()",
        "n_elements": N,
        "forward_engine": out.data,
        "forward_torch": tout.item(),
        "forward_abs_diff": abs(out.data - tout.item()),
        "max_abs_diff_vs_torch": maxdiff(ours, theirs),
        "max_abs_diff_vs_numerical": maxdiff(ours, num),
        "graph_nodes": len(out.topo_order()),
    }
    print("=== Main check: ((a*b + c)**2).relu().sum() ===")
    print(f"forward  engine={out.data:.12f}  torch={tout.item():.12f}")
    print(f"graph nodes built during forward: {res['graph_nodes']}")
    for name, o, t, n in zip("abc", ours, theirs, num):
        print(f"\nd/d{name}")
        print(f"  {'engine':>14} {'torch':>14} {'numerical':>14}")
        for oi, ti, ni in zip(o, t, n):
            print(f"  {oi:14.8f} {ti:14.8f} {ni:14.8f}")
    print(f"\nmax |engine - torch|     = {res['max_abs_diff_vs_torch']:.3e}")
    print(f"max |engine - numerical| = {res['max_abs_diff_vs_numerical']:.3e}")
    assert res["max_abs_diff_vs_torch"] < TOL, "mismatch vs PyTorch"
    assert res["max_abs_diff_vs_numerical"] < TOL, "mismatch vs numerical"
    print(f"PASS (both < {TOL:g})")
    return res


def op_checks():
    """Each op on its own vs PyTorch, plus reused-node accumulation."""
    cases = {
        "add":       (lambda x, y: x + y),
        "mul":       (lambda x, y: x * y),
        "pow3":      (lambda x, y: x ** 3 + y),
        "relu_pos":  (lambda x, y: (x * y).relu()),
        "relu_neg":  (lambda x, y: (x * -y).relu() + y),
        "sub_div":   (lambda x, y: (x - y) / y),
        "sigmoid":   (lambda x, y: (x * y).sigmoid()),
        "exp_log":   (lambda x, y: (x * x).exp().log() + y),
        # accumulation: x appears 3 times, y twice -> grads must be summed
        "reuse":     (lambda x, y: x * x + x * y + y),
    }
    x0, y0 = 1.3, 0.7
    out = {}
    print("\n=== Per-op checks vs PyTorch ===")
    for name, fn in cases.items():
        x, y = Value(x0), Value(y0)
        r = fn(x, y)
        r.backward()
        tx = torch.tensor(x0, requires_grad=True)
        ty = torch.tensor(y0, requires_grad=True)
        tr = fn(tx, ty)
        tr.backward()
        d = max(abs(x.grad - tx.grad.item()), abs(y.grad - ty.grad.item()),
                abs(r.data - tr.item()))
        out[name] = d
        print(f"  {name:10s} max abs diff = {d:.2e}  {'OK' if d < TOL else 'FAIL'}")
        assert d < TOL, name

    # calling backward on a graph twice must keep accumulating (like torch)
    x = Value(2.0)
    y = x * x
    y.backward()
    first = x.grad
    y.backward()  # note: out.grad is re-seeded to 1, x.grad accumulates
    print(f"  accumulate across two backward() calls: {first} -> {x.grad} (expected 4 -> 8)")
    assert first == 4.0 and x.grad == 8.0
    return out


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    res = main_check()
    res["per_op_max_abs_diff"] = op_checks()
    with open(RESULTS_DIR / "gradcheck.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print("\nSaved results/gradcheck.json")
