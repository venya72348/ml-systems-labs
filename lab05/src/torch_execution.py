"""
Task 6: execution modes and tensor layout in PyTorch.

  A. Inspect the graph of a small model (autograd graph via grad_fn, and the
     forward graph via torch.fx).
  B. Eager vs torch.compile vs torch.jit.trace, timed after warm-up.
     If torch.compile is unsupported on this machine (e.g. Windows without a
     C++ toolchain), it is reported and tracing is used.
  C. Tensor shape / dtype / stride / contiguity, and how a non-contiguous view
     changes operation speed.
"""
import json
import os
from pathlib import Path
import platform
import statistics
import time
import warnings

import torch

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
import torch.nn as nn

warnings.filterwarnings("ignore")
torch.manual_seed(0)
RESULTS = {"torch_version": torch.__version__, "platform": platform.platform(),
           "threads": torch.get_num_threads()}


class SmallMLP(nn.Module):
    def __init__(self, d_in=2, d_hidden=8, d_out=1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))


class CheapPointwiseChain(nn.Module):
    """Cheap pointwise ops only (add/mul/relu): memory-bound. Eager runs ~11
    separate kernels, each reading/writing the whole tensor; a compiler fuses
    them into one pass over memory."""
    def forward(self, x):
        h = x * 2.0 + 1.0
        h = h * h - x
        h = torch.relu(h) * 0.5 + x * 0.25
        return h - 3.0


class PointwiseChain(nn.Module):
    """Transcendental pointwise ops (tanh/sigmoid/gelu): compute-bound on CPU."""
    def forward(self, x):
        h = torch.relu(x) * 0.5 + 1.0
        h = torch.tanh(h) * torch.sigmoid(h)
        h = h * h - h / 3.0
        return torch.nn.functional.gelu(h) + x


class ElementwiseHeavy(nn.Module):
    """Linear followed by a chain of pointwise ops -> a good fusion target."""
    def __init__(self, d=1024):
        super().__init__()
        self.fc = nn.Linear(d, d)

    def forward(self, x):
        h = self.fc(x)
        h = torch.relu(h) * 0.5 + 1.0
        h = torch.tanh(h) * torch.sigmoid(h)
        h = h * h - h / 3.0
        return torch.nn.functional.gelu(h) + x


# --------------------------------------------------------------------- A. graph
def inspect_graph():
    print("=== A. Graph of SmallMLP (2 -> 8 -> 1) ===")
    model = SmallMLP()
    x = torch.randn(4, 2)
    y = model(x).sum()

    print("\nAutograd (backward) graph, walking grad_fn from the output:")
    lines, seen = [], set()

    def walk(fn, depth=0):
        if fn is None or fn in seen:
            return
        seen.add(fn)
        lines.append("  " * depth + type(fn).__name__)
        for nxt, _ in fn.next_functions:
            walk(nxt, depth + 1)

    walk(y.grad_fn)
    print("\n".join("  " + l for l in lines))

    print("\nForward graph via torch.fx.symbolic_trace:")
    gm = torch.fx.symbolic_trace(model)
    fx_nodes = [(n.op, getattr(n.target, "__name__", str(n.target))) for n in gm.graph.nodes]
    for op, tgt in fx_nodes:
        print(f"  {op:14s} {tgt}")
    print("\nGenerated code:")
    print("  " + gm.code.strip().replace("\n", "\n  "))
    RESULTS["autograd_graph"] = lines
    RESULTS["fx_graph"] = fx_nodes


# --------------------------------------------------------------------- timing
def bench(fn, x, warmup=10, iters=None, repeats=5):
    """Median over `repeats` of the mean time per call (ms), after warm-up."""
    with torch.inference_mode():
        for _ in range(warmup):
            fn(x)
        if iters is None:  # aim for ~0.2 s per repeat
            t0 = time.perf_counter(); fn(x); one = time.perf_counter() - t0
            iters = max(5, min(500, int(0.2 / max(one, 1e-6))))
        per_rep = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for _ in range(iters):
                fn(x)
            per_rep.append((time.perf_counter() - t0) / iters * 1e3)
    return statistics.median(per_rep)


def try_compile(model, x):
    try:
        cm = torch.compile(model)
        t0 = time.perf_counter()
        with torch.inference_mode():
            cm(x)  # first call triggers compilation
        return cm, time.perf_counter() - t0, None
    except Exception as e:  # unsupported backend / missing compiler
        return None, None, f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"


# ----------------------------------------------------------- B. eager vs compiled
def eager_vs_compiled():
    print("\n=== B. Eager vs torch.compile vs torch.jit.trace (inference, after warm-up) ===")
    configs = [
        ("SmallMLP 2-8-1, batch 32", SmallMLP().eval(), torch.randn(32, 2)),
        ("SmallMLP 2-8-1, batch 4096", SmallMLP().eval(), torch.randn(4096, 2)),
        ("Cheap pointwise chain (memory-bound), 4096x4096", CheapPointwiseChain().eval(), torch.randn(4096, 4096)),
        ("Transcendental pointwise chain (compute-bound), 2048x2048", PointwiseChain().eval(), torch.randn(2048, 2048)),
        ("Linear+pointwise chain d=1024, batch 256", ElementwiseHeavy().eval(), torch.randn(256, 1024)),
    ]
    rows = []
    for name, model, x in configs:
        eager_ms = bench(model, x)

        traced = torch.jit.trace(model, x)
        traced = torch.jit.freeze(traced)
        trace_ms = bench(traced, x)

        cm, compile_s, err = try_compile(model, x)
        comp_ms = bench(cm, x) if cm is not None else None
        if cm is not None:
            with torch.inference_mode():
                ok = torch.allclose(cm(x), model(x), atol=1e-5)
        else:
            ok = None

        row = {"config": name, "eager_ms": eager_ms, "jit_trace_ms": trace_ms,
               "compile_ms": comp_ms, "compile_first_call_s": compile_s,
               "compile_error": err, "compile_matches_eager": ok}
        rows.append(row)
        print(f"\n{name}")
        print(f"  eager        {eager_ms:9.4f} ms/call")
        print(f"  jit.trace    {trace_ms:9.4f} ms/call  ({eager_ms / trace_ms:4.2f}x vs eager)")
        if comp_ms is not None:
            print(f"  torch.compile{comp_ms:9.4f} ms/call  ({eager_ms / comp_ms:4.2f}x vs eager), "
                  f"first call (compilation) {compile_s:.2f}s, output matches eager: {ok}")
        else:
            print(f"  torch.compile unsupported here -> {err}")
    RESULTS["eager_vs_compiled"] = rows


# ---------------------------------------------------- C. layout / contiguity
def describe(name, t):
    d = {"shape": list(t.shape), "dtype": str(t.dtype), "stride": list(t.stride()),
         "contiguous": t.is_contiguous(), "shares_storage_with_x":
         t.untyped_storage().data_ptr() == X_PTR}
    print(f"  {name:22s} shape={d['shape']} dtype={d['dtype']} stride={d['stride']} "
          f"contiguous={d['contiguous']} same_storage={d['shares_storage_with_x']}")
    return d


def time_op(fn, iters=20, repeats=5):
    fn()
    reps = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        reps.append((time.perf_counter() - t0) / iters * 1e3)
    return statistics.median(reps)


def contiguity():
    global X_PTR
    print("\n=== C. Tensor layout and non-contiguous views ===")
    n = 4096
    x = torch.randn(n, n, dtype=torch.float32)
    X_PTR = x.untyped_storage().data_ptr()
    xt = x.t()                    # view: same memory, swapped strides
    xs = x[:, ::2]                # strided slice view
    xtc = xt.contiguous()         # real copy in row-major order
    layouts = {
        "x": describe("x", x),
        "x.t()": describe("x.t()", xt),
        "x[:, ::2]": describe("x[:, ::2]", xs),
        "x.t().contiguous()": describe("x.t().contiguous()", xtc),
    }

    out = torch.empty_like(x)
    ops = {
        "row sum  .sum(dim=1)": (lambda: x.sum(dim=1), lambda: xt.sum(dim=1), lambda: xtc.sum(dim=1)),
        "elementwise  a*2+1": (lambda: torch.add(x * 2, 1, out=out),
                               lambda: torch.add(xt * 2, 1, out=out),
                               lambda: torch.add(xtc * 2, 1, out=out)),
        "row-wise softmax": (lambda: torch.softmax(x, dim=1), lambda: torch.softmax(xt, dim=1),
                             lambda: torch.softmax(xtc, dim=1)),
        "cumsum(dim=1)": (lambda: torch.cumsum(x, dim=1), lambda: torch.cumsum(xt, dim=1),
                          lambda: torch.cumsum(xtc, dim=1)),
    }
    rows = []
    print(f"\n  {'op on 4096x4096 fp32':24s} {'contiguous':>12s} {'x.t() view':>12s} "
          f"{'slowdown':>9s} {'after .contiguous()':>20s}")
    for name, (f_c, f_nc, f_cc) in ops.items():
        tc, tnc, tcc = time_op(f_c), time_op(f_nc), time_op(f_cc)
        rows.append({"op": name, "contiguous_ms": tc, "noncontiguous_view_ms": tnc,
                     "after_contiguous_copy_ms": tcc, "slowdown": tnc / tc})
        print(f"  {name:24s} {tc:10.2f}ms {tnc:10.2f}ms {tnc / tc:8.2f}x {tcc:18.2f}ms")
    copy_ms = time_op(lambda: xt.contiguous())
    print(f"\n  cost of the copy itself  x.t().contiguous(): {copy_ms:.2f} ms")
    RESULTS["layouts"] = layouts
    RESULTS["noncontiguous_ops"] = rows
    RESULTS["contiguous_copy_ms"] = copy_ms


if __name__ == "__main__":
    print(f"torch {torch.__version__} on {platform.system()}, threads={torch.get_num_threads()}")
    inspect_graph()
    eager_vs_compiled()
    contiguity()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_DIR / "torch_execution.json", "w") as fh:
        json.dump(RESULTS, fh, indent=2, default=str)
    print("\nSaved results/torch_execution.json")
