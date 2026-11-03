# Lab 5: Frameworks and automatic differentiation

Scalar autograd engine from scratch, validated against PyTorch, plus a study of
PyTorch execution modes (eager / `torch.jit.trace` / `torch.compile`) and tensor layout.

## Files

| File | Task | What it does |
|---|---|---|
| `src/engine.py` | 1-3 | `Value` node + ops (`+`, `*`, `**`, `relu`, plus `exp`, `log`, `sigmoid`, `-`, `/` for BCE) + topological `backward()` |
| `src/test_gradients.py` | 4 | `((a*b + c)**2).relu().sum()` vs PyTorch autograd and central finite differences; per-op checks; accumulation checks |
| `src/train_moons.py` | 5 | 2 -> 8 -> 1 MLP on `make_moons`, sigmoid + BCE, our engine vs PyTorch, median of 5 runs |
| `src/torch_execution.py` | 6 | graph inspection, eager vs `jit.trace` vs `torch.compile`, shape/dtype/stride/contiguity, non-contiguous view timing |
| `results/` | | JSON results, console logs, loss plot |

## How to run

```bash
pip install torch numpy scikit-learn matplotlib
cd src
python test_gradients.py     # task 4, < 1 s
python train_moons.py        # task 5, ~1 min (our engine is slow on purpose)
python torch_execution.py    # task 6, ~1-2 min
```

Results below come from: Linux, 2 vCPU, Python 3.13, PyTorch 2.15 (CPU), float64 for
tasks 4-5, float32 for task 6. Absolute timings will differ on another machine; the ratios are what matter.

## Tasks 1-3: engine design

`Value` stores:

- `data` - scalar computed in the forward pass
- `grad` - d(output)/d(node), filled by `backward()`
- `_prev` - parent nodes (inputs of the op that created the node)
- `_backward` - closure with the local derivative rule (chain rule step)
- `_op` - op name, for printing/debugging

Each operator creates a new `Value` whose `_prev` points to its inputs, so the
forward pass builds the computation graph as a side effect (define-by-run, like PyTorch eager).

Local rules:

| Op | Forward | Backward (into parents) |
|---|---|---|
| `a + b` | a + b | `a.grad += out.grad`, `b.grad += out.grad` |
| `a * b` | a * b | `a.grad += b * out.grad`, `b.grad += a * out.grad` |
| `a ** n` | a^n | `a.grad += n * a^(n-1) * out.grad` |
| `relu(a)` | max(a, 0) | `a.grad += (a > 0) * out.grad` |
| `sigmoid(a)` | s = 1/(1+e^-a) | `a.grad += s(1-s) * out.grad` |
| `exp(a)`, `log(a)` | e^a, ln a | `e^a * out.grad`, `out.grad / a` |

`backward()`:

1. Topologically sort the graph (iterative DFS post-order, so no recursion limit on large graphs).
2. Set `output.grad = 1`.
3. Visit nodes in reverse topological order and call each `_backward`.
4. Every rule uses `+=`, so a node used several times (e.g. `x*x + x*y`) gets the sum of all paths.

## Task 4: gradient verification

Function: `((a*b + c)**2).relu().sum()` with `a, b, c` vectors of 6 random values (float64). Forward pass builds 49 graph nodes.

| Comparison | Max absolute difference | Required |
|---|---|---|
| engine vs PyTorch autograd | **0.0** | < 1e-6 |
| engine vs numerical (central diff, eps=1e-6) | **8.9e-10** | < 1e-6 |
| forward value, engine vs PyTorch | 0.0 | |

All 9 per-op checks (add, mul, pow, relu on both sides of 0, sub/div, sigmoid, exp/log, node reuse) match PyTorch with difference 0.
Accumulation across two `backward()` calls: `x.grad` goes 4 -> 8, same as PyTorch.
Full table: `results/gradcheck_log.txt`.

## Task 5: make_moons, our engine vs PyTorch

Setup: 200 samples (noise 0.1), 2 -> 8 (ReLU) -> 1 (sigmoid), BCE, full-batch SGD,
lr 0.5, 100 epochs. Both models start from identical weights; PyTorch limited to 1 thread for a fair comparison.

| | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | **Median** |
|---|---|---|---|---|---|---|
| Our engine (s) | 7.78 | 8.63 | 9.85 | 10.76 | 11.43 | **9.85** |
| PyTorch (s) | 0.0267 | 0.0220 | 0.0277 | 0.0259 | 0.0215 | **0.0259** |

PyTorch is about **380x faster**.

Correctness: the loss curves are identical (max |diff| over 100 epochs = 4.4e-16),
final loss 0.2457 in both, training accuracy 0.885 in both. See `results/moons_loss.png`.

Why the gap: our engine creates one Python object and one closure per scalar
operation (~16k graph nodes per epoch here), and walks them in the interpreter.
PyTorch does the same math as a few batched tensor ops (`addmm`, `relu`, `sigmoid`)
whose loops run in optimized C++/BLAS, so per-op Python overhead is paid ~10 times per epoch instead of ~16 000 times.

## Task 6: PyTorch graph, execution modes, tensor layout

### Graph of a small model (2 -> 8 -> 1)

Autograd graph from the output (`grad_fn` chain):

```
SumBackward0
  SigmoidBackward0
    AddmmBackward0          <- fc2
      AccumulateGrad        <- fc2.bias
      ReluBackward0
        AddmmBackward0      <- fc1
          AccumulateGrad    <- fc1.bias
          TBackward0
            AccumulateGrad  <- fc1.weight
      TBackward0
        AccumulateGrad      <- fc2.weight
```

Forward graph via `torch.fx.symbolic_trace`: `placeholder x -> call_module fc1 -> relu -> call_module fc2 -> sigmoid -> output`.
It is the same structure as our engine builds, only at tensor granularity (a whole `Linear` is one `addmm` node instead of 16 + 8 scalar nodes).

### Eager vs compiled (inference, after warm-up, median of 5)

| Workload | Eager (ms) | `jit.trace` (ms) | `torch.compile` (ms) | compile speedup |
|---|---|---|---|---|
| SmallMLP 2-8-1, batch 32 | 0.0128 | 0.0104 | 0.0280 | 0.46x (slower) |
| SmallMLP 2-8-1, batch 4096 | 0.0965 | 0.0874 | 0.0971 | 0.99x |
| Cheap pointwise chain, 4096x4096 (memory-bound) | 167.6 | 170.5 | 20.1 | **8.4x** |
| tanh/sigmoid/gelu chain, 2048x2048 (compute-bound) | 9.24 | 9.33 | 8.15 | 1.13x |
| Linear 1024x1024 + pointwise chain, batch 256 | 2.84 | 2.48 | 2.74 | 1.03x |

- All compiled outputs match eager (`allclose`, atol 1e-5).
- First `torch.compile` call (compilation) took ~1-24 s on a cold cache, ~0.05-1 s with the inductor cache warm. This cost is paid once per input shape.
- Inductor really fused the chain: the generated kernel is `cpp_fused_add_div_gelu_mul_relu_sigmoid_sub_tanh_0` (one loop instead of ~10).

### Tensor layout and non-contiguous views (4096x4096 float32)

| Tensor | shape | dtype | stride | contiguous | same storage as x |
|---|---|---|---|---|---|
| `x` | [4096, 4096] | float32 | [4096, 1] | True | yes |
| `x.t()` | [4096, 4096] | float32 | [1, 4096] | **False** | yes (view, no copy) |
| `x[:, ::2]` | [4096, 2048] | float32 | [4096, 2] | **False** | yes |
| `x.t().contiguous()` | [4096, 4096] | float32 | [4096, 1] | True | no (copy) |

| Op | contiguous | `x.t()` view | slowdown | after `.contiguous()` |
|---|---|---|---|---|
| `.sum(dim=1)` | 1.25 ms | 2.21 ms | 1.8x | 1.27 ms |
| `a*2 + 1` | 24.9 ms | 84.2 ms | 3.4x | 20.3 ms |
| `softmax(dim=1)` | 22.1 ms | 70.4 ms | 3.2x | 23.6 ms |
| `cumsum(dim=1)` | 21.4 ms | 70.9 ms | 3.3x | 20.9 ms |

The copy `x.t().contiguous()` itself costs 43.9 ms. A transposed view is free to
create, but every op on it walks memory with stride 4096, so it misses cache and
cannot use contiguous SIMD loads: 1.8-3.4x slower. If the tensor is used once,
running the op on the view is cheaper than copying first; if it is reused several
times, pay for `.contiguous()` once.

## Note: when compilation helps

**Helps:**

- Long chains of cheap pointwise ops (activations, scaling, residual adds, normalization).
  Eager launches one kernel per op and writes every intermediate to memory;
  the compiler fuses them into one pass. Here: **8.4x** on a memory-bound chain.
- Long-running jobs (training, serving) where the one-time compile cost (seconds)
  is amortized over thousands of steps with stable input shapes.
- GPUs, where per-kernel launch overhead and memory traffic dominate even more than on CPU.

**Does not help (or hurts):**

- Tiny models / tiny batches: the work is a few microseconds, and the compiled
  wrapper (guard checks, dispatch) adds overhead. Here: 2-8-1 MLP, batch 32 got **2x slower**.
- Workloads dominated by one big matmul/conv: those already call optimized
  BLAS/cuDNN kernels in eager mode; fusion only touches the small pointwise tail (~1.0x).
- Compute-bound pointwise math (tanh, gelu): fusion saves memory traffic but the
  bottleneck is arithmetic (~1.1x).
- Short scripts, notebooks, frequently changing shapes or Python control flow:
  compile time and recompilations eat the gain.

`torch.jit.trace` removes some Python overhead (1.1-1.2x on the tiny MLP) but does
not fuse ops on CPU, so it does not help the pointwise chains. It is mainly useful
for exporting a static graph. If `torch.compile` is unavailable (e.g. Windows without a
C++ compiler), the script falls back to reporting tracing only.
