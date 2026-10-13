"""Time forward and backward passes separately (1 warm-up + median of 100 runs)."""
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data import SEED, load_data, set_seeds
from nn import backward, forward, init_params

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
REPEATS = 100


def median_ms(fn):
    fn()  # warm-up
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000


def matmul_flops(batch, width):
    """2*m*k*n FLOPs per matmul (a multiply and an add per term)."""
    fwd = 2 * batch * (64 * width + width * width + width * 10)            # 3 matmuls
    bwd = 2 * batch * (width * 10 * 2 + width * width * 2 + 64 * width)    # 5 matmuls
    return fwd, bwd


def main():
    set_seeds()
    X_train, _, Y_train, _ = load_data()
    rows = []
    for width in (16, 64):
        params = init_params(width, np.random.default_rng(SEED))
        for batch in (32, len(X_train)):
            Xb, Yb = X_train[:batch], Y_train[:batch]
            _, cache = forward(params, Xb)
            f_ms = median_ms(lambda: forward(params, Xb))
            b_ms = median_ms(lambda: backward(params, cache, Yb))
            f_flops, b_flops = matmul_flops(batch, width)
            rows.append({
                "width": width, "batch": batch,
                "forward_ms": round(f_ms, 4), "backward_ms": round(b_ms, 4),
                "backward_to_forward": round(b_ms / f_ms, 2),
                "forward_matmul_flops": f_flops, "backward_matmul_flops": b_flops,
                "flop_ratio": round(b_flops / f_flops, 2),
            })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "timing.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(f"\nSaved to {RESULTS / 'timing.csv'}")


if __name__ == "__main__":
    main()