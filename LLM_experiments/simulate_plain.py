"""Deterministic replicator simulation under the plain prompted LLM at
threshold 0.5. Per-state accuracies are constant in p (no learning at all)
and equal each state's positive-label rate, since π_ref predicts y=1 on ~90%
of rows — see the wandb baseline_pi_ref_acc/{state} numbers logged at the
start of every cluster run.

Outputs a CSV in the same schema as fig5_<*>.csv files, so it plugs straight
into `make_fig5.py` for the 3-panel view (ternary + composition + fitness).

Usage::

    python -m LLM_experiments.simulate_plain --out _results/fig5_plain_s0.csv
    python -m LLM_experiments.make_fig5 _results/fig5_plain_s0.csv \
        --title "plain prompted (threshold 0.5)" --out figs/fig5_plain.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from LLM_experiments.data import STATES


# Plain π_ref at threshold 0.5: accuracy = positive-label rate per state.
# These are the wandb baseline_pi_ref_acc/{state} numbers from cluster runs.
DEFAULT_ACC_TX = 0.3728
DEFAULT_ACC_NY = 0.4090
DEFAULT_ACC_CA = 0.4052


def discrete_replicator(p0: np.ndarray, acc: np.ndarray, T: int) -> np.ndarray:
    """Taylor-Jonker discrete replicator with constant fitness."""
    p = np.asarray(p0, dtype=float)
    f = np.asarray(acc, dtype=float) + 1.0
    out = [p.copy()]
    for _ in range(T):
        p = (p * f) / (p @ f)
        out.append(p.copy())
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--T", type=int, default=600)
    ap.add_argument("--n-per-step", type=int, default=1000)
    ap.add_argument("--acc-CA", type=float, default=DEFAULT_ACC_CA)
    ap.add_argument("--acc-NY", type=float, default=DEFAULT_ACC_NY)
    ap.add_argument("--acc-TX", type=float, default=DEFAULT_ACC_TX)
    args = ap.parse_args()

    # data.STATES is ("CA", "NY", "TX"); keep that ordering throughout.
    accs_by_state = {"CA": args.acc_CA, "NY": args.acc_NY, "TX": args.acc_TX}
    acc_vec = np.array([accs_by_state[s] for s in STATES])
    print(f"[simulate_plain] STATES={STATES} accs={acc_vec.tolist()}")

    p0 = np.full(len(STATES), 1.0 / len(STATES))
    path = discrete_replicator(p0, acc_vec, T=args.T)

    rows = []
    for t in range(args.T + 1):
        for k, state in enumerate(STATES):
            p_k = float(path[t, k])
            sample_count = int(round(args.n_per_step * p_k))
            # The accuracy and fitness columns are populated for every step
            # except the very last (t=T), to match how the empirical pipeline
            # logs a final p_next without an associated acc/fitness pair.
            if t < args.T:
                acc = float(acc_vec[k])
                fitness = acc
            else:
                acc = ""
                fitness = ""
            rows.append({
                "t": t,
                "state": state,
                "p": p_k,
                "acc": acc,
                "fitness": fitness,
                "sample_count": sample_count if t < args.T else 0,
                "step_seconds": "",
            })
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[simulate_plain] wrote {args.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
