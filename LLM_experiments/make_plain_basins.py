"""Plain-model (threshold-0.5, no learning) replicator basins.

The plain prompted LLM at fixed threshold 0.5 has per-state accuracy equal to
each state's positive-label rate (since π_ref predicts y=1 on ~90% of rows in
this dataset). Those rates are the wandb `baseline_pi_ref_acc/{state}` numbers
that every cluster run logs.

This script integrates the deterministic discrete replicator (Taylor-Jonker)
from a set of initial conditions using those per-state accuracies, and plots
all trajectories on one ternary panel with the corresponding vector field.
The expected outcome — NY is the unique attractor — visually confirms the
β → ∞ limit of the closed-form sweep.

Usage::

    python -m LLM_experiments.make_plain_basins \
        --out /Users/gabesmithline/Desktop/fig5_plain_basins.png

Override the accuracies via flags if you want to test counterfactuals.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from LLM_experiments.make_fig5 import _static_replicator_field

try:
    import mpltern  # noqa: F401
    _HAVE_MPLTERN = True
except ImportError:
    print("[make_plain_basins] mpltern not installed; install with `pip install mpltern`.",
          file=sys.stderr)
    sys.exit(2)


# Plain π_ref at threshold 0.5: accuracy = pos_rate per state on the cached
# test pools, matching the wandb baseline_pi_ref_acc/{state} numbers logged at
# the start of every cluster run.
DEFAULT_ACC_TX = 0.3728
DEFAULT_ACC_NY = 0.4090
DEFAULT_ACC_CA = 0.4052


# Initial conditions in (TX, NY, CA) order to match the panel apex/left/right.
DEFAULT_INITS_TNC = [
    (1 / 3, 1 / 3, 1 / 3),
    (0.10, 0.10, 0.80),
    (0.10, 0.80, 0.10),
    (0.80, 0.10, 0.10),
    (0.45, 0.45, 0.10),
    (0.45, 0.10, 0.45),
    (0.10, 0.45, 0.45),
]


def discrete_replicator(p0: np.ndarray, acc: np.ndarray, T: int) -> np.ndarray:
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
    ap.add_argument("--n-grid", type=int, default=21)
    ap.add_argument("--acc-tx", type=float, default=DEFAULT_ACC_TX)
    ap.add_argument("--acc-ny", type=float, default=DEFAULT_ACC_NY)
    ap.add_argument("--acc-ca", type=float, default=DEFAULT_ACC_CA)
    args = ap.parse_args()

    acc_TNC = np.array([args.acc_tx, args.acc_ny, args.acc_ca])
    print(f"[make_plain_basins] plain-model accs (threshold 0.5): "
          f"TX={acc_TNC[0]:.4f} NY={acc_TNC[1]:.4f} CA={acc_TNC[2]:.4f}")

    P, acc_p, rep_f = _static_replicator_field(acc_TNC, n=args.n_grid)

    fig, ax = plt.subplots(
        figsize=(6.2, 5.2),
        subplot_kw={"projection": "ternary"},
        layout="constrained",
    )

    potential = ax.tripcolor(*P, acc_p * 100, cmap="turbo", shading="gouraud", alpha=0.8)
    rep_norm = float(np.linalg.norm(rep_f, axis=0).max())
    rep_scale = max(rep_norm * 6.0, 1e-6)
    ax.quiver(*P, *rep_f, scale=rep_scale, clip_on=True, color="black", alpha=0.7, width=0.004)

    cax = ax.inset_axes([1.18, 0.1, 0.045, 0.85], transform=ax.transAxes)
    cbar = fig.colorbar(potential, cax=cax, format="{x:.0f}%")
    cbar.set_label(r"$\mathrm{acc}_{\mathbf{p}}$", rotation=270, va="baseline", labelpad=10)

    cmap = matplotlib.colormaps["viridis"]
    for i, p0 in enumerate(DEFAULT_INITS_TNC):
        path = discrete_replicator(np.array(p0), acc_TNC, T=args.T)
        color = cmap(i / max(len(DEFAULT_INITS_TNC) - 1, 1))
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color, linewidth=1.7, zorder=10,
                alpha=0.95)
        ax.scatter(path[0, 0], path[0, 1], path[0, 2], color=color, marker="o", s=28,
                   zorder=11, edgecolors="white", linewidths=0.6)
        ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], color=color, marker="X", s=42,
                   zorder=12, edgecolors="white", linewidths=0.6)

    ax.set_tlabel("TX"); ax.set_llabel("NY"); ax.set_rlabel("CA")
    ax.taxis.set_label_position("tick1")
    ax.laxis.set_label_position("tick1")
    ax.raxis.set_label_position("tick1")
    for axname in ("t", "l", "r"):
        getattr(ax, f"{axname}axis").set_ticks(np.linspace(0, 1, 6))
    ax.set_title("Plain prompted LLM @ threshold 0.5: replicator basins")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=150)
    print(f"[make_plain_basins] wrote {args.out}")


if __name__ == "__main__":
    main()
