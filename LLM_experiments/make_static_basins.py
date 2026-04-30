"""Multi-init replicator trajectories under a static policy.

Static-policy per-group accuracies do not depend on the population mixture, so
we can read the per-state mean accuracies from an existing static CSV and
integrate the discrete replicator from many initial conditions without any
new LLM evaluation. Mirrors the discrete-replicator form in
``evoml.dynamics.discrete_replicator``::

    p_{t+1} = p_t * (acc + 1) / (p_t @ (acc + 1))

Usage::

    python -m LLM_experiments.make_static_basins _results/fig5_static_s0.csv \
        --out figs/fig5_static_basins.png --T 600
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from LLM_experiments.make_fig5 import _static_replicator_field, GROUP_COLORS_3

try:
    import mpltern  # noqa: F401
    _HAVE_MPLTERN = True
except ImportError:
    print("[make_static_basins] mpltern not installed; install with `pip install mpltern`.",
          file=sys.stderr)
    sys.exit(2)


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
    ap.add_argument("csv", type=Path, help="Static-policy trajectory CSV (for per-state accs).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output figure path (default: <csv stem>_basins.png).")
    ap.add_argument("--T", type=int, default=600, help="Replicator steps per init.")
    ap.add_argument("--n-grid", type=int, default=21, help="Triangular grid density for the field.")
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"[make_static_basins] CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(2)

    traj = pd.read_csv(args.csv)
    valid = traj["sample_count"] > 0 if "sample_count" in traj.columns else traj["acc"].notna()
    accs_by_state = traj[valid].groupby("state")["acc"].mean()
    acc_TNC = np.array([accs_by_state["TX"], accs_by_state["NY"], accs_by_state["CA"]])
    print(f"[make_static_basins] static accs: TX={acc_TNC[0]:.4f} NY={acc_TNC[1]:.4f} "
          f"CA={acc_TNC[2]:.4f}")

    P, acc_p, rep_f = _static_replicator_field(acc_TNC, n=args.n_grid)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), subplot_kw={"projection": "ternary"},
                           layout="constrained")

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
    ax.set_title("Static replicator: trajectories from varied initial conditions")

    out_path = args.out or args.csv.with_name(args.csv.stem + "_basins.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"[make_static_basins] wrote {out_path}")


if __name__ == "__main__":
    main()
