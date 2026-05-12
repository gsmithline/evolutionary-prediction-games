"""Plot the EGTA decoupled-PR tensor produced by egta_matrix.py.

Three panels:
  * heatmaps: K x K matrices M_t at a handful of selected timesteps.
  * diag-vs-offdiag: per intervention i, M[t, i, i] (own-world risk) vs
    mean_{j != i} M[t, i, j] (mean risk across other models in i's world).
  * column-means: per intervention j, mean_i M[t, i, j] (model j's mean risk
    across all induced worlds — a robustness/generalist score).

Usage:
    python -m LLM_experiments.plot_egta_matrix /path/egta_sft_sweep_s0.npz \
        --out-dir /path/figs_egta/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

try:
    from notebook_env import *  # noqa: F401,F403
    import evoml.analysis as evan
    _HAVE_NOTEBOOK_ENV = True
except Exception as e:
    print(f"[plot_egta_matrix] notebook_env import failed: {e}", file=sys.stderr)
    matplotlib.rcParams["text.usetex"] = False
    evan = None
    _HAVE_NOTEBOOK_ENV = False


def _default_timesteps(T: int, n: int = 4):
    if T <= n:
        return list(range(T))
    return [int(round(x)) for x in np.linspace(0, T - 1, n)]


def _grid(ax_grid):
    if evan is not None:
        ax_grid.grid(**evan.background_line_style)
    else:
        ax_grid.grid(alpha=0.25)


def plot_heatmaps(M, labels, t_axis, timesteps, *, out_path: Path):
    n = len(timesteps)
    fig, axs = plt.subplots(1, n, figsize=(3.2 * n, 3.4), constrained_layout=True)
    if n == 1:
        axs = [axs]
    vmin, vmax = float(M.min()), float(M.max())
    K = M.shape[1]
    for ax, tt in zip(axs, timesteps):
        im = ax.imshow(M[tt], cmap="magma", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("model $j$")
        ax.set_ylabel("world $i$")
        ax.set_title(f"$t={int(t_axis[tt])}$")
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{M[tt, i, j]:.2f}", ha="center", va="center",
                        color="white" if M[tt, i, j] < 0.6 * vmax + 0.4 * vmin else "black",
                        fontsize=7)
    cbar = fig.colorbar(im, ax=axs, shrink=0.85, pad=0.02)
    cbar.set_label(r"$M_t[i,j] = \mathrm{DPR}(\pi_t^{(i)}\!\to i,\; \pi_t^{(j)})$")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_diag_vs_offdiag(M, labels, t_axis, *, out_path: Path):
    T, K, _ = M.shape
    diag = np.einsum("tii->ti", M)
    off_sum = M.sum(axis=2) - diag
    off_mean = off_sum / max(K - 1, 1)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.0), constrained_layout=True, sharex=True)
    cmap = plt.get_cmap("tab10")
    for i, label in enumerate(labels):
        c = cmap(i % 10)
        axs[0].plot(t_axis, diag[:, i], color=c, linewidth=1.6, label=str(label))
        axs[1].plot(t_axis, off_mean[:, i], color=c, linewidth=1.6, label=str(label))
    axs[0].set_title(r"Own-world risk: $M_t[i,i]$")
    axs[1].set_title(r"Mean risk in $i$'s world across other models: $\mathrm{mean}_{j\neq i} M_t[i,j]$")
    for ax in axs:
        ax.set_xlabel("Time ($t$)")
        ax.set_ylabel("Risk (1 - accuracy)")
        _grid(ax)
    axs[0].legend(loc="best", fontsize=8)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_column_means(M, labels, t_axis, *, out_path: Path):
    col_mean = M.mean(axis=1)  # (T, K) -- model j averaged across worlds i.
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for j, label in enumerate(labels):
        ax.plot(t_axis, col_mean[:, j], color=cmap(j % 10), linewidth=1.6, label=str(label))
    ax.set_title(r"Model robustness across induced worlds: $\mathrm{mean}_i M_t[i,j]$")
    ax.set_xlabel("Time ($t$)")
    ax.set_ylabel("Risk (1 - accuracy)")
    _grid(ax)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path, help=".npz produced by egta_matrix.py")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Directory for output PNGs (default: alongside the .npz).")
    ap.add_argument("--timesteps", type=int, nargs="*", default=None,
                    help="Time indices at which to draw heatmaps. Default: 4 evenly spaced.")
    ap.add_argument("--n-heatmaps", type=int, default=4,
                    help="If --timesteps not given, draw this many evenly spaced heatmaps.")
    args = ap.parse_args()

    if not args.npz.exists():
        print(f"[plot_egta_matrix] .npz not found: {args.npz}", file=sys.stderr)
        sys.exit(2)

    data = np.load(args.npz, allow_pickle=False)
    M = data["M"]
    t_axis = data["t"]
    labels = [str(x) for x in data["labels"]]
    T, K, K2 = M.shape
    if K != K2:
        raise ValueError(f"M is not square in the (K, K) axes: {M.shape}")
    timesteps = args.timesteps if args.timesteps is not None else _default_timesteps(T, args.n_heatmaps)
    for tt in timesteps:
        if not (0 <= tt < T):
            raise ValueError(f"timestep {tt} out of range [0, {T})")

    out_dir = args.out_dir or args.npz.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.npz.stem
    plot_heatmaps(M, labels, t_axis, timesteps, out_path=out_dir / f"{stem}_heatmaps.png")
    plot_diag_vs_offdiag(M, labels, t_axis, out_path=out_dir / f"{stem}_diag_vs_offdiag.png")
    plot_column_means(M, labels, t_axis, out_path=out_dir / f"{stem}_column_means.png")
    print(f"[plot_egta_matrix] wrote 3 PNGs to {out_dir}/{stem}_*.png")


if __name__ == "__main__":
    main()
