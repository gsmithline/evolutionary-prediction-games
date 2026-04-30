"""S&R Fig.5 reproduction from a single trajectory CSV.

Three panels matching the paper's Fig.5 layout:
  - Left:   Ternary replicator trajectory on the K=3 simplex (mpltern).
  - Center: Population composition over time (one line per state).
  - Right:  Group fitness/accuracy over time + population-weighted overall acc.

Mirrors the styling idioms in `ternary.ipynb` and `empirical.ipynb` of the EPG
repo (notebook_env preamble, evoml.analysis helpers, mpltern projection).

Usage:
    cd /path/to/evolutionary-prediction-games
    python -m LLM_experiments.make_fig5 _results/fig5_static_s0.csv
    python -m LLM_experiments.make_fig5 _results/fig5_sft_b1_s0.csv --out figs/fig5_sft_b1_s0.pdf

For batch (one figure per β) just loop in shell:
    for f in _results/fig5_sft_*.csv; do
        python -m LLM_experiments.make_fig5 "$f"
    done
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import the EPG repo's notebook preamble (rcParams, evoml, etc.). LaTeX is
# enabled by default there; we fall back gracefully if no LaTeX install.
import matplotlib
import matplotlib.pyplot as plt

try:
    from notebook_env import *  # noqa: F401,F403
    import evoml.analysis as evan
    from mpltern.datasets import get_triangular_grid  # noqa: F401  (validates install)
    _HAVE_NOTEBOOK_ENV = True
except Exception as e:  # missing LaTeX, mpltern, or running outside repo root
    print(f"[make_fig5] notebook_env import failed: {e}", file=sys.stderr)
    print("[make_fig5] continuing with plain matplotlib defaults.", file=sys.stderr)
    matplotlib.rcParams["text.usetex"] = False
    evan = None
    _HAVE_NOTEBOOK_ENV = False

try:
    import mpltern  # noqa: F401
    from mpltern.datasets import get_triangular_grid
    _HAVE_MPLTERN = True
except ImportError:
    _HAVE_MPLTERN = False


# Match Fig.5's right-panel palette in the paper.
GROUP_COLORS_3 = {"CA": "tab:cyan", "NY": "tab:green", "TX": "tab:orange"}


def _static_replicator_field(acc_per_group: np.ndarray, *, n: int = 21):
    """Return (P, acc_p, replicator_f) on a triangular grid for a static policy.

    P has shape (3, M) with rows indexed (TX, NY, CA) to match the panel layout.
    acc_per_group must be in the same (TX, NY, CA) order. acc_p is the
    population-weighted accuracy at each grid point; replicator_f[k, m] is
    p_k * (acc_k - acc_p[m]) — the replicator vector at that grid point.
    """
    P = np.asarray(get_triangular_grid(n=n))  # (3, M), each column sums to 1.
    acc_p = (P * acc_per_group[:, None]).sum(axis=0)
    replicator_f = P * (acc_per_group[:, None] - acc_p[None, :])
    return P, acc_p, replicator_f


def _trajectory_to_pivots(traj: pd.DataFrame):
    pivot_p = traj.pivot(index="t", columns="state", values="p")
    pivot_acc = traj.pivot(index="t", columns="state", values="acc")
    pivot_fit = traj.pivot(index="t", columns="state", values="fitness")
    # Re-order columns deterministically.
    cols = ["CA", "NY", "TX"]
    return pivot_p[cols], pivot_acc[cols], pivot_fit[cols]


def make_fig5(traj: pd.DataFrame, *, title_suffix: str = "", figsize=(13.5, 4.0)):
    pivot_p, pivot_acc, pivot_fit = _trajectory_to_pivots(traj)

    # mpltern is part of the evoml env; if unavailable we draw a plain p_min(t)
    # panel instead of the simplex projection.
    ternary_ok = _HAVE_MPLTERN
    per_subplot_kw = {"T": {"projection": "ternary"}} if ternary_ok else None
    fig, axs = plt.subplot_mosaic(
        [["T", "C", "F"]],
        figsize=figsize,
        per_subplot_kw=per_subplot_kw,
        layout="constrained",
        width_ratios=[1.1, 1.5, 1.5],
    )

    # ---- Left panel: ternary trajectory + replicator vector field ----
    ax = axs["T"]
    if ternary_ok:
        # Per-group accuracy under the static policy: mean over rows where the
        # state was actually sampled (sample_count > 0). For non-static runs
        # this estimate is along the trajectory only — the field below assumes
        # a constant policy, so it is informative only for static.
        valid = traj["sample_count"] > 0 if "sample_count" in traj.columns else traj["acc"].notna()
        accs_by_state = traj[valid].groupby("state")["acc"].mean()
        # Order (TX, NY, CA) to match the panel apex/left/right.
        acc_per_group = np.array([accs_by_state["TX"], accs_by_state["NY"], accs_by_state["CA"]])
        P, acc_p, rep_f = _static_replicator_field(acc_per_group, n=21)

        potential = ax.tripcolor(*P, acc_p * 100, cmap="turbo", shading="gouraud", alpha=0.85)
        # Scale arrows to the field's max magnitude so they are visible regardless
        # of how small per-group accuracy differences are.
        rep_norm = float(np.linalg.norm(rep_f, axis=0).max())
        rep_scale = max(rep_norm * 6.0, 1e-6)
        ax.quiver(*P, *rep_f, scale=rep_scale, clip_on=True, color="black", alpha=0.85, width=0.004)

        cax = ax.inset_axes([1.18, 0.1, 0.045, 0.85], transform=ax.transAxes)
        cbar = fig.colorbar(potential, cax=cax, format="{x:.0f}%")
        cbar.set_label(r"$\mathrm{acc}_{\mathbf{p}}$", rotation=270, va="baseline", labelpad=10)

        # Trajectory on top of the field.
        ax.plot(pivot_p["TX"], pivot_p["NY"], pivot_p["CA"], color="red",
                linewidth=2.0, zorder=10)
        ax.scatter(pivot_p["TX"].iloc[0], pivot_p["NY"].iloc[0], pivot_p["CA"].iloc[0],
                   color="black", marker="o", s=22, zorder=11, edgecolors="white", linewidths=0.6)
        ax.scatter(pivot_p["TX"].iloc[-1], pivot_p["NY"].iloc[-1], pivot_p["CA"].iloc[-1],
                   color="red", marker="X", s=42, zorder=11, edgecolors="white", linewidths=0.6)

        ax.set_tlabel("TX"); ax.set_llabel("NY"); ax.set_rlabel("CA")
        ax.taxis.set_label_position("tick1")
        ax.laxis.set_label_position("tick1")
        ax.raxis.set_label_position("tick1")
        for axname in ("t", "l", "r"):
            getattr(ax, f"{axname}axis").set_ticks(np.linspace(0, 1, 6))
        ax.set_title("Replicator Dynamics")
    else:
        ax.plot(pivot_p.index, pivot_p.min(axis=1), color="red", linewidth=2.0,
                label="$\\min_k p_k$")
        ax.axhline(0.01, color="black", linestyle=":", alpha=0.5)
        ax.set_xlabel("Time ($t$)")
        ax.set_ylabel("$\\min_k p_k$")
        ax.set_title("Smallest group share over time")
        ax.legend()

    # ---- Center panel: population composition ----
    ax = axs["C"]
    for state in ["CA", "NY", "TX"]:
        ax.plot(pivot_p.index, pivot_p[state], color=GROUP_COLORS_3[state],
                linewidth=1.7, label=f"$p_{{\\mathrm{{{state}}}}}$")
    ax.axhline(0.01, color="red", linestyle=":", alpha=0.6)
    ax.text(pivot_p.index[-1], 0.012, "$\\leq 1\\%$", color="red", ha="right",
            va="bottom", fontsize=8)
    ax.set_xlabel("Time ($t$)")
    ax.set_ylabel("Relative group size")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Population Composition Over Time")
    ax.legend(loc="best", fontsize=8)
    if evan is not None:
        ax.grid(**{**evan.background_line_style, "axis": "y"})
    else:
        ax.grid(alpha=0.25)

    # ---- Right panel: group fitness over time ----
    ax = axs["F"]
    for state in ["CA", "NY", "TX"]:
        ax.plot(pivot_acc.index, pivot_acc[state], color=GROUP_COLORS_3[state],
                linewidth=1.7, label=f"$\\mathrm{{acc}}_{{\\mathrm{{{state}}}}}$")
    acc_p = (pivot_p * pivot_acc).sum(axis=1)
    ax.plot(acc_p.index, acc_p, color="black", linestyle="--", linewidth=1.4,
            label="$\\mathrm{acc}_{\\mathbf{p}}$")
    ax.set_xlabel("Time ($t$)")
    ax.set_ylabel("Classification accuracy")
    ax.set_title("Group Fitness Over Time")
    ax.legend(loc="best", fontsize=8)
    if evan is not None:
        ax.grid(**evan.background_line_style)
    else:
        ax.grid(alpha=0.25)

    if title_suffix:
        fig.suptitle(title_suffix, fontsize=11)
    return fig, axs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, help="Trajectory CSV from run_fig5.py")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output figure path (default: <csv stem>_fig5.pdf next to the CSV).")
    ap.add_argument("--title", type=str, default="",
                    help="Optional suptitle (e.g. β=1.0, sft).")
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"[make_fig5] CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(2)

    traj = pd.read_csv(args.csv)
    title = args.title or args.csv.stem
    fig, _ = make_fig5(traj, title_suffix=title)

    out_path = args.out or args.csv.with_name(args.csv.stem + "_fig5.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if evan is not None and out_path.suffix == ".pdf":
        evan.save_and_download_fig(fig, str(out_path), bbox_inches="tight")
    else:
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"[make_fig5] wrote {out_path}")


if __name__ == "__main__":
    main()
