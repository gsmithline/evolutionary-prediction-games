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
    _HAVE_MPLTERN = True
except ImportError:
    _HAVE_MPLTERN = False


# Match Fig.5's right-panel palette in the paper.
GROUP_COLORS_3 = {"CA": "tab:cyan", "NY": "tab:green", "TX": "tab:orange"}


def _trajectory_to_pivots(traj: pd.DataFrame):
    pivot_p = traj.pivot(index="t", columns="state", values="p")
    pivot_acc = traj.pivot(index="t", columns="state", values="acc")
    pivot_fit = traj.pivot(index="t", columns="state", values="fitness")
    # Re-order columns deterministically.
    cols = ["CA", "NY", "TX"]
    return pivot_p[cols], pivot_acc[cols], pivot_fit[cols]


def make_fig5(traj: pd.DataFrame, *, title_suffix: str = "", figsize=(11, 3.4)):
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

    # ---- Left panel: ternary trajectory (or fallback) ----
    ax = axs["T"]
    if ternary_ok:
        # mpltern.plot expects (top, left, right) coordinates. Match the paper's
        # Fig.5 left: TX at the apex, NY on the left, CA on the right.
        ax.plot(pivot_p["TX"], pivot_p["NY"], pivot_p["CA"], color="red",
                linewidth=2.0, zorder=10)
        # Mark the start (uniform 1/3,1/3,1/3) and the end.
        ax.scatter(pivot_p["TX"].iloc[0], pivot_p["NY"].iloc[0], pivot_p["CA"].iloc[0],
                   color="black", marker="o", s=18, zorder=11)
        ax.scatter(pivot_p["TX"].iloc[-1], pivot_p["NY"].iloc[-1], pivot_p["CA"].iloc[-1],
                   color="red", marker="X", s=36, zorder=11)
        ax.set_tlabel("TX"); ax.set_llabel("NY"); ax.set_rlabel("CA")
        ax.taxis.set_label_position("tick1")
        ax.laxis.set_label_position("tick1")
        ax.raxis.set_label_position("tick1")
        ax.set_title("ACSIncome Replicator Dynamics")
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
            label="$\\mathrm{acc}_{\\bm{p}}$")
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
