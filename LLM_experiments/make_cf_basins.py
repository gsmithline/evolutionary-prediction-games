"""Closed-form sweep visualization: one ternary panel per β, all seeds overlaid.

Reads every fig5_cf_*.csv in a results directory and groups by β. Each panel
shows the per-seed trajectory on the (TX, NY, CA) simplex with start (●) and
end (✕) markers. β values are sorted ascending; the layout is a row-major
grid sized to the number of distinct β values.

Usage::

    python -m LLM_experiments.make_cf_basins \
        --results-dir LLM_experiments/_results \
        --out figs/fig5_cf_basins.png
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

try:
    import mpltern  # noqa: F401
    _HAVE_MPLTERN = True
except ImportError:
    print("[make_cf_basins] mpltern not installed; install with `pip install mpltern`.",
          file=sys.stderr)
    sys.exit(2)


_BETA_RE = re.compile(r"fig5_cf_b([0-9]+(?:p[0-9]+)?)_s(\d+)\.csv")


def _parse_beta_seed(path: Path):
    m = _BETA_RE.match(path.name)
    if not m:
        return None
    beta_str = m.group(1).replace("p", ".")
    return float(beta_str), int(m.group(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("LLM_experiments/_results"))
    ap.add_argument("--out", type=Path, default=Path("LLM_experiments/_results/fig5_cf_basins.png"))
    ap.add_argument("--ncols", type=int, default=4)
    args = ap.parse_args()

    grouped: dict[float, list[tuple[int, pd.DataFrame]]] = defaultdict(list)
    for csv in sorted(args.results_dir.glob("fig5_cf_*.csv")):
        parsed = _parse_beta_seed(csv)
        if parsed is None:
            continue
        beta, seed = parsed
        grouped[beta].append((seed, pd.read_csv(csv)))

    if not grouped:
        print(f"[make_cf_basins] no fig5_cf_*.csv in {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    betas = sorted(grouped.keys())
    n = len(betas)
    ncols = min(args.ncols, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.4 * ncols, 3.4 * nrows),
        subplot_kw={"projection": "ternary"},
        layout="constrained",
    )
    axes = np.atleast_2d(axes)

    cmap = matplotlib.colormaps["viridis"]
    seed_colors = {0: cmap(0.15), 1: cmap(0.5), 2: cmap(0.85)}

    for i, beta in enumerate(betas):
        ax = axes[i // ncols, i % ncols]
        for seed, df in sorted(grouped[beta], key=lambda kv: kv[0]):
            pivot = df.pivot(index="t", columns="state", values="p")
            color = seed_colors.get(seed, "tab:gray")
            ax.plot(pivot["TX"], pivot["NY"], pivot["CA"],
                    color=color, linewidth=1.5, zorder=10, alpha=0.9,
                    label=f"s={seed}")
            ax.scatter(pivot["TX"].iloc[0], pivot["NY"].iloc[0], pivot["CA"].iloc[0],
                       color=color, marker="o", s=22, zorder=11,
                       edgecolors="white", linewidths=0.5)
            ax.scatter(pivot["TX"].iloc[-1], pivot["NY"].iloc[-1], pivot["CA"].iloc[-1],
                       color=color, marker="X", s=42, zorder=12,
                       edgecolors="white", linewidths=0.5)

        ax.set_tlabel("TX"); ax.set_llabel("NY"); ax.set_rlabel("CA")
        ax.taxis.set_label_position("tick1")
        ax.laxis.set_label_position("tick1")
        ax.raxis.set_label_position("tick1")
        for axname in ("t", "l", "r"):
            getattr(ax, f"{axname}axis").set_ticks(np.linspace(0, 1, 6))
        beta_label = f"$\\beta={beta:g}$" if beta >= 1 else f"$\\beta={beta:g}$"
        ax.set_title(beta_label, fontsize=11)
        if i == 0:
            ax.legend(loc="upper right", fontsize=7, frameon=False)

    # Hide any unused panels.
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=150)
    print(f"[make_cf_basins] wrote {args.out}")


if __name__ == "__main__":
    main()
