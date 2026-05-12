"""Bifurcation graph: terminal performative deviation gain Delta_T(beta) vs beta.

Reads two CF tensors:
  * egta_cf_sweep_with_static_s0.npz (K=8 dense beta grid at seed 0):
      betas {0.01, 0.05, 0.1, 0.3, 1, 3, 10}, the within-tensor Delta is taken
      against this K-set. Static is dropped from the curve (not a beta-axis
      point) but stays in the K-set for the min_j computation.
  * egta_cf_4beta_s{0,1,2}.npz (K=5 sparse beta grid x 3 seeds): betas
      {0.01, 0.1, 1, 10}. Used for cross-seed error bars on those four points.

Honest caveat: Delta depends on which K-set min_j is taken over, so the dense
single-seed curve and the cross-seed error bars use slightly different K-sets.
The plot annotates this.

Usage: python -m LLM_experiments.bifurcation_plot \
    --dense /path/egta_cf_sweep_with_static_s0.npz \
    --cross-seed /path/egta_cf_4beta_s0.npz \
    --cross-seed /path/egta_cf_4beta_s1.npz \
    --cross-seed /path/egta_cf_4beta_s2.npz \
    --out /path/bifurcation_delta_vs_beta.png
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

try:
    from notebook_env import *  # noqa: F401,F403
    import evoml.analysis as evan
except Exception:
    matplotlib.rcParams["text.usetex"] = False
    evan = None

from LLM_experiments.egta_wandb import WandbSession, add_wandb_args


DEFAULT_LABEL_REGEX = r"^cf_b([0-9]+(?:\.[0-9]+)?)$"


def _parse_beta_from_label(label: str, pattern: re.Pattern):
    """Extract a float beta from the label using `pattern`. Returns None on no match.

    The pattern must have exactly one capture group containing the float.
    Examples that match the default:
        'cf_b0.01' -> 0.01
        'cf_b10'   -> 10.0
    For FJ, pass --label-regex 'fj_b([0-9]+(?:\\.[0-9]+)?)'.
    """
    m = pattern.match(label)
    if not m:
        return None
    return float(m.group(1))


def _terminal_delta_by_beta(npz_path: Path, pattern: re.Pattern):
    d = np.load(npz_path, allow_pickle=False)
    M = d["M"]
    labels = [str(x) for x in d["labels"]]
    diag = np.einsum("tii->ti", M[-1:None, ...])[0]  # (K,) at t = T-1
    row_min = M[-1].min(axis=1)
    delta = diag - row_min
    out = []
    for i, label in enumerate(labels):
        beta = _parse_beta_from_label(label, pattern)
        if beta is None:
            continue
        out.append((beta, float(delta[i])))
    out.sort()
    return out  # list of (beta, delta_T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", type=Path, required=True,
                    help="CF tensor with dense beta grid (e.g. egta_cf_sweep_with_static_s0.npz)")
    ap.add_argument("--cross-seed", action="append", type=Path, default=[],
                    help="CF tensors at different seeds for error bars (4-beta tensors). Repeatable.")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--label-regex", type=str, default=DEFAULT_LABEL_REGEX,
                    help=(r"Regex with one capture group for beta. Default matches "
                          r"'cf_b<float>'. FJ usage: 'fj_b([0-9]+(?:\.[0-9]+)?)'."))
    ap.add_argument("--out", type=Path, required=True)
    add_wandb_args(ap, default_project="egta-analysis")
    args = ap.parse_args()

    pattern = re.compile(args.label_regex)
    dense_points = _terminal_delta_by_beta(args.dense, pattern)
    if not dense_points:
        print(f"[bifurcation_plot] no cf_b* labels found in {args.dense}", file=sys.stderr)
        sys.exit(2)
    dense_betas, dense_deltas = zip(*dense_points)

    cross_by_beta = {}  # beta -> list of delta_T values
    for path in args.cross_seed:
        pts = _terminal_delta_by_beta(path, pattern)
        for beta, delta in pts:
            cross_by_beta.setdefault(beta, []).append(delta)
    cross_betas = sorted(cross_by_beta.keys())
    cross_mean = np.array([np.mean(cross_by_beta[b]) for b in cross_betas])
    cross_std = np.array([np.std(cross_by_beta[b], ddof=0) for b in cross_betas])
    n_seeds = max(len(v) for v in cross_by_beta.values()) if cross_by_beta else 0

    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    ax.plot(dense_betas, dense_deltas, "o-", color="tab:blue", linewidth=1.7, markersize=6,
            label=fr"dense $\beta$ grid (K={len(dense_points)+1})")
    if cross_betas:
        ax.errorbar(cross_betas, cross_mean, yerr=cross_std, fmt="s", color="tab:red",
                    capsize=4, markersize=6, linewidth=1.2,
                    label=fr"4-$\beta$ cross-seed (n={n_seeds}), mean $\pm$ std")
    ax.axhline(args.tol, color="black", linestyle=":", alpha=0.7, label=fr"stability tol = {args.tol:g}")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$ (KL prior weight)")
    ax.set_ylabel(r"$\Delta_T(\beta) = M_T[\beta, \beta] - \min_j M_T[\beta, j]$")
    ax.set_title(r"Performative bifurcation: in-set deviation gain at terminal $t$, closed-form sweep")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    # Mark stable region.
    ax.fill_between(dense_betas, 0, args.tol, color="tab:green", alpha=0.10)
    fig.savefig(args.out, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print("[bifurcation_plot] dense-grid terminal Delta(beta):")
    for b, d in dense_points:
        flag = "stable" if d <= args.tol else ""
        print(f"  beta={b:>6}  Delta={d:.4f}  {flag}")
    if cross_betas:
        print("[bifurcation_plot] cross-seed mean+-std at terminal (different K-set, slightly different Delta):")
        for b, m, s in zip(cross_betas, cross_mean, cross_std):
            print(f"  beta={b:>6}  Delta={m:.4f} +- {s:.4f}")
    print(f"[bifurcation_plot] wrote {args.out}")

    wb = WandbSession(
        enabled=args.wandb,
        project=args.wandb_project,
        name=args.wandb_name or args.out.stem,
        config={
            "stage": "bifurcation_plot",
            "dense_npz": str(args.dense),
            "cross_seed_npzs": [str(p) for p in args.cross_seed],
            "tol": float(args.tol),
            "label_regex": args.label_regex,
        },
    )
    if wb.active:
        for beta, delta in dense_points:
            wb.log_metrics({"bif/beta": float(beta), "bif/delta_terminal": float(delta),
                            "bif/stable": int(delta <= args.tol)})
        wb.log_image(args.out, key="bif/figure")
        wb.finish()


if __name__ == "__main__":
    main()
