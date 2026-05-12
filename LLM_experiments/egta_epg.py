"""EPG (ACSIncome / Saig-Rosenfeld 2025) adapter for the agnostic EGTA core.

Reads Fig.5 trajectory CSVs produced by run_fig5.py (columns: t, state, p, acc,
fitness, sample_count, step_seconds) and produces an EGTAInputs whose buckets
are the EPG groups (CA, NY, TX).

Why this maps cleanly: per-group accuracy depends only on the model + the fixed
per-group holdout, not on which world induced the model. So bucket_losses[j,t,b]
can be lifted from intervention j's own CSV.

Cross-intervention alignment: t-axes are intersected (run lengths may differ;
e.g. closed-form runs go to T=600 while SFT runs go to T=200). NaN t-steps
(typically the terminal row) are dropped.

CLI:
    python -m LLM_experiments.egta_epg \
        --csv /path/fig5_sft_b0_s0.csv --label sft_b0 \
        --csv /path/fig5_sft_b0.01_s0.csv --label sft_b0.01 \
        --csv /path/fig5_sft_b1_s0.csv --label sft_b1 \
        --csv /path/fig5_sft_b10_s0.csv --label sft_b10 \
        --out /path/egta_sft_sweep_s0.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from LLM_experiments.egta_matrix import (
    EGTAInputs,
    build_bundle,
    drop_nan_t,
    print_summary,
    save_npz,
)
from LLM_experiments.egta_wandb import WandbSession, add_wandb_args

EPG_GROUPS = ("CA", "NY", "TX")


def load_epg_csv(path: Path, bucket_order=EPG_GROUPS):
    """Return (t_axis, bucket_names, p_tb, acc_tb) from a Fig.5 trajectory CSV."""
    df = pd.read_csv(path)
    required = {"t", "state", "p", "acc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    pivot_p = df.pivot(index="t", columns="state", values="p")
    pivot_a = df.pivot(index="t", columns="state", values="acc")
    cols = list(bucket_order)
    missing_states = set(cols) - set(pivot_p.columns)
    if missing_states:
        raise ValueError(f"{path}: missing states {missing_states}")
    pivot_p = pivot_p[cols].sort_index()
    pivot_a = pivot_a[cols].sort_index()
    t_axis = pivot_p.index.to_numpy()
    return t_axis, tuple(cols), pivot_p.to_numpy(), pivot_a.to_numpy()


def build_epg_inputs(csv_paths, labels=None, bucket_order=EPG_GROUPS) -> EGTAInputs:
    """Load K EPG trajectory CSVs, intersect t-axes, return EGTAInputs."""
    paths = [Path(p) for p in csv_paths]
    if labels is None:
        labels = [p.stem for p in paths]
    if len(labels) != len(paths):
        raise ValueError(f"labels length {len(labels)} != csv count {len(paths)}")

    loaded = [load_epg_csv(p, bucket_order=bucket_order) for p in paths]
    _, buckets_ref, _, _ = loaded[0]
    for path, (_, buckets, _, _) in zip(paths[1:], loaded[1:]):
        if buckets != buckets_ref:
            raise ValueError(f"{path}: bucket order mismatch with {paths[0]}")

    common_t = loaded[0][0]
    for t_axis, _, _, _ in loaded[1:]:
        common_t = np.intersect1d(common_t, t_axis, assume_unique=True)
    if len(common_t) == 0:
        raise ValueError("no overlapping t between interventions")
    if any(len(loaded[i][0]) != len(common_t) for i in range(len(loaded))):
        print(f"[egta_epg] intersecting t-axes; common T={len(common_t)} "
              f"(individual lengths: {[len(loaded[i][0]) for i in range(len(loaded))]})")

    K = len(paths)
    T = len(common_t)
    B = len(buckets_ref)
    world_mixtures = np.empty((K, T, B), dtype=np.float64)
    bucket_losses = np.empty((K, T, B), dtype=np.float64)
    for i, (t_axis, _, p_tb, acc_tb) in enumerate(loaded):
        sel = np.isin(t_axis, common_t, assume_unique=True)
        world_mixtures[i] = p_tb[sel]
        bucket_losses[i] = 1.0 - acc_tb[sel]

    return EGTAInputs(
        world_mixtures=world_mixtures,
        bucket_losses=bucket_losses,
        t=common_t,
        labels=labels,
        buckets=list(buckets_ref),
        metadata={"adapter": "epg", "csv_paths": [str(p) for p in paths]},
    )


def _parse_args():
    ap = argparse.ArgumentParser(description="EPG adapter -> EGTA decoupled-PR tensor.")
    ap.add_argument("--csv", action="append", required=True, type=Path,
                    help="Trajectory CSV (Fig.5 schema). Repeat for K interventions.")
    ap.add_argument("--label", action="append", default=None,
                    help="Label for the matching --csv. Repeat in the same order.")
    ap.add_argument("--out", type=Path, required=True, help="Output .npz path.")
    add_wandb_args(ap, default_project="egta-analysis")
    return ap.parse_args()


def main():
    args = _parse_args()
    labels = args.label
    if labels is not None and len(labels) != len(args.csv):
        print(f"[egta_epg] --label count {len(labels)} != --csv count {len(args.csv)}", file=sys.stderr)
        sys.exit(2)
    inputs = build_epg_inputs(args.csv, labels=labels)
    inputs = drop_nan_t(inputs)
    bundle = build_bundle(inputs)
    # Preserve csv_paths in the saved bundle for traceability.
    bundle["csv_paths"] = np.asarray(inputs.metadata.get("csv_paths", []))
    save_npz(bundle, args.out)
    print(f"[egta_epg] wrote {args.out}")
    print_summary(bundle)

    wb = WandbSession(
        enabled=args.wandb,
        project=args.wandb_project,
        name=args.wandb_name or args.out.stem,
        config={
            "adapter": "epg",
            "labels": list(bundle["labels"]),
            "K": int(bundle["M"].shape[1]),
            "T": int(bundle["M"].shape[0]),
            "buckets": list(bundle["buckets"]),
            "csv_paths": list(bundle["csv_paths"]),
        },
    )
    if wb.active:
        M = bundle["M"]; K = M.shape[1]
        diag = np.einsum("tii->ti", M)
        for tt in range(M.shape[0]):
            slice_t = M[tt]
            off_t = (slice_t.sum() - diag[tt].sum()) / max(K * (K - 1), 1)
            wb.log_metrics({
                "egta/diag_min": float(diag[tt].min()),
                "egta/diag_max": float(diag[tt].max()),
                "egta/diag_mean": float(diag[tt].mean()),
                "egta/offdiag_mean": float(off_t),
            }, step=int(bundle["t"][tt]))
        diag_T = diag[-1]
        offT = (M[-1].sum() - diag_T.sum()) / max(K * (K - 1), 1)
        wb.log_metrics({
            "egta/terminal_diag_min": float(diag_T.min()),
            "egta/terminal_diag_max": float(diag_T.max()),
            "egta/terminal_diag_mean": float(diag_T.mean()),
            "egta/terminal_offdiag_mean": float(offT),
        })
        wb.log_artifact(args.out, artifact_type="egta-epg-tensor")
        wb.finish()


if __name__ == "__main__":
    main()
