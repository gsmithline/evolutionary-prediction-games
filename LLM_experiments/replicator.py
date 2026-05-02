"""Discrete replicator loop for the K=3 ACSIncome EPG.

At each step t with population p_t we
  1. draw n examples from the mixture sum_k p_k D_k (one population sample
     per replicator step, mirroring S&R §6.3),
  2. fit the policy on the sample,
  3. evaluate per-state accuracy on the held-out test pool,
  4. step p_{t+1} = p_t · (acc + 1) / (p_t · (acc + 1)).

The trajectory is recorded as a long-form DataFrame (one row per (t, state))
so existing notebooks can plot the three Fig. 5 panels directly.
"""
from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .data import STATES, K, ACSIncome3State, sample_from_mixture
from .policies import SamplePack, TestPack


def _atomic_write_csv(rows: list, path: Path) -> None:
    """Overwrite path with rows, going through a tmp file + rename so a crash
    mid-write can never leave a half-formed CSV."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def _build_sample_pack(data: ACSIncome3State, pi_ref: dict, df, group_labels, within_state_idx) -> SamplePack:
    p1 = np.zeros(len(df), dtype=np.float64)
    for k, state in enumerate(STATES):
        mask = group_labels == k
        if mask.any():
            # Map within-train-DataFrame index to original state-row position,
            # then look up the position-indexed cached π_ref values.
            state_pos = data.train_pos[state][within_state_idx[mask]]
            p1[mask] = pi_ref[state]["full_p1"][state_pos]
    X = df[data.feature_cols].reset_index(drop=True)
    y = df[data.target_col].astype(int).values
    return SamplePack(X=X, y=y, pi_ref_p1=p1)


def _build_test_pack(data: ACSIncome3State, pi_ref: dict, state: str) -> TestPack:
    pool = data.test[state]
    test_pi_ref = pi_ref[state]["full_p1"][data.test_pos[state]]
    return TestPack(
        X=pool[data.feature_cols].reset_index(drop=True),
        y=pool[data.target_col].astype(int).values,
        pi_ref_p1=test_pi_ref,
    )


def run_replicator(
    policy,
    data: ACSIncome3State,
    pi_ref: dict,
    p0: np.ndarray,
    T: int,
    n_per_step: int,
    seed: int,
    on_step=None,
    incremental_csv: Path | str | None = None,
) -> pd.DataFrame:
    """Run the discrete replicator. Optionally invoke `on_step(record)` after each step.

    If `incremental_csv` is provided, the partial trajectory is flushed to that
    path after every round (atomic temp-file + rename). A crash at round k
    leaves a CSV containing rows for t=0..k-1 instead of nothing.

    `record` is a flat dict with scalar values suitable for wandb.log:
      t, p/CA, p/NY, p/TX, p/min, p/entropy,
      acc/CA, acc/NY, acc/TX, acc/overall,
      fitness/CA, fitness/NY, fitness/TX, step_seconds,
      sample_y_mean/CA, sample_y_mean/NY, sample_y_mean/TX  (per-state pos rate in sample),
      sample_count/CA, sample_count/NY, sample_count/TX,
      threshold (if policy exposes one).
    """
    incremental_path: Path | None = None
    if incremental_csv is not None:
        incremental_path = Path(incremental_csv)
        incremental_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    p = np.asarray(p0, dtype=np.float64)
    if abs(p.sum() - 1.0) > 1e-6:
        raise ValueError("p0 must sum to 1.")

    test_packs = {state: _build_test_pack(data, pi_ref, state) for state in STATES}

    rows = []
    for t in tqdm(range(T), desc="replicator", unit="round"):
        t0 = perf_counter()
        df, g, wsi = sample_from_mixture(data, p, n_per_step, rng)
        sample = _build_sample_pack(data, pi_ref, df, g, wsi)
        policy.fit(sample)

        accs = np.array([policy.score(test_packs[state]) for state in STATES])
        f = accs + 1.0
        denom = float(p @ f)
        p_next = (p * f) / denom
        dt = perf_counter() - t0

        # Per-state sample diagnostics (helps explain head/threshold drift).
        sample_count = np.zeros(len(STATES), dtype=np.int64)
        sample_y_mean = np.full(len(STATES), np.nan, dtype=np.float64)
        for k, state in enumerate(STATES):
            mask = g == k
            sample_count[k] = int(mask.sum())
            if sample_count[k] > 0:
                sample_y_mean[k] = float(sample.y[mask].mean())

        record = {"t": int(t), "step_seconds": float(dt)}
        eps = 1e-12
        record["p/min"] = float(p.min())
        record["p/entropy"] = float(-(p * np.log(p + eps)).sum())
        record["acc/overall"] = float((p * accs).sum())
        for k, state in enumerate(STATES):
            record[f"p/{state}"] = float(p[k])
            record[f"acc/{state}"] = float(accs[k])
            record[f"fitness/{state}"] = float(accs[k])
            record[f"sample_count/{state}"] = int(sample_count[k])
            record[f"sample_y_mean/{state}"] = float(sample_y_mean[k]) if sample_count[k] > 0 else float("nan")
        if hasattr(policy, "threshold"):
            record["threshold"] = float(policy.threshold)
        if on_step is not None:
            on_step(record)

        for k, state in enumerate(STATES):
            rows.append({
                "t": t, "state": state,
                "p": float(p[k]), "acc": float(accs[k]),
                "fitness": float(accs[k]),
                "sample_count": int(sample_count[k]),
                "step_seconds": dt,
            })
        p = p_next

        if incremental_path is not None:
            _atomic_write_csv(rows, incremental_path)

    # Final population row, no acc.
    for k, state in enumerate(STATES):
        rows.append({
            "t": T, "state": state, "p": float(p[k]),
            "acc": np.nan, "fitness": np.nan,
            "sample_count": 0, "step_seconds": np.nan,
        })

    if incremental_path is not None:
        _atomic_write_csv(rows, incremental_path)

    return pd.DataFrame(rows)
